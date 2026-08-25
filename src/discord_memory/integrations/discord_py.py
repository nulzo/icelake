"""Optional discord.py integration (extra: ``discord``).

Wires listeners and provides a slash-command cog. Core library never imports this
module — it exists purely as thin transport glue (PLAN.md §6.3).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import discord
    from discord.ext import commands

from discord_memory.api.client import DiscordMemory
from discord_memory.config import MemoryConfig
from discord_memory.errors import SubjectNotAllowedError
from discord_memory.models.events import MessageEvent
from discord_memory.models.facts import FactCategory

logger = logging.getLogger(__name__)


def _event_from_message(message: discord.Message) -> MessageEvent:
    guild = message.guild
    return MessageEvent(
        message_id=str(message.id),
        guild_id=str(guild.id) if guild else "0",
        channel_id=str(message.channel.id),
        author_id=str(message.author.id),
        content=message.content or "",
        created_at=message.created_at.astimezone(UTC)
        if message.created_at.tzinfo
        else datetime.now(UTC),
        author_username=getattr(message.author, "name", "") or "",
        author_display_name=getattr(message.author, "display_name", "") or str(message.author),
        author_is_bot=bool(message.author.bot),
        mention_ids=tuple(str(m.id) for m in message.mentions if not m.bot),
        reply_to_message_id=(str(message.reference.message_id) if message.reference else None),
        thread_parent_id=_thread_parent(message),
    )


def _thread_parent(message: discord.Message) -> str | None:
    thread = getattr(message, "thread", None)
    if thread is not None and getattr(thread, "parent_id", None) is not None:
        return str(thread.parent_id)
    return None


async def setup_discord_memory(
    bot: commands.Bot,
    config: MemoryConfig,
    **client_overrides: object,
) -> tuple[DiscordMemory, MemoryCog]:
    """Build the client, wire listeners onto ``bot``, and return ``(memory, cog)``.

    Call from ``async def setup_hook()``; add the returned cog with
    ``await bot.add_cog(cog)``.
    """
    import discord  # noqa: F401 - ensures the extra is installed

    from discord_memory.api.client import DiscordMemory

    memory = DiscordMemory(config, **client_overrides)

    @bot.listen("on_message")
    async def _on_message(message: discord.Message) -> None:
        if message.guild is None:
            return
        try:
            await memory.observe(_event_from_message(message))
        except Exception:
            logger.exception("observe failed for message %s", message.id)

    @bot.listen("on_member_update")
    async def _on_member_update(before: discord.Member, after: discord.Member) -> None:
        before_name = before.display_name
        after_name = after.display_name
        if before_name != after_name:
            await memory.identity.handle_member_rename(
                str(after.guild.id),
                str(after.id),
                after_name,
            )

    @bot.listen("on_ready")
    async def _on_ready() -> None:
        if bot.user is not None:
            memory._guard.register(str(bot.user.id))
        if not memory.started:
            await memory.start()

    cog = MemoryCog(memory)
    return memory, cog


class MemoryCog:
    """Slash commands: ``/memory me``, ``/memory remember``, owner purge/export."""

    def __init__(self, memory: DiscordMemory) -> None:
        self.memory = memory

    async def me(self, interaction: discord.Interaction) -> None:
        """Show what the bot remembers about you."""
        user_id = str(interaction.user.id)
        guild_id = str(interaction.guild_id)
        page = await self.memory.facts.list_for_subject(
            guild_id,
            user_id,
            include_server=False,
            limit=25,
        )
        lines = [f"- {fact.text}" for fact in page.items] or ["Nothing yet."]
        await interaction.response.send_message(
            "**Your stored memories:**\n" + "\n".join(lines),
            ephemeral=True,
        )

    async def remember(self, interaction: discord.Interaction, text: str) -> None:
        """Teach the bot a durable fact about yourself."""
        try:
            await self.memory.facts.remember(
                guild_id=str(interaction.guild_id),
                subject_id=str(interaction.user.id),
                text=text,
                category=FactCategory.GENERAL,
                actor_id=str(interaction.user.id),
            )
        except SubjectNotAllowedError:
            await interaction.response.send_message(
                "You are opted out of memory.",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            f"Noted: {text}",
            ephemeral=True,
        )

    async def forget_me(self, interaction: discord.Interaction) -> None:
        """Purge everything the bot remembers about you (two-phase)."""
        report = await self.memory.admin.purge_user(
            str(interaction.guild_id),
            str(interaction.user.id),
            dry_run=True,
        )
        if not report.dry_run:
            pass
        confirmed = await self.memory.admin.purge_user(
            str(interaction.guild_id),
            str(interaction.user.id),
            dry_run=False,
        )
        del confirmed, report
        await interaction.response.send_message(
            "All memories about you have been purged.",
            ephemeral=True,
        )


__all__ = ["MemoryCog", "setup_discord_memory"]
