"""Optional discord.py integration (extra: ``discord``).

One composition root: ``setup_discord_memory`` wires observation, rename
tracking, bot-self guards, and the ``/memory`` command group onto the bot.
The core library never imports this module — it is thin transport glue, so
``discord`` is imported at module level here (importing this module without
the extra installed is an error by design).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import discord
from discord import app_commands
from discord.ext import commands

from icelake.api.client import DiscordMemory
from icelake.config import MemoryConfig
from icelake.errors import SubjectNotAllowedError
from icelake.models.events import MessageEvent
from icelake.models.facts import FactCategory
from icelake.models.retrieval import PromptContext

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


async def prompt_from_message(
    memory: DiscordMemory,
    message: discord.Message,
    *,
    token_budget_tokens: int | None = None,
) -> PromptContext:
    """Reply-path hot call: map the message and build the injection block.

    Pair recall and discovery channels engage automatically when the message
    mentions other members.
    """
    event = _event_from_message(message)
    return await memory.prompt_context(
        guild_id=event.guild_id,
        asker_id=event.author_id,
        text=event.content,
        mentioned_ids=event.mention_ids,
        token_budget_tokens=token_budget_tokens,
    )


async def setup_discord_memory(
    bot: commands.Bot,
    config: MemoryConfig,
    **client_overrides: object,
) -> DiscordMemory:
    """Build the client, wire listeners + ``/memory`` commands onto ``bot``.

    Call from ``async def setup_hook()``. Returns the started-on-ready client;
    the cog is registered on the bot (``bot.cogs["MemoryCog"]``).
    """
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
        if before.display_name != after.display_name:
            await memory.identity.handle_member_rename(
                str(after.guild.id),
                str(after.id),
                after.display_name,
            )

    @bot.listen("on_ready")
    async def _on_ready() -> None:
        if bot.user is not None:
            memory.register_bot_id(bot.user.id)
        if not memory.started:
            await memory.start()

    await bot.add_cog(MemoryCog(memory))
    return memory


class _ConfirmPurgeView(discord.ui.View):
    """One-shot confirm for ``/memory forget`` — irreversible ops get a preview."""

    def __init__(
        self,
        memory: DiscordMemory,
        *,
        guild_id: str,
        user_id: str,
        preview_count: int,
    ) -> None:
        super().__init__(timeout=60)
        self._memory = memory
        self._guild_id = guild_id
        self._user_id = user_id
        self._preview_count = preview_count

    @discord.ui.button(label="Confirm purge", style=discord.ButtonStyle.danger)
    async def confirm(
        self,
        interaction: discord.Interaction,
        button: discord.ui.Button[_ConfirmPurgeView],
    ) -> None:
        del button
        if str(interaction.user.id) != self._user_id:
            await interaction.response.send_message(
                "Only the person who asked can confirm this.",
                ephemeral=True,
            )
            return
        await self._memory.admin.purge_user(self._guild_id, self._user_id, dry_run=False)
        await interaction.response.edit_message(
            content=f"Purged {self._preview_count} memories about you.",
            view=None,
        )
        self.stop()


class MemoryCog(commands.Cog):
    """``/memory me`` · ``/memory remember`` · ``/memory forget`` (two-phase)."""

    group = app_commands.Group(name="memory", description="What the bot remembers")

    def __init__(self, memory: DiscordMemory) -> None:
        self.memory = memory

    @group.command(name="me")
    async def me(self, interaction: discord.Interaction) -> None:
        """Show what the bot remembers about you."""
        page = await self.memory.facts.list_for_subject(
            str(interaction.guild_id),
            str(interaction.user.id),
            include_server=False,
            limit=25,
        )
        lines = [f"- {fact.text}" for fact in page.items] or ["Nothing yet."]
        await interaction.response.send_message(
            "**Your stored memories:**\n" + "\n".join(lines),
            ephemeral=True,
        )

    @group.command(name="remember")
    @app_commands.describe(text="A durable fact about yourself")
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
        await interaction.response.send_message(f"Noted: {text}", ephemeral=True)

    @group.command(name="forget")
    async def forget(self, interaction: discord.Interaction) -> None:
        """Preview what would be purged; a confirm button executes it."""
        guild_id = str(interaction.guild_id)
        user_id = str(interaction.user.id)
        preview = await self.memory.admin.purge_user(guild_id, user_id, dry_run=True)
        if preview.facts_removed == 0:
            await interaction.response.send_message(
                "I hold no memories of you.",
                ephemeral=True,
            )
            return
        await interaction.response.send_message(
            f"This will permanently purge **{preview.facts_removed}** memories "
            "about you. Confirm below — this cannot be undone.",
            view=_ConfirmPurgeView(
                self.memory,
                guild_id=guild_id,
                user_id=user_id,
                preview_count=preview.facts_removed,
            ),
            ephemeral=True,
        )


__all__ = ["MemoryCog", "prompt_from_message", "setup_discord_memory"]
