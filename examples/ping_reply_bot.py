"""A complete ping-reply Discord bot with passive user memory.

This mirrors the classic "chat bot" deployment (see ~/Github/CringeDiscordBot):

1. Every human message is passively learned in the background (fire-and-forget).
2. When someone pings the bot, the turn context resolves:
   - the asker's memories,
   - every referenced (@mentioned) user's memories,
   - community/server-wide facts,
   each clearly labeled so facts never bleed across users.
3. The model's echoed [mem:N] citation tags are resolved into jump links.
4. Natural-language commands ("remember that ...", "forget ...") work in chat.
5. Nickname changes re-index identity automatically.

Run:  DISCORD_TOKEN=... OPENROUTER_API_KEY=... python examples/ping_reply_bot.py
"""

from __future__ import annotations

import logging
import os

import discord
from discord.ext import commands

from discord_memory import (
    DiscordMemory,
    MemoryConfig,
    UserMemoryCommand,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("ping-bot")


def build_memory() -> DiscordMemory:
    """Compose the memory client. Swap storage/embeddings here for production."""
    return DiscordMemory(
        MemoryConfig(
            storage="sqlite:///bot-memory.db",
            llm=("openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=google/gemini-2.5-flash"),
            batching={"batch_size_messages": 8, "max_age_seconds": 180},
            budgets={"guild_daily_prompt_tokens": 200_000},
        )
    )


class PingReplyBot(commands.Bot):
    """Replies only when pinged; learns from everyone all the time."""

    def __init__(self, memory: DiscordMemory) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix="!", intents=intents)
        self.memory = memory

    async def setup_hook(self) -> None:
        await self.memory.start()
        await self.tree.sync()

    async def close(self) -> None:
        await self.memory.close(drain=True)
        await super().close()

    # ------------------------------------------------------------------ #
    # Passive learning: observe EVERY human message, never block the     #
    # response path. Extraction happens on background workers.           #
    # ------------------------------------------------------------------ #

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None or message.author.bot:
            return
        receipt = await self.memory.observe(
            _to_event(message, bot_id=self.user.id if self.user else 0),
        )
        log.debug("observed %s -> %s", message.id, receipt.status.value)

        # A ping means it's a turn: answer it.
        if self.user and self.user in message.mentions:
            await self._reply_to_ping(message)

        # Natural-language memory commands ("remember that ...").
        await self._maybe_handle_memory_command(message)

    # ------------------------------------------------------------------ #
    # Identity upkeep: nicknames change; aliases must follow.            #
    # ------------------------------------------------------------------ #

    @commands.Cog.listener()
    async def on_member_update(self, before: discord.Member, after: discord.Member) -> None:
        if before.display_name != after.display_name:
            await self.memory.identity.handle_member_rename(
                str(after.guild.id),
                str(after.id),
                after.display_name,
            )

    # ------------------------------------------------------------------ #
    # The reply path: resolve memories, build context, generate.         #
    # ------------------------------------------------------------------ #

    async def _reply_to_ping(self, message: discord.Message) -> None:
        assert self.user is not None
        question = _strip_mention(message.content, self.user.id).strip()
        if not question:
            await message.reply("Yes?", mention_author=False)
            return

        guild_id = str(message.guild.id) if message.guild else "0"
        mentioned_ids = tuple(
            str(member.id) for member in message.mentions if member.id != self.user.id
        )

        # ONE CALL resolves: asker profile, referenced-user profiles, server facts.
        ctx = await self.memory.prompt_context(
            guild_id=guild_id,
            asker_id=str(message.author.id),
            text=question,
            mentioned_ids=mentioned_ids,
            token_budget_tokens=700,
        )
        if ctx.warnings:
            log.info("turn warnings: %s", [w.value for w in ctx.warnings])

        history = await _recent_channel_history(message, limit=6)
        system_prompt = (
            "You are a helpful community bot. Use the labeled MEMORY CONTEXT "
            "below. Facts belong ONLY to the person named in their header.\n\n"
            f"{ctx.injection_block}"
        )
        reply_text = await _generate(system_prompt, history, question)

        # Resolve any echoed [mem:N] tags into Discord jump links.
        reply_text = ctx.apply_citations(reply_text)
        if not reply_text.strip():
            reply_text = "I don't have anything useful to add yet!"
        await message.reply(reply_text[:1900], mention_author=False)

    # ------------------------------------------------------------------ #
    # Chat-native memory commands, ChatGPT style.                        #
    # ------------------------------------------------------------------ #

    async def _maybe_handle_memory_command(self, message: discord.Message) -> None:
        assert self.user is not None
        if self.user in message.mentions:
            return  # ping turns are answered above; avoid double-handling
        command = await self.memory.classify_command(message.content)
        if command.action == "none":
            return
        await self._execute_memory_command(message, command)

    async def _execute_memory_command(
        self,
        message: discord.Message,
        command: UserMemoryCommand,
    ) -> None:
        guild_id = str(message.guild.id) if message.guild else "0"
        user_id = str(message.author.id)

        if command.action.value == "remember" and command.target_text:
            fact = await self.memory.facts.remember(
                guild_id=guild_id,
                subject_id=user_id,
                text=command.target_text,
                actor_id=user_id,
            )
            confirm = f"Got it — noted: “{fact.text}”"
        elif command.action.value == "forget":
            page = await self.memory.facts.list_for_subject(
                guild_id,
                user_id,
                include_server=False,
                limit=25,
            )
            matches = (
                [fact for fact in page.items if command.target_text.lower() in fact.text.lower()]
                if command.target_text
                else list(page.items)[:1]
            )
            if not matches:
                confirm = "I couldn't find a matching memory to forget."
            else:
                for fact in matches[:3]:
                    await self.memory.facts.forget(
                        fact.id,
                        guild_id=guild_id,
                        reason="user request",
                        actor_id=user_id,
                    )
                confirm = f"Forgot {len(matches)} memory(ies) you asked about."
        else:
            return  # query/update intents: leave to your normal reply path

        await message.reply(confirm, mention_author=False)


# ---------------------------------------------------------------------- #
# discord.py <-> library adapters (thin, consumer-side helpers).         #
# ---------------------------------------------------------------------- #


def _to_event(message: discord.Message, *, bot_id: int):
    from datetime import UTC

    from discord_memory.models.events import MessageEvent

    created = message.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    return MessageEvent(
        message_id=str(message.id),
        guild_id=str(message.guild.id) if message.guild else "0",
        channel_id=str(message.channel.id),
        author_id=str(message.author.id),
        content=message.content or "",
        created_at=created.astimezone(UTC),
        author_username=getattr(message.author, "name", "") or "",
        author_display_name=getattr(message.author, "display_name", "") or str(message.author),
        author_is_bot=bool(message.author.bot),
        mention_ids=tuple(str(m.id) for m in message.mentions),
        reply_to_message_id=(str(message.reference.message_id) if message.reference else None),
    )


def _strip_mention(content: str, bot_user_id: int) -> str:
    import re

    return re.sub(rf"<@!?{bot_user_id}>", "", content)


async def _recent_channel_history(message: discord.Message, *, limit: int):
    """Hydrate the recent channel window as simple (author, content) turns."""
    turns: list[tuple[str, str]] = []
    try:
        async for msg in message.channel.history(limit=limit, before=message):
            if not msg.author.bot:
                turns.append((msg.author.display_name, msg.content))
    except (discord.HTTPException, discord.Forbidden):
        pass
    turns.reverse()
    return turns


async def _generate(system_prompt: str, history, question: str) -> str:
    """Call YOUR LLM here (OpenAI-compatible chat completions).

    Kept provider-agnostic on purpose: the library owns memory; this is where
    your existing generation stack plugs in.
    """
    raise NotImplementedError(
        "wire your LLM call here: system_prompt + history + question -> reply text",
    )


def main() -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("set DISCORD_TOKEN")
    bot = PingReplyBot(build_memory())

    @bot.command(name="mystats")
    async def my_stats(ctx: commands.Context) -> None:
        stats = await bot.memory.stats(str(ctx.guild.id))
        await ctx.reply(
            f"I hold {stats.active_facts} active memories across {stats.user_count} members.",
            mention_author=False,
        )

    bot.run(token)


if __name__ == "__main__":
    main()
