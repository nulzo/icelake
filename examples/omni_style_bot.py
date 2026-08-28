"""A full "omni-style" Discord bot with discord-memory as its memory layer.

Mirrors the architecture of a production memory-native bot:

  composition root      one place wires config -> adapters -> services
  persist everything    every message (bots included) is stored for citations
  addressing            replies when pinged OR when someone replies to it
  requester-first       turn context = asker + mentions/reply-targets, capped
  two-verb seam         observe()/recall()/prompt_context(); nothing else touches storage
  identity truth        Discord user IDs; names resolve through the alias ladder
                        (UNIQUE / AMBIGUOUS / UNKNOWN — ambiguity never guesses)
  coreference           multi-name members get an explicit "one person" line
  unfakeable citations  [mem:N] tags resolved to jump links; strays stripped
  governance            opt-out, purge, budgets, health — first-class

Slash commands mirror a /memory group: show, related, shared, edit, alias.

Run:
    DISCORD_TOKEN=... OPENROUTER_API_KEY=... python examples/omni_style_bot.py
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC

import discord
from discord import app_commands
from discord.ext import commands

from discord_memory import (
    DiscordMemory,
    MemoryConfig,
    MessageEvent,
)
from discord_memory.adapters.llm_openai_compat import OpenAICompatLLM, build_chat_llm
from discord_memory.config import LlmConfig
from discord_memory.ports.llm import ChatRequest, LlmMessage

logging.basicConfig(level=logging.INFO)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("discord").setLevel(logging.WARNING)
log = logging.getLogger("omni-style")

MAX_CONTEXT_SUBJECTS = 4
TURN_TOKEN_BUDGET = 800
LLM_URL = "openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=google/gemini-3.7-flash"
EMBEDDINGS_URL = (
    "openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=openai/text-embedding-3-small"
)


# --------------------------------------------------------------------------- #
# Composition root — the only place anything is wired together.               #
# --------------------------------------------------------------------------- #


def build_memory() -> DiscordMemory:
    # Hosted embeddings for reconcile + recall. If this DB was created with the
    # default hashing embedder, delete dm_vectors or start a fresh sqlite file.
    return DiscordMemory(
        MemoryConfig(
            storage="sqlite:///omni-style.db",
            llm=LLM_URL,
            embeddings=EMBEDDINGS_URL,
            batching={"batch_size_messages": 12, "max_age_seconds": 90},
            extraction={"auto_consolidate_after_adds": 6},
            retrieval={"default_token_budget": TURN_TOKEN_BUDGET},
            budgets={"guild_daily_prompt_tokens": 150_000},
        )
    )


class OmniStyleBot(commands.Bot):
    """Memory-native bot: passive learning for everyone, answers when addressed."""

    def __init__(self, memory: DiscordMemory) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix="!", intents=intents)
        self.memory = memory

    async def setup_hook(self) -> None:
        # Start memory BEFORE the gateway connects so early messages are safe.
        await self.memory.start()
        if self.user:
            self.memory.register_bot_id(self.user.id)  # never remember ourselves
        await self.tree.sync()

    async def close(self) -> None:
        await self.memory.close(drain=True)
        await super().close()

    # ------------------------------------------------------------------ #
    # on_message: learn from everyone; answer whoever addresses us.       #
    # ------------------------------------------------------------------ #

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return

        event = to_event(message)
        # Register our own id once; the bot is structurally never a subject.
        await self.memory.observe(event)
        if message.author.bot:
            return

        if await self._is_addressed(message):
            await self._handle_turn(message)

    # ------------------------------------------------------------------ #
    # Addressing: direct mention, <@id> token, or reply-to-our-message.   #
    # ------------------------------------------------------------------ #

    async def _is_addressed(self, message: discord.Message) -> bool:
        if self.user and self.user in message.mentions:
            return True
        if message.reference and isinstance(
            message.reference.resolved,
            discord.Message,
        ):
            author = message.reference.resolved.author
            return bool(self.user) and author.id == self.user
        return False

    # ------------------------------------------------------------------ #
    # The turn: resolve participants -> recall context -> generate.       #
    # ------------------------------------------------------------------ #

    async def _handle_turn(self, message: discord.Message) -> None:
        assert self.user is not None
        question = strip_bot_mention(message.content, self.user.id).strip()
        if not question:
            await message.reply("Yes?", mention_author=False)
            return

        guild_id = str(message.guild.id) if message.guild else "0"
        subject_ids = await self._collect_subjects(message)

        ctx = await self.memory.prompt_context(
            guild_id=guild_id,
            asker_id=str(message.author.id),
            text=question,
            mentioned_ids=tuple(subject_ids),
            token_budget_tokens=TURN_TOKEN_BUDGET,
        )
        for warning in ctx.warnings:
            log.info("turn warning: %s", warning.value)

        system_prompt = (
            "You are a memory-native community bot. Ground every claim about a "
            "member in the labeled MEMORY CONTEXT. Facts belong ONLY to the "
            "person named in their header; coreference lines tell you which "
            f"names are the same person.\n\n{ctx.injection_block}"
        )
        history = await recent_history(message, limit=6)
        reply = await generate(system_prompt, history, question)

        reply = ctx.apply_citations(reply)  # echo tags -> jump links
        if not reply.strip():
            reply = "I don't know enough about that yet."
        await message.reply(reply[:1900], mention_author=False)

    async def _collect_subjects(self, message: discord.Message) -> list[str]:
        """Requester-first participant collection, capped (omni pattern)."""
        subjects: list[str] = []
        for member in message.mentions:
            if member.bot or member.id == self.user:
                continue
            subjects.append(str(member.id))
        # Reply target counts as a referenced person too.
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            other = message.reference.resolved.author
            if not other.bot and str(other.id) not in subjects:
                subjects.append(str(other.id))
        return subjects[:MAX_CONTEXT_SUBJECTS]

    # ------------------------------------------------------------------ #
    # /memory group — mirrors omni's five ops on this library's APIs.     #
    # ------------------------------------------------------------------ #

    group_memory = app_commands.Group(name="memory", description="What the bot remembers")

    @group_memory.command(name="show")  # type: ignore[arg-type]
    async def memory_show(
        self, interaction: discord.Interaction, user: discord.Member | None = None
    ) -> None:
        """Show what the bot remembers about someone."""
        member = user or interaction.user
        page = await self.memory.facts.list_for_subject(
            str(interaction.guild_id),
            str(member.id),
            include_server=True,
            limit=15,
        )
        lines = [f"• {fact.text}" for fact in page.items] or ["Nothing yet."]
        aliases = await self.memory.identity.aliases_of(str(interaction.guild_id), str(member.id))
        alias_line = ""
        names = sorted({record.alias_norm for record in aliases})
        if len(names) > 1:
            alias_line = f"\nAlso known as: {', '.join(names)}"
        await interaction.response.send_message(
            f"**Memories for {member.display_name}**{alias_line}\n" + "\n".join(lines),
            ephemeral=True,
        )

    @group_memory.command(name="related")  # type: ignore[arg-type]
    async def memory_related(self, interaction: discord.Interaction, user: discord.Member) -> None:
        """Typed relationship edges touching a member."""
        edges = await self.memory.graph.relations_of(
            str(interaction.guild_id),
            str(user.id),
            limit=15,
        )
        if not edges:
            await interaction.response.send_message("No relations yet.", ephemeral=True)
            return
        lines = [
            f"• {edge.src_id} —{edge.verb}→ {edge.dst_id} ({edge.polarity.value})" for edge in edges
        ]
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @group_memory.command(name="shared")  # type: ignore[arg-type]
    async def memory_shared(
        self, interaction: discord.Interaction, a: discord.Member, b: discord.Member
    ) -> None:
        """What two members share (entities both touch)."""
        guild_id = str(interaction.guild_id)
        a_entities = {
            edge.dst_id
            for edge in await self.memory.graph.relations_of(guild_id, str(a.id), limit=200)
            if edge.dst_type.value == "entity"
        }
        b_edges = await self.memory.graph.relations_of(guild_id, str(b.id), limit=200)
        shared = sorted(
            {edge.dst_id for edge in b_edges if edge.dst_type.value == "entity"} & a_entities
        )
        body = ", ".join(shared) if shared else "nothing notable yet"
        await interaction.response.send_message(
            f"{a.display_name} and {b.display_name} share: {body}",
            ephemeral=True,
        )

    @group_memory.command(name="edit")  # type: ignore[arg-type]
    async def memory_edit(self, interaction: discord.Interaction, correction: str) -> None:
        """Teach a durable fact about yourself (jumps the learning queue)."""
        fact = await self.memory.facts.remember(
            guild_id=str(interaction.guild_id),
            subject_id=str(interaction.user.id),
            text=correction,
            actor_id=str(interaction.user.id),
        )
        await interaction.response.send_message(
            f"Noted: “{fact.text}”",
            ephemeral=True,
        )

    @group_memory.command(name="alias")  # type: ignore[arg-type]
    async def memory_alias(self, interaction: discord.Interaction, alias: str) -> None:
        """Teach the bot another name for you."""
        await self.memory.identity.register_alias(
            str(interaction.guild_id),
            str(interaction.user.id),
            alias,
        )
        await interaction.response.send_message(
            f"Got it — I'll also know you as “{alias}”.",
            ephemeral=True,
        )


# --------------------------------------------------------------------------- #
# Governance commands.                                                        #
# --------------------------------------------------------------------------- #


def register_governance(bot: OmniStyleBot) -> None:
    @bot.command(name="forgetme")
    async def forgetme(ctx: commands.Context) -> None:
        assert ctx.guild is not None
        report = await bot.memory.admin.purge_user(
            str(ctx.guild.id),
            str(ctx.author.id),
            dry_run=False,
        )
        await ctx.reply(
            f"Purged {report.facts_removed} facts, {report.aliases_removed} aliases about you.",
            mention_author=False,
        )

    @bot.command(name="optout")
    async def optout(ctx: commands.Context) -> None:
        assert ctx.guild is not None
        await bot.memory.admin.set_opt_out(str(ctx.guild.id), str(ctx.author.id), True)
        await ctx.reply("You're opted out — I'll stop observing you.", mention_author=False)


# --------------------------------------------------------------------------- #
# Pure helpers.                                                               #
# --------------------------------------------------------------------------- #


def to_event(message: discord.Message) -> MessageEvent:
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


def strip_bot_mention(content: str, bot_user_id: int) -> str:
    return re.sub(rf"<@!?{bot_user_id}>", "", content)


async def recent_history(message: discord.Message, *, limit: int):
    turns: list[tuple[str, str]] = []
    try:
        async for msg in message.channel.history(limit=limit, before=message):
            if not msg.author.bot:
                turns.append((msg.author.display_name, msg.content))
    except (discord.HTTPException, discord.Forbidden):
        pass
    turns.reverse()
    return turns


_reply_llm: OpenAICompatLLM | None = None


def _reply_llm_client() -> OpenAICompatLLM:
    global _reply_llm
    if _reply_llm is None:
        _reply_llm = build_chat_llm(LlmConfig.from_url(LLM_URL))
    return _reply_llm


async def generate(system_prompt: str, history, question: str) -> str:
    messages: list[LlmMessage] = [LlmMessage(role="system", content=system_prompt)]
    for author, content in history:
        if content.strip():
            messages.append(LlmMessage(role="user", content=f"{author}: {content}"))
    messages.append(LlmMessage(role="user", content=question))
    response = await _reply_llm_client().complete(
        ChatRequest(
            messages=tuple(messages),
            max_tokens=900,
            purpose="reply",
        )
    )
    return response.text.strip()


def main() -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("set DISCORD_TOKEN")

    bot = OmniStyleBot(build_memory())
    register_governance(bot)

    bot.run(token)


if __name__ == "__main__":
    main()
