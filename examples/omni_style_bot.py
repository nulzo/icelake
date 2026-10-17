"""A full "omni-style" Discord bot with icelake as its memory layer.

Mirrors the architecture of a production memory-native bot:

  composition root      one place wires config -> adapters -> services
  persist everything    every message (bots included) is stored for citations
  addressing            replies when pinged OR when someone replies to it
  requester-first       turn context = asker + mentions/reply-targets, capped
  two-verb seam         observe()/recall()/prompt_context(), nothing else touches storage
  identity truth        current Discord user IDs; ambiguous names require a mention
  name-in-prose lookup  "what do you know about klim?" (no @mention) resolves
                        via Discord membership into audience-authorized recall,
                        from /memory lookup (zero LLM) or one router call
  grounded citations    the model writes plain prose, then memory.cite weaves
                        jump links where the reply matches a registered source
  governance            opt-out, purge, budgets, health: all first-class

Slash commands: /memory show, lookup, edit. Graph inspection stays in trusted tooling.

Run:
    DISCORD_TOKEN=... OPENROUTER_API_KEY=... python examples/omni_style_bot.py
"""

from __future__ import annotations

import logging
import os
import re
from datetime import UTC
from enum import StrEnum

import discord
from discord import app_commands
from discord.ext import commands
from pydantic import BaseModel, ValidationError

from icelake import (
    Audience,
    ChatRequest,
    DiscordMemory,
    LlmMessage,
    MemoryConfig,
    MessageEvent,
    MessageRole,
    RecallQuery,
)
from icelake.adapters.llm_openai_compat import OpenAICompatLLM, build_chat_llm
from icelake.config import LlmConfig
from icelake.integrations.discord_permissions import DiscordChannelAuthorizer

logging.basicConfig(level=logging.INFO)
# logging.getLogger("httpx").setLevel(logging.WARNING)
# logging.getLogger("discord").setLevel(logging.WARNING)
log = logging.getLogger("omni-style")

MAX_CONTEXT_SUBJECTS = 4
TURN_TOKEN_BUDGET = 2400
# Benchmark winner (see the README model table): best extraction quality at
# about a cent per full eval run. reasoning=low is the cheapest allowed effort.
LLM_URL = "openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=z-ai/glm-5.3-flash&reasoning=low"
EMBEDDINGS_URL = (
    "openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=openai/text-embedding-3-small"
)

# prompt_context scopes people from structured mentions and reply targets
# only. It never scans prose for names (recall is zero-LLM by design). For
# "what do you know about klim?" typed as prose, one cheap structured-output
# call decides whether a strict profile lookup is needed and extracts the
# name. response_schema is enforced server-side on capable providers. On
# others it degrades to JSON mode (LlmConfig.structured_outputs).
_ROUTER_PROMPT = (
    "Decide what this message needs. If it asks what you know about a specific "
    "server member who is NOT @mentioned in the message, answer action="
    '"memory_show" with user set to their name exactly as written. For '
    'anything else, action="answer".'
)


class RouteAction(StrEnum):
    ANSWER = "answer"
    MEMORY_SHOW = "memory_show"


class RouteDecision(BaseModel):
    action: RouteAction
    user: str = ""


_USER_REF_RE = re.compile(r"<@!?(?P<snowflake>\d+)>")


# =-------------------------------------------------------------------------= #
# Composition root: the only place anything is wired together                 #
# =-------------------------------------------------------------------------= #


def build_memory(client: discord.Client) -> DiscordMemory:
    return DiscordMemory(
        MemoryConfig(
            storage="sqlite:///omni-style.db",
            llm=LLM_URL,
            embeddings=EMBEDDINGS_URL,
            batching={"batch_size_messages": 12, "max_age_seconds": 90},
            extraction={"auto_consolidate_after_adds": 6},
            retrieval={"default_token_budget": TURN_TOKEN_BUDGET},
            budgets={"guild_daily_prompt_tokens": 150_000},
        ),
        channel_authorizer=DiscordChannelAuthorizer(client),
    )


class OmniStyleBot(commands.Bot):
    """simple mem-native bot with passive learning for everyone, answers when addressed."""

    def __init__(self, memory: DiscordMemory | None = None) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(command_prefix="!", intents=intents)
        self.memory = memory if memory is not None else build_memory(self)

    async def setup_hook(self) -> None:
        # Start memory before the gateway connects so early messages are safe
        await self.memory.start()
        if self.user:
            self.memory.register_bot_id(self.user.id)  # never remember ourselves
        await self.tree.sync()

    async def close(self) -> None:
        await self.memory.close(drain=True)
        await super().close()

    # =----------------------------------------------------------------= #
    # on_message: learn from everyone and answer whoever addresses us    #
    # =----------------------------------------------------------------= #

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return

        event = to_event(message)
        # register our own id once. the bot should never be a subject.
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
            return bool(self.user) and author.id == self.user.id
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
        mentioned, thread = self._collect_subjects(message)

        # Resolve names through Discord membership, then use the same authorized
        # context/citation path as explicit mentions.
        lookup_id = await self._route_name_lookup(guild_id, question)
        if lookup_id and lookup_id not in mentioned and len(mentioned) < MAX_CONTEXT_SUBJECTS:
            mentioned.append(lookup_id)
            thread = thread[: MAX_CONTEXT_SUBJECTS - len(mentioned)]

        ctx = await self.memory.prompt_context(
            guild_id=guild_id,
            asker_id=str(message.author.id),
            audience=Audience(
                channel_id=str(message.channel.id), requester_id=str(message.author.id)
            ),
            text=question,
            mentioned_ids=tuple(mentioned),
            thread_participant_ids=tuple(thread),
            token_budget_tokens=TURN_TOKEN_BUDGET,
        )
        for warning in ctx.warnings:
            log.info("turn warning: %s", warning.value)

        system_prompt = (
            "You are a memory-native community bot. Ground every claim about a "
            "member in the labeled MEMORY CONTEXT. Facts belong ONLY to the "
            f"person named in their header.\n\n{ctx.injection_block}"
        )
        history: list[tuple[str, str]] = []
        reply = await generate(system_prompt, history, question)

        # Reattach provenance post-generation: deterministic embedder scoring
        # (no LLM call) weaves jump links at the spans sources support. The
        # answer model never sees tags or URLs. Custom footers, mixed web
        # sources, and used-only numbering: examples/citations.py.
        cited = await self.memory.cite(reply, ctx)
        reply = cited.text
        if not reply.strip():
            reply = "I don't know enough about that yet."
        await message.reply(reply[:1900], mention_author=False)

    def _collect_subjects(self, message: discord.Message) -> tuple[list[str], list[str]]:
        """Mentions vs reply-chain participants. Icelake pair-intersects both."""
        mentioned: list[str] = []
        for member in message.mentions:
            if member.bot or member.id == self.user.id:
                continue
            mentioned.append(str(member.id))
        thread: list[str] = []
        if message.reference and isinstance(message.reference.resolved, discord.Message):
            other = message.reference.resolved.author
            if not other.bot and str(other.id) not in mentioned:
                thread.append(str(other.id))
        cap = MAX_CONTEXT_SUBJECTS
        mentioned = mentioned[:cap]
        thread = thread[: max(0, cap - len(mentioned))]
        return mentioned, thread

    # ------------------------------------------------------------------ #
    # Name-in-prose lookup: "what do you know about klim?" (no @mention). #
    # Every entry point (/memory lookup with zero LLM, the structured-    #
    # output router below, or a native tool call) converges on            #
    # current Discord membership and audience-authorized recall.        #
    # ------------------------------------------------------------------ #

    async def _route_name_lookup(self, guild_id: str, question: str) -> str | None:
        """One structured-output call: answer directly, or look someone up by name."""
        response = await _reply_llm_client().complete(
            ChatRequest(
                messages=(
                    LlmMessage(role=MessageRole.SYSTEM, content=_ROUTER_PROMPT),
                    LlmMessage(role=MessageRole.USER, content=question),
                ),
                response_schema=RouteDecision.model_json_schema(),
                max_tokens=60,
                purpose="reply_router",
            )
        )
        try:
            route = RouteDecision.model_validate_json(response.text)
        except ValidationError:
            return None  # non-JSON reply: fall through to a plain answer
        if route.action is not RouteAction.MEMORY_SHOW:
            return None
        name = route.user.strip()
        if not name:
            return None
        resolved = await self._resolve_lookup(guild_id, name)
        return resolved[0] if isinstance(resolved, tuple) else None

    async def _resolve_lookup(self, guild_id: str, name: str) -> tuple[str, str] | str:
        """Resolve current Discord members without exposing private memory aliases."""
        guild = self.get_guild(int(guild_id))
        if guild is None:
            return "Guild unavailable"
        ref = name.strip()
        if match := _USER_REF_RE.fullmatch(ref):
            ref = match.group("snowflake")
        if ref.isdecimal():
            try:
                member = await guild.fetch_member(int(ref))
            except discord.HTTPException:
                return "Member unavailable"
            return str(member.id), member.display_name
        candidates = [
            member
            for member in guild.members
            if ref.casefold() in {member.name.casefold(), member.display_name.casefold()}
        ]
        if len(candidates) != 1:
            return "Please @mention one current member to identify them unambiguously"
        return str(candidates[0].id), candidates[0].display_name

    group_memory = app_commands.Group(name="memory", description="What the bot remembers")

    @group_memory.command(name="show")  # type: ignore[arg-type]
    async def memory_show(
        self, interaction: discord.Interaction, user: discord.Member | None = None
    ) -> None:
        """Show what the bot remembers about someone."""
        member = user or interaction.user
        result = await self.memory.recall(
            RecallQuery(
                guild_id=str(interaction.guild_id),
                subject_ids=(str(member.id),),
                top_k=15,
                max_per_subject=15,
                audience=Audience(
                    channel_id=str(interaction.channel_id),
                    requester_id=str(interaction.user.id),
                ),
            )
        )
        lines = [
            f"• {item.fact.text}" for item in result.facts if item.fact.subject_id == str(member.id)
        ] or ["Nothing available here."]
        await interaction.response.send_message(
            f"**Memories for {member.display_name}**\n" + "\n".join(lines),
            ephemeral=True,
        )

    @group_memory.command(name="lookup")  # type: ignore[arg-type]
    async def memory_lookup(self, interaction: discord.Interaction, name: str) -> None:
        """Look someone up by a typed name. No member picker, no LLM."""
        guild_id = str(interaction.guild_id)
        result = await self._resolve_lookup(guild_id, name)
        if isinstance(result, str):
            await interaction.response.send_message(f"{result}.", ephemeral=True)
            return
        user_id, display = result
        result = await self.memory.recall(
            RecallQuery(
                guild_id=guild_id,
                subject_ids=(user_id,),
                top_k=15,
                max_per_subject=15,
                audience=Audience(
                    channel_id=str(interaction.channel_id),
                    requester_id=str(interaction.user.id),
                ),
            )
        )
        lines = [
            f"• {item.fact.text}" for item in result.facts if item.fact.subject_id == user_id
        ] or ["Nothing available here."]
        await interaction.response.send_message(
            f"**Memories for {display}**\n" + "\n".join(lines),
            ephemeral=True,
        )

    # Aggregate graph edges do not carry an audience boundary. Keep graph
    # inspection in trusted tooling (see relationship_queries.py).

    @group_memory.command(name="edit")  # type: ignore[arg-type]
    async def memory_edit(self, interaction: discord.Interaction, correction: str) -> None:
        """Teach a durable fact about yourself (jumps the learning queue)."""
        fact = await self.memory.facts.remember(
            guild_id=str(interaction.guild_id),
            subject_id=str(interaction.user.id),
            text=correction,
            actor_id=str(interaction.user.id),
            idempotency_key=str(interaction.id),
        )
        await interaction.response.send_message(
            f'Saved privately: "{fact.text}". This has no public message source, '
            "so it will not be injected into channel replies.",
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
        await ctx.reply("You're opted out. I'll stop observing you.", mention_author=False)


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


_reply_llm: OpenAICompatLLM | None = None


def _reply_llm_client() -> OpenAICompatLLM:
    global _reply_llm
    if _reply_llm is None:
        _reply_llm = build_chat_llm(LlmConfig.from_url(LLM_URL))
    return _reply_llm


async def generate(system_prompt: str, history, question: str) -> str:
    messages: list[LlmMessage] = [LlmMessage(role=MessageRole.SYSTEM, content=system_prompt)]
    for author, content in history:
        if content.strip():
            messages.append(LlmMessage(role=MessageRole.USER, content=f"{author}: {content}"))
    messages.append(LlmMessage(role=MessageRole.USER, content=question))
    response = await _reply_llm_client().complete(
        ChatRequest(
            messages=tuple(messages),
            max_tokens=900,
            # Consumer-defined purpose: the meter vocabulary is open. The
            # library's own calls use MeterPurpose members.
            purpose="reply",
        )
    )
    return response.text.strip()


def main() -> None:
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("set DISCORD_TOKEN")

    bot = OmniStyleBot()
    register_governance(bot)

    bot.run(token)


if __name__ == "__main__":
    main()
