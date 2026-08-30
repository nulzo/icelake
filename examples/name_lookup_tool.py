"""Answering "what do you know about <name>?" when the name is prose, not a mention.

Runnable WITHOUT Discord or an LLM — ``python examples/name_lookup_tool.py``
seeds a guild into SQLite via ``facts.remember`` and runs the lookup handler
end to end.

The pattern is two library calls — ``identity.resolve`` then a STRICT
``facts.list_for_subject`` — and it is transport-agnostic. The ``user``
argument can come from anywhere; no tool runtime is required:

1. A slash command with a free-text option (zero LLM — see ``memory_lookup``
   in ``examples/omni_style_bot.py``).
2. One structured-output router call via ``ChatRequest.response_schema``
   (see ``_route_name_lookup`` in the same file).
3. Native function calling, if your LLM client already speaks tools —
   ``MEMORY_SHOW_TOOL`` below is the schema you hand it.

The library never scans prose for names itself; ``prompt_context`` scopes
sections from structured mentions and reply targets only. ``memory_show``
here is the handler every one of those entry points converges on: resolve
the name through the identity ladder, refuse to guess on ambiguity, and
fetch a subject-only profile so the model cannot attribute someone else's
claim to the person asked about.
"""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime

from icelake import (
    ChannelName,
    DiscordMemory,
    MemoryConfig,
    MessageEvent,
    RecallQuery,
    channels,
)

GUILD = "555"
MEMBERS = {
    "alice": "100000000000000001",
    "bob": "200000000000000002",
    "carol": "300000000000000003",
    "alex_s": "400000000000000004",
    "alex_j": "500000000000000005",
}

# Only needed for the LLM-driven entry points: native function calling takes
# this schema verbatim; a JSON-mode router uses the same shape as its
# response_schema. A slash-command option needs no schema at all.
MEMORY_SHOW_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_show",
        "description": (
            "Look up everything stored about a server member. Call this when the "
            "user asks about a person BY NAME. Pass the name exactly as written; "
            "never invent an ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "string",
                    "description": "Display name, @mention, or Discord ID of the member.",
                },
            },
            "required": ["user"],
            "additionalProperties": False,
        },
    },
}

_MENTION_RE = re.compile(r"<@!?(?P<snowflake>\d+)>")


async def memory_show(
    memory: DiscordMemory,
    guild_id: str,
    *,
    user: str,
    limit: int = 8,
) -> str:
    """Lookup handler: name in, labeled profile out. Never guesses on ambiguity."""
    ref = user.strip()
    if match := _MENTION_RE.fullmatch(ref):
        ref = match.group("snowflake")
    if not ref:
        return "No user given — ask which member they mean."

    resolution = await memory.identity.resolve(guild_id, ref)
    if resolution.ambiguous:
        choices = [
            f"{await memory.identity.display_name(guild_id, c.user_id) or c.user_id}"
            f" (ID {c.user_id})"
            for c in resolution.candidates
        ]
        return f"{ref!r} matches more than one member ({', '.join(choices)}) — ask which one."
    if resolution.resolved is None:
        mentions = await memory.facts.search(guild_id, ref, limit=3)
        if not mentions:
            return f"No member named {ref!r} — say you don't recognize them."
        lines = "\n".join(f"- {fact.text}" for fact, _ in mentions)
        return (
            f"No member named {ref!r} — say you don't recognize them as a member.\n"
            f"Memories that mention {ref!r} (may be a thing, not a person):\n{lines}"
        )

    user_id = resolution.resolved.user_id
    display = await memory.identity.display_name(guild_id, user_id) or ref
    page = await memory.facts.list_for_subject(
        guild_id,
        user_id,
        include_server=False,
        limit=limit,
    )
    if not page.items:
        return f"{display} is a member, but nothing is stored about them yet."
    lines = "\n".join(f"- {fact.text}" for fact in page.items)
    return (
        f"Facts about {display} ONLY (subject-scoped — other members' claims "
        f"about them are not listed here; do not attribute these to the asker):\n{lines}"
    )


async def _hello(memory: DiscordMemory, name: str) -> None:
    """Observe one message so the identity ladder learns the display name."""
    await memory.observe(
        MessageEvent(
            message_id=f"seed-hello-{name}",
            guild_id=GUILD,
            channel_id="general",
            author_id=MEMBERS[name],
            content=f"hey, {name} checking in",
            created_at=datetime.now(UTC),
            author_display_name=name,
        )
    )


async def seed(memory: DiscordMemory) -> None:
    for name in MEMBERS:
        await _hello(memory, name)
    await memory.identity.register_alias(GUILD, MEMBERS["bob"], "bobby")
    # Two members share a nickname: resolution must refuse to guess.
    await memory.identity.register_alias(GUILD, MEMBERS["alex_s"], "alex")
    await memory.identity.register_alias(GUILD, MEMBERS["alex_j"], "alex")

    await memory.facts.remember(
        guild_id=GUILD,
        subject_id=MEMBERS["bob"],
        text="bob maintains the server's minecraft world and hosts sunday builds",
        actor_id=MEMBERS["bob"],
    )
    await memory.facts.remember(
        guild_id=GUILD,
        subject_id=MEMBERS["bob"],
        text="bob is learning japanese before a tokyo trip",
        actor_id=MEMBERS["bob"],
    )
    # Stored on carol, but bob co-participates (actor), so he is on the fact's
    # incidence links: links-channel recall surfaces it under bob while a
    # strict profile fetch must not. (Extraction links @mentioned users the
    # same way; remember() derives mentions from the actor.)
    await memory.facts.remember(
        guild_id=GUILD,
        subject_id=MEMBERS["carol"],
        text="carol and bob argued over the game night loss",
        actor_id=MEMBERS["bob"],
    )
    # "zenith" is a place, not a member — the unknown-name fallback finds it.
    await memory.facts.remember(
        guild_id=GUILD,
        subject_id=MEMBERS["alice"],
        text="alice is planning a group hike at zenith ridge",
        actor_id=MEMBERS["alice"],
    )


async def main() -> None:
    memory = DiscordMemory(MemoryConfig(storage="sqlite://:memory:", llm=None))
    async with memory:
        await seed(memory)

        print("=== memory_show(user=...) — the lookup handler, end to end ===\n")

        print('tool call: memory_show(user="bob")  — strict profile')
        print(await memory_show(memory, GUILD, user="bob"))
        print()

        print('tool call: memory_show(user="bobby")  — nickname resolves the same')
        print(await memory_show(memory, GUILD, user="bobby"))
        print()

        print('tool call: memory_show(user="alex")  — shared nickname, never guesses')
        print(await memory_show(memory, GUILD, user="alex"))
        print()

        print('tool call: memory_show(user="zenith")  — not a member, mention fallback')
        print(await memory_show(memory, GUILD, user="zenith"))
        print()

        # CONTRAST: why the strict fetch matters. Recall's links channel
        # (in the default set, and in prompt_context) surfaces every fact
        # INCIDENT on bob — carol's fact links him as a participant, so it
        # appears under bob even though carol is the subject. Right for
        # passive context (each section is labeled); wrong for a direct
        # "about bob" answer, which must be subject-scoped.
        print("=== CONTRAST: recall(subject_ids=(bob,)) is cross-subject by design ===\n")
        result = await memory.recall(
            RecallQuery(
                guild_id=GUILD,
                subject_ids=(MEMBERS["bob"],),
                channels=channels(ChannelName.LINKS),
            )
        )
        for scored in result.facts:
            owner = scored.fact.subject_id
            owner_name = next(n for n, uid in MEMBERS.items() if uid == owner)
            print(f"  [subject={owner_name}] {scored.fact.text}")


if __name__ == "__main__":
    asyncio.run(main())
