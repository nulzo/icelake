"""Cross-user memory patterns: relationships, stances, discovery.

Runnable WITHOUT Discord or an LLM — ``python examples/relationship_queries.py``
seeds a small community into SQLite via ``facts.remember`` (the curation API) and
walks query shapes that are zero-LLM by design:

1. "What does X think about Y?"      -> relationship recall (pair mode)
2. "Did X ever call out Y?"          -> typed relation edges with evidence
3. "What does the server think of movies?" -> entity stance aggregation
4. "Who shares X's entities?"        -> Jaccard over the knowledge graph

Extraction, reconcile, and profile digests are not exercised here. Pass
``llm=`` / ``embeddings=`` on ``MemoryConfig`` only if you change the seed to
``observe`` + ``flush`` real chat; this file will not call a provider as written.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from discord_memory import (
    DiscordMemory,
    MemoryConfig,
    MessageEvent,
    RecallQuery,
)
from discord_memory.models.operations import ProposedRelation


async def seed_community(memory: DiscordMemory) -> dict[str, str]:
    """Seed aliases + curated facts. No extraction LLM is involved."""
    members = {
        "alice": "100000000000000001",
        "bob": "200000000000000002",
        "carol": "300000000000000003",
    }

    async def say(author: str, content: str) -> None:
        await memory.observe(
            MessageEvent(
                message_id=f"seed-{author}-{abs(hash(content)) % 10**9}",
                guild_id="555",
                channel_id="general",
                author_id=members[author],
                content=content,
                created_at=datetime.now(UTC),
                author_display_name=author,
            )
        )

    # In production the configured LLM extracts facts. Here we write them
    # directly so the example runs deterministically without any provider.
    from discord_memory.models.operations import ProposedEntity

    await memory.facts.remember(
        guild_id="555",
        subject_id=members["alice"],
        text="alice loves watching movies on weekends",
        actor_id="seed",
        entities=(ProposedEntity(name="Movies"),),
        relations=(
            ProposedRelation(verb="likes", from_token=members["alice"], to_entity="Movies"),
        ),
    )
    await memory.facts.remember(
        guild_id="555",
        subject_id=members["bob"],
        text="bob dislikes movies and prefers board games",
        actor_id="seed",
        entities=(ProposedEntity(name="Movies"),),
        relations=(
            ProposedRelation(verb="dislikes", from_token=members["bob"], to_entity="Movies"),
        ),
    )
    # Greetings register display-name aliases (identity ladder needs them).
    await say("alice", "hey everyone, quick hello from me to start the week")
    await say("bob", "hello all, bob here ready for some games later")
    await say("carol", "hi folks! carol checking in before game night")

    await memory.facts.remember(
        guild_id="555",
        subject_id=members["bob"],
        text="carol called bob a sore loser during game night",
        actor_id=members["carol"],
        speaker_id=members["carol"],
        relations=(
            ProposedRelation(
                verb="called_out",
                from_token=members["carol"],
                to_token=members["bob"],
            ),
        ),
    )
    return members


async def what_does_x_think_of_y(
    memory: DiscordMemory, guild_id: str, x_name: str, y_name: str
) -> str:
    """Resolve flexible names, then combine relation edges + bidirectional facts."""
    x = await memory.identity.resolve(guild_id, x_name)
    y = await memory.identity.resolve(guild_id, y_name)
    if x.resolved is None or y.resolved is None or x.ambiguous or y.ambiguous:
        return f"I can't uniquely tell who {x_name!r}/{y_name!r} is."

    x_id, y_id = x.resolved.user_id, y.resolved.user_id

    edges = await memory.graph.between(guild_id, x_id, y_id)
    lines = [f"- {edge.verb} (weight {edge.weight:.2f})" for edge in edges]

    result = await memory.recall(
        RecallQuery(
            guild_id=guild_id,
            text=f"{x_name} about {y_name}",
            subject_ids=(x_id, y_id),
        )
    )
    lines += [f"- {sf.fact.text}" for sf in result.facts]
    header = f"What {x.resolved.matched_alias} thinks about {y.resolved.matched_alias}:"
    return "\n".join([header, *lines]) if len(lines) > 1 else header + " nothing notable."


async def main() -> None:
    memory = DiscordMemory(MemoryConfig(storage="sqlite://:memory:", llm=None))
    async with memory:
        members = await seed_community(memory)

        # 1. Relationship recall — names resolved through the alias ladder.
        print(await what_does_x_think_of_y(memory, "555", "carol", "bob"))
        print()

        # 2. Typed relation edges between two members.
        carol, bob = members["carol"], members["bob"]
        edges = await memory.graph.between("555", carol, bob)
        print(f"edges carol<->bob: {[e.verb for e in edges]}")

        # 3. Entity stances — opposing views aggregate on one node.
        stances = await memory.graph.entity_stances("555", "movies")
        print(
            f"movies: {len(stances.positive)} positive / {len(stances.negative)} negative stances"
        )

        # 4. Shared-trait discovery.
        alice_id = members["alice"]
        similar = await memory.graph.similar_users("555", alice_id, limit=3)
        print(f"members similar to alice: {similar}")

        stats = await memory.stats("555")
        print(f"server holds {stats.active_facts} active memories")


if __name__ == "__main__":
    asyncio.run(main())
