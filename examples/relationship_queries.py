"""Cross-user memory patterns: relationships, stances, discovery.

Runnable WITHOUT Discord or an LLM — ``python examples/relationship_queries.py``
seeds a guild into SQLite via ``facts.remember`` (the curation API) and walks
query shapes that are zero-LLM by design:

1. Name resolution (display names + nicknames; ambiguity never guesses)
2. "What does X think about Y?" — directed edges + pair-intersect recall
3. Typed relation edges and 2-hop neighborhood
4. Entity stance aggregation (who likes / dislikes movies, coffee, …)
5. Shared-entity discovery (Jaccard — overlap, not "same taste")

Extraction, reconcile, and profile digests are not exercised here. Pass
``llm=`` / ``embeddings=`` on ``MemoryConfig`` only if you change the seed to
``observe`` + ``flush`` real chat; this file will not call a provider as written.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from icelake import (
    ChannelName,
    DiscordMemory,
    MemoryConfig,
    MessageEvent,
    RecallQuery,
    channels,
)
from icelake.models.graph import NodeType
from icelake.models.operations import ProposedEntity, ProposedRelation

GUILD = "555"
MEMBERS = {
    "alice": "100000000000000001",
    "bob": "200000000000000002",
    "carol": "300000000000000003",
    "dave": "400000000000000004",
    "eve": "500000000000000005",
    "frank": "600000000000000006",
    "grace": "700000000000000007",
    "henrik": "800000000000000008",
}
NAMES = {user_id: name for name, user_id in MEMBERS.items()}


def _who(user_id: str) -> str:
    return NAMES.get(user_id, user_id)


async def _hello(memory: DiscordMemory, author: str) -> None:
    """Observe a greeting so the identity ladder learns the display name."""
    await memory.observe(
        MessageEvent(
            message_id=f"seed-hello-{author}",
            guild_id=GUILD,
            channel_id="general",
            author_id=MEMBERS[author],
            content=f"hey, {author} checking in",
            created_at=datetime.now(UTC),
            author_display_name=author,
        )
    )


async def _teach(
    memory: DiscordMemory,
    *,
    subject: str,
    text: str,
    speaker: str | None = None,
    likes: tuple[str, ...] = (),
    dislikes: tuple[str, ...] = (),
    edges: tuple[tuple[str, str, str], ...] = (),
) -> None:
    """Curate one fact + graph participation. ``edges`` are (verb, from, to) member names."""
    subject_id = MEMBERS[subject]
    speaker_id = MEMBERS[speaker] if speaker else None
    entities = tuple(ProposedEntity(name=name) for name in (*likes, *dislikes))
    relations = [
        *(ProposedRelation(verb="likes", from_token=subject_id, to_entity=name) for name in likes),
        *(
            ProposedRelation(verb="dislikes", from_token=subject_id, to_entity=name)
            for name in dislikes
        ),
        *(
            ProposedRelation(verb=verb, from_token=MEMBERS[src], to_token=MEMBERS[dst])
            for verb, src, dst in edges
        ),
    ]
    other_ids = tuple(
        uid for _, src, dst in edges for uid in (MEMBERS[src], MEMBERS[dst]) if uid != subject_id
    )
    await memory.facts.remember(
        guild_id=GUILD,
        subject_id=subject_id,
        text=text,
        # remember() only puts subject/actor/speaker on the relation roster —
        # the other endpoint of a person-to-person edge must be actor or speaker.
        actor_id=speaker_id or (other_ids[0] if other_ids else "seed"),
        speaker_id=speaker_id,
        entities=entities,
        relations=tuple(relations),
    )


async def seed_community(memory: DiscordMemory) -> None:
    """Eight members, overlapping tastes, and a handful of person-to-person edges."""
    for name in MEMBERS:
        await _hello(memory, name)
    await memory.identity.register_alias(GUILD, MEMBERS["bob"], "bobby")
    await memory.identity.register_alias(GUILD, MEMBERS["alice"], "ali")

    await _teach(
        memory,
        subject="alice",
        text="alice loves weekend movies and writes Rust at work",
        likes=("Movies", "Rust", "Coffee"),
    )
    await _teach(
        memory,
        subject="bob",
        text="bob dislikes movies and prefers board games with coffee",
        likes=("Board Games", "Coffee"),
        dislikes=("Movies",),
    )
    await _teach(
        memory,
        subject="carol",
        text="carol hosts movie night every friday",
        likes=("Movies",),
    )
    await _teach(
        memory,
        subject="dave",
        text="dave ships Rust with alice and plays chess on lunch",
        likes=("Rust", "Chess"),
        edges=(("teammate_of", "dave", "alice"),),
    )
    await _teach(
        memory,
        subject="eve",
        text="eve plays chess but cannot stand coffee",
        likes=("Chess",),
        dislikes=("Coffee",),
        edges=(("friend_of", "eve", "carol"),),
    )
    await _teach(
        memory,
        subject="frank",
        text="frank is in bob's board-game group and dabbles in Rust",
        likes=("Board Games", "Rust"),
        edges=(("friend_of", "frank", "bob"),),
    )
    await _teach(
        memory,
        subject="grace",
        text="grace mainlines coffee and never misses a movie",
        likes=("Movies", "Coffee"),
    )
    await _teach(
        memory,
        subject="henrik",
        text="henrik only shows up for ranked chess",
        likes=("Chess",),
    )
    await _teach(
        memory,
        subject="bob",
        speaker="carol",
        text="carol called bob a sore loser during game night",
        edges=(("called_out", "carol", "bob"),),
    )


async def _resolve(memory: DiscordMemory, identifier: str) -> str:
    resolution = await memory.identity.resolve(GUILD, identifier)
    if resolution.ambiguous:
        ids = ", ".join(_who(c.user_id) for c in resolution.candidates)
        return f"{identifier!r} is ambiguous ({ids}); refusing to guess"
    if resolution.resolved is None:
        return f"{identifier!r} matched nobody"
    return f"{identifier!r} -> {_who(resolution.resolved.user_id)} ({resolution.resolved.user_id})"


async def what_x_thinks_of_y(memory: DiscordMemory, x_name: str, y_name: str) -> str:
    """Directed edges X→Y plus facts linked to *both* people (not each profile dumped)."""
    x = await memory.identity.resolve(GUILD, x_name)
    y = await memory.identity.resolve(GUILD, y_name)
    if x.resolved is None or y.resolved is None or x.ambiguous or y.ambiguous:
        return f"cannot uniquely resolve {x_name!r} / {y_name!r}"
    x_id, y_id = x.resolved.user_id, y.resolved.user_id
    lines = [f"What {_who(x_id)} thinks about {_who(y_id)}:"]
    for edge in await memory.graph.between(GUILD, x_id, y_id):
        lines.append(
            f"  edge  {_who(x_id)} -{edge.verb}-> {_who(y_id)}  (weight {edge.weight:.2f})"
        )
    result = await memory.recall(
        RecallQuery(
            guild_id=GUILD,
            pair_ids=((x_id, y_id),),
            # Default channels are guild-wide; LINKS-only lets pair_ids be the
            # sole candidate source (facts incident on both people).
            channels=channels(ChannelName.LINKS),
        )
    )
    if result.facts:
        lines.extend(f"  fact  {scored.fact.text}" for scored in result.facts)
    elif len(lines) == 1:
        lines.append("  (no shared facts or directed edges)")
    return "\n".join(lines)


def _print_stances(label: str, stances) -> None:
    pos = ", ".join(_who(e.src_id) for e in stances.positive if e.src_type is NodeType.USER)
    neg = ", ".join(_who(e.src_id) for e in stances.negative if e.src_type is NodeType.USER)
    print(f"  {label}: +[{pos or '-'}]  -[{neg or '-'}]  ({stances.total_evidence} evidence)")


async def main() -> None:
    memory = DiscordMemory(MemoryConfig(storage="sqlite://:memory:", llm=None))
    async with memory:
        await seed_community(memory)

        print("=== 1. Identity ladder (nicknames, misses) ===")
        for ident in ("carol", "bobby", "ali", "nobody-here"):
            print(f"  {await _resolve(memory, ident)}")
        print()

        print("=== 2. What does X think about Y? (pair-intersect, not both profiles) ===")
        print(await what_x_thinks_of_y(memory, "carol", "bob"))
        print(await what_x_thinks_of_y(memory, "dave", "alice"))
        print(await what_x_thinks_of_y(memory, "eve", "carol"))
        print()

        print("=== 3. Graph around dave (incident edges + 2-hop neighbors) ===")
        dave = MEMBERS["dave"]
        for edge in await memory.graph.relations_of(GUILD, dave):
            src = _who(edge.src_id) if edge.src_type is NodeType.USER else edge.src_id
            dst = _who(edge.dst_id) if edge.dst_type is NodeType.USER else edge.dst_id
            print(f"  {src} -{edge.verb}-> {dst}")
        print("  hops:")
        for hop in await memory.graph.neighbors(GUILD, dave, depth=2):
            node = _who(hop.node_id) if hop.node_type is NodeType.USER else hop.node_id
            path = " / ".join(hop.relation_path) or "(seed)"
            print(f"    {node}  via {path}")
        print()

        print("=== 4. Entity stances ===")
        for entity in ("movies", "coffee", "chess", "rust"):
            _print_stances(entity, await memory.graph.entity_stances(GUILD, entity))
        print()

        print("=== 5. Similar members (shared entities; polarity is ignored) ===")
        for seed in ("alice", "bob", "henrik"):
            similar = await memory.graph.similar_users(GUILD, MEMBERS[seed], limit=4)
            shown = ", ".join(f"{_who(uid)} {score:.2f}" for uid, score in similar) or "(none)"
            print(f"  like {seed}: {shown}")

        stats = await memory.stats(GUILD)
        print(f"\nserver holds {stats.active_facts} active memories")


if __name__ == "__main__":
    asyncio.run(main())
