"""Recall channels: independent ranked candidate producers (PLAN.md §5.2).

Each channel returns ranked fact ids plus optional calibrated score components.
Channels degrade independently — failures log and return empty, never raise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from discord_memory.models.graph import NodeType
from discord_memory.models.retrieval import ChannelName
from discord_memory.ports.llm import Embedder
from discord_memory.ports.store import MemoryStore
from discord_memory.ports.vectors import VectorIndex

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChannelOutput:
    """One channel's contribution: ranked ids + component maps."""

    channel: ChannelName
    ranked_ids: tuple[str, ...] = ()
    semantic: dict[str, float] = field(default_factory=dict)
    lexical: dict[str, float] = field(default_factory=dict)
    entity: dict[str, float] = field(default_factory=dict)


async def vector_channel(
    *,
    vectors: VectorIndex | None,
    embedder: Embedder | None,
    guild_id: str,
    query_text: str,
    subject_ids: tuple[str, ...] | None,
    server_only: bool,
    limit: int,
    candidate_cap: int,
) -> ChannelOutput:
    if vectors is None or embedder is None or not query_text.strip():
        return ChannelOutput(channel=ChannelName.VECTOR)
    (embedding,) = await embedder.embed((query_text,))
    hits = await vectors.search(
        embedding,
        guild_id=guild_id,
        subject_ids=subject_ids,
        server_only=server_only,
        limit=limit,
        candidate_cap=candidate_cap,
    )
    return ChannelOutput(
        channel=ChannelName.VECTOR,
        ranked_ids=tuple(h.id for h in hits),
        semantic={h.id: h.score for h in hits},
    )


async def keyword_channel(
    *,
    store: MemoryStore,
    guild_id: str,
    query_text: str,
    subject_ids: tuple[str, ...] | None,
    server_only: bool,
    limit: int,
    as_of: datetime | None = None,
) -> ChannelOutput:
    if not query_text.strip():
        return ChannelOutput(channel=ChannelName.KEYWORD)
    scored = await store.search_facts_text(
        guild_id,
        query_text,
        subject_ids=subject_ids,
        server_only=server_only,
        limit=limit,
        as_of=as_of,
    )
    return ChannelOutput(
        channel=ChannelName.KEYWORD,
        ranked_ids=tuple(r.id for r, _ in scored),
        lexical={r.id: s for r, s in scored},
    )


async def links_channel(
    *,
    store: MemoryStore,
    guild_id: str,
    subject_ids: tuple[str, ...],
    limit: int,
) -> ChannelOutput:
    """Facts touching any subject via incidence edges — cross-profile reach."""
    ranked: list[str] = []
    seen: set[str] = set()

    for user_id in subject_ids:
        linked = await store.links_for_node(
            guild_id,
            NodeType.USER,
            user_id,
            active_only=True,
            limit=limit,
        )
        for _row, record in linked:
            if record.id not in seen:
                seen.add(record.id)
                ranked.append(record.id)
    return ChannelOutput(channel=ChannelName.LINKS, ranked_ids=tuple(ranked[:limit]))


async def baseline_channel(
    *,
    store: MemoryStore,
    guild_id: str,
    subject_ids: tuple[str, ...] | None,
    server_only: bool,
    limit: int,
) -> ChannelOutput:
    """Top-strength anchor facts — cheap profile baseline."""
    records = await store.top_strength_facts(
        guild_id,
        subject_ids=subject_ids,
        server_only=server_only,
        limit=limit,
    )
    return ChannelOutput(channel=ChannelName.BASELINE, ranked_ids=tuple(r.id for r in records))


async def entity_channel(
    *,
    store: MemoryStore,
    guild_id: str,
    query_text: str,
    limit: int,
) -> ChannelOutput:
    """Exact entity-slug hits from the query text."""
    from discord_memory.identity.aliases import normalize_alias

    if not query_text.strip():
        return ChannelOutput(channel=ChannelName.ENTITY)
    words = normalize_alias(query_text).split()
    ranked: list[str] = []
    seen: set[str] = set()

    for word in words:
        if len(word) < 3:
            continue
        slug = await store.resolve_entity_alias(guild_id, word)
        if slug is None:
            continue
        linked = await store.links_for_node(
            guild_id,
            NodeType.ENTITY,
            slug,
            active_only=True,
            limit=limit,
        )
        for _row, record in linked:
            if record.id not in seen:
                seen.add(record.id)
                ranked.append(record.id)
    return ChannelOutput(
        channel=ChannelName.ENTITY,
        ranked_ids=tuple(ranked[:limit]),
        entity={fact_id: 1.0 for fact_id in ranked[:limit]},
    )


async def graph_hop_channel(
    *,
    store: MemoryStore,
    guild_id: str,
    subject_ids: tuple[str, ...],
    depth: int,
    fan_out_per_node: int,
    limit: int,
) -> ChannelOutput:
    """Bounded 2-hop expansion: neighbor nodes' facts enter the candidate pool."""
    from collections import deque

    from discord_memory.graph.traversal import node_key

    ranked: list[str] = []
    seen_facts: set[str] = set()
    visited: set[str] = set()
    frontier: deque[tuple[NodeType, str, int]] = deque()

    for user_id in subject_ids:
        frontier.append((NodeType.USER, user_id, 0))
        visited.add(node_key(NodeType.USER.value, user_id))

    while frontier:
        node_type, node_id, level = frontier.popleft()
        edges = await store.incident_edges(
            guild_id,
            (node_type, node_id),
            limit=fan_out_per_node,
        )
        for edge in sorted(edges, key=lambda e: -e.weight)[:fan_out_per_node]:
            child_key = node_key(edge.dst_type.value, edge.dst_id)
            linked = await store.links_for_node(
                guild_id,
                edge.dst_type,
                edge.dst_id,
                active_only=True,
                limit=limit,
            )
            for _row, record in linked:
                if record.id not in seen_facts:
                    seen_facts.add(record.id)
                    ranked.append(record.id)
            if level + 1 < depth and child_key not in visited:
                visited.add(child_key)
                frontier.append((edge.dst_type, edge.dst_id, level + 1))

    return ChannelOutput(channel=ChannelName.GRAPH_HOP, ranked_ids=tuple(ranked[:limit]))
