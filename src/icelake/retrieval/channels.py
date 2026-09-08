from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from icelake.models.graph import NodeType, RelationEdge
from icelake.models.retrieval import ChannelName
from icelake.ports.llm import Embedder
from icelake.ports.store import MemoryStore, NodeRef
from icelake.ports.vectors import VectorIndex

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChannelOutput:
    """One channel's contribution: ranked ids + component maps."""

    channel: ChannelName
    ranked_ids: tuple[str, ...] = ()
    semantic: dict[str, float] = field(default_factory=dict)
    lexical: dict[str, float] = field(default_factory=dict)
    entity: dict[str, float] = field(default_factory=dict)
    #: GRAPH_HOP provenance: fact id → node path that surfaced it.
    hop_paths: dict[str, tuple[str, ...]] = field(default_factory=dict)


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
    refs: tuple[NodeRef, ...],
    limit: int,
) -> ChannelOutput:
    """Facts touching any subject node via incidence edges — cross-profile reach.

    ``refs`` carries each subject's user node plus its entity twins (the caller
    expands via ``twin_refs``), so a member mentioned by name before they ever
    spoke still contributes their facts. One batched store round-trip.
    """
    if not refs:
        return ChannelOutput(channel=ChannelName.LINKS)
    linked = await store.links_for_nodes(
        guild_id,
        refs,
        active_only=True,
        limit_per_node=limit,
    )
    ranked: list[str] = []
    seen: set[str] = set()
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


#: Query words scanned for entity aliases, and the longest alias span tried.
#: Bounds the exact-match lookups at ``_ENTITY_MAX_WORDS * _ENTITY_MAX_NGRAM``.
_ENTITY_MAX_WORDS = 16
_ENTITY_MAX_NGRAM = 4


async def entity_channel(
    *,
    store: MemoryStore,
    guild_id: str,
    query_text: str,
    limit: int,
) -> ChannelOutput:
    """Exact entity-alias hits from the query text, longest match first.

    Multi-word entities resolve as one node — "board games" matches the
    ``board-games`` alias instead of missing on "board" + "games". A matched
    span is consumed, so its sub-phrases are not looked up again.
    """
    from icelake.identity.aliases import normalize_alias

    # Word tokens only — punctuation ("games?") must not break alias lookup.
    words = re.findall(r"[a-z0-9]+", normalize_alias(query_text))[:_ENTITY_MAX_WORDS]
    if not words:
        return ChannelOutput(channel=ChannelName.ENTITY)

    slugs: list[str] = []
    seen_slugs: set[str] = set()
    index = 0
    while index < len(words):
        hit: tuple[int, str] | None = None
        for size in range(min(_ENTITY_MAX_NGRAM, len(words) - index), 1, -1):
            slug = await store.resolve_entity_alias(guild_id, " ".join(words[index : index + size]))
            if slug is not None:
                hit = (size, slug)
                break
        if hit is None and len(words[index]) >= 3:
            slug = await store.resolve_entity_alias(guild_id, words[index])
            if slug is not None:
                hit = (1, slug)
        if hit is None:
            index += 1
            continue
        size, slug = hit
        index += size
        if slug not in seen_slugs:
            seen_slugs.add(slug)
            slugs.append(slug)

    ranked: list[str] = []
    seen: set[str] = set()
    for slug in slugs:
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
    seed_refs: tuple[NodeRef, ...],
    depth: int,
    fan_out_per_node: int,
    limit: int,
) -> ChannelOutput:
    """Bounded n-hop expansion: neighbor nodes' facts enter the candidate pool.

    Seeds are user nodes (plus entity twins, expanded by the caller) and —
    for entity-centric questions — the query's entity node, so "who likes
    golf?" hops golf → members → their other hubs (Graphiti-style BFS from
    query entities). Batched by BFS level: one ``incident_edges_many`` + one
    ``links_for_nodes`` per level — 2 round-trips per level regardless of
    fan-out. Each surfaced fact records the node path that reached it.
    """
    from icelake.graph.traversal import node_key

    ranked: list[str] = []
    seen_facts: set[str] = set()
    hop_paths: dict[str, tuple[str, ...]] = {}
    visited: set[str] = {node_key(node_type.value, node_id) for node_type, node_id in seed_refs}
    frontier: list[tuple[NodeType, str]] = list(seed_refs)
    #: node key → path of node keys from a seed to that node.
    paths: dict[str, tuple[str, ...]] = {
        node_key(node_type.value, node_id): (node_key(node_type.value, node_id),)
        for node_type, node_id in seed_refs
    }

    for level in range(depth):
        if not frontier:
            break
        edges = await store.incident_edges_many(
            guild_id,
            tuple(frontier),
            limit_per_node=fan_out_per_node,
        )
        by_node: dict[tuple[NodeType, str], list[RelationEdge]] = {node: [] for node in frontier}
        for edge in edges:
            for node in ((edge.src_type, edge.src_id), (edge.dst_type, edge.dst_id)):
                if node in by_node:
                    by_node[node].append(edge)
        destinations: list[tuple[NodeType, str]] = []
        destination_paths: dict[str, tuple[str, ...]] = {}
        next_frontier: list[tuple[NodeType, str]] = []
        for node in frontier:
            node_path = paths.get(node_key(node[0].value, node[1]), ())
            for edge in sorted(by_node[node], key=lambda e: -e.weight)[:fan_out_per_node]:
                destinations.append((edge.dst_type, edge.dst_id))
                child_key = node_key(edge.dst_type.value, edge.dst_id)
                destination_paths.setdefault(child_key, (*node_path, child_key))
                if level + 1 < depth and child_key not in visited:
                    visited.add(child_key)
                    next_frontier.append((edge.dst_type, edge.dst_id))
                    paths[child_key] = destination_paths[child_key]
        if not destinations:
            break
        linked = await store.links_for_nodes(
            guild_id,
            tuple(destinations),
            limit_per_node=limit,
        )
        for row, record in linked:
            if record.id not in seen_facts:
                seen_facts.add(record.id)
                ranked.append(record.id)
                hop_paths[record.id] = destination_paths.get(
                    node_key(row.node_type.value, row.node_id), ()
                )
        frontier = next_frontier

    return ChannelOutput(
        channel=ChannelName.GRAPH_HOP,
        ranked_ids=tuple(ranked[:limit]),
        hop_paths=hop_paths,
    )
