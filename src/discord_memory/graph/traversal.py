"""Hop traversal, entity stance aggregation, and shared-trait discovery (§5.4)."""

from __future__ import annotations

from collections import deque

from discord_memory.models.graph import (
    NeighborInfo,
    Polarity,
    RelationEdge,
    StanceSummary,
)


def node_key(node_type_value: str, node_id: str) -> str:
    """Adjacency map key: ``"type:id"``."""
    return f"{node_type_value}:{node_id}"


def hop_neighbors(
    seed_key: str,
    adjacency: dict[str, list[RelationEdge]],
    *,
    depth: int,
    fan_out_per_node: int,
) -> tuple[NeighborInfo, ...]:
    """Bounded BFS over pre-fetched adjacency lists.

    ``adjacency`` maps ``node_key`` to that node's weight-ranked incident edges.
    Depth is caller-capped; each hop expands only the top-weight edges (hub
    mitigation). Every discovered neighbor carries its relation path so callers can
    phrase provenance honestly.
    """
    results: dict[tuple[str, str], NeighborInfo] = {}
    visited: set[str] = {seed_key}
    queue: deque[tuple[int, str, tuple[str, ...], float]] = deque([(0, seed_key, (), 0.0)])

    while queue:
        level, current_key, path, accumulated = queue.popleft()
        if level >= depth:
            continue
        for edge in adjacency.get(current_key, [])[:fan_out_per_node]:
            child_key = node_key(edge.dst_type.value, edge.dst_id)
            identity = (edge.dst_type.value, edge.dst_id)
            if child_key in visited:
                continue
            child = NeighborInfo(
                node_type=edge.dst_type,
                node_id=edge.dst_id,
                strength=round(accumulated + edge.weight, 6),
                relation_path=(*path, edge.verb),
                evidence_fact_ids=edge.evidence_fact_ids[:8],
            )
            results[identity] = child
            visited.add(child_key)
            queue.append((level + 1, child_key, child.relation_path, accumulated + edge.weight))

    return tuple(sorted(results.values(), key=lambda n: -n.strength))


def aggregate_stances(
    entity_slug: str,
    entity_name: str,
    edges: tuple[RelationEdge, ...],
) -> StanceSummary:
    """Group an entity's incident edges by polarity (Q5 honest co-presentation)."""
    positive: list[RelationEdge] = []
    negative: list[RelationEdge] = []
    other: list[RelationEdge] = []
    for edge in sorted(edges, key=lambda e: -e.weight):
        if edge.polarity is Polarity.POSITIVE:
            positive.append(edge)
        elif edge.polarity is Polarity.NEGATIVE:
            negative.append(edge)
        else:
            other.append(edge)
    total = sum(e.occurrences for e in edges)
    return StanceSummary(
        entity_slug=entity_slug,
        entity_name=entity_name,
        positive=tuple(positive),
        negative=tuple(negative),
        other=tuple(other),
        total_evidence=total,
    )


def jaccard_similarity(set_a: frozenset[str], set_b: frozenset[str]) -> float:
    """Set similarity for shared-trait discovery; merge-join friendly."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


__all__ = [
    "aggregate_stances",
    "hop_neighbors",
    "jaccard_similarity",
    "node_key",
]
