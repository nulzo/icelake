"""Reciprocal Rank Fusion and calibrated hybrid reranking (PLAN.md §5.3)."""

from __future__ import annotations

from dataclasses import dataclass, field

from discord_memory.models.retrieval import ChannelName


@dataclass(slots=True)
class RankedChannel:
    """One channel's ranked candidate ids (best first) with its name."""

    channel: ChannelName
    ranked_ids: tuple[str, ...]


@dataclass(slots=True)
class FusedCandidate:
    """A candidate after RRF fusion and component scoring."""

    fact_id: str
    rrf_score: float = 0.0
    channels: list[ChannelName] = field(default_factory=list)


def reciprocal_rank_fusion(
    channels: list[RankedChannel],
    *,
    k: int = 60,
    pool_size: int = 100,
) -> list[FusedCandidate]:
    """Merge per-channel rankings by ``1/(k + rank)``; returns top ``pool_size``."""
    fused: dict[str, FusedCandidate] = {}
    for channel in channels:
        for rank, fact_id in enumerate(channel.ranked_ids):
            candidate = fused.setdefault(fact_id, FusedCandidate(fact_id=fact_id))
            candidate.rrf_score += 1.0 / (k + rank + 1)
            if channel.channel not in candidate.channels:
                candidate.channels.append(channel.channel)
    ordered = sorted(fused.values(), key=lambda c: -c.rrf_score)
    return ordered[:pool_size]


@dataclass(slots=True)
class RerankInputs:
    """Per-fact [0,1] components feeding the weighted hybrid score."""

    semantic: dict[str, float] = field(default_factory=dict)
    lexical: dict[str, float] = field(default_factory=dict)
    entity: dict[str, float] = field(default_factory=dict)
    strength: dict[str, float] = field(default_factory=dict)


RerankResult = tuple[str, float, tuple[float, float, float, float], tuple[ChannelName, ...]]
"""One reranked candidate: (fact_id, score, components, matched channels)."""


def hybrid_rerank(
    candidates: list[FusedCandidate],
    inputs: RerankInputs,
    *,
    weight_semantic: float,
    weight_lexical: float,
    weight_entity: float,
    weight_strength: float,
) -> list[RerankResult]:
    """Weighted rescore of the fused pool. All components must be [0,1]-normalized.

    Missing components score 0 — no fabricated values, ever. Returns
    ``(fact_id, score, components, channels)`` sorted by score descending.
    """
    total_weight = weight_semantic + weight_lexical + weight_entity + weight_strength
    if total_weight <= 0:
        return []
    semantic_map = inputs.semantic
    lexical_map = inputs.lexical
    entity_map = inputs.entity
    strength_map = inputs.strength
    results: list[RerankResult] = []
    for candidate in candidates:
        sem = _clip01(semantic_map.get(candidate.fact_id, 0.0))
        lex = _clip01(lexical_map.get(candidate.fact_id, 0.0))
        ent = _clip01(entity_map.get(candidate.fact_id, 0.0))
        stre = _clip01(strength_map.get(candidate.fact_id, 0.0))
        combined = (
            weight_semantic * sem
            + weight_lexical * lex
            + weight_entity * ent
            + weight_strength * stre
        ) / total_weight
        results.append(
            (
                candidate.fact_id,
                round(combined, 6),
                (sem, lex, ent, stre),
                tuple(candidate.channels),
            )
        )
    results.sort(key=lambda item: -item[1])
    return results


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))
