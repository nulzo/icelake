"""Relation edge derivation and weighting."""

from __future__ import annotations

import math
from datetime import datetime

from icelake.models.graph import Polarity, RelationEdge

_NEGATIVE_VERBS = frozenset(
    {
        "dislikes",
        "hates",
        "called_out",
        "argues_with",
        "distrusts",
        "feuds_with",
        "disapproves_of",
        "avoids",
    }
)
_POSITIVE_VERBS = frozenset(
    {
        "likes",
        "loves",
        "enjoys",
        "prefers",
        "friends_with",
        "friend_of",
        "supports",
        "admires",
        "collaborates_with",
        "teammate_of",
        "fan_of",
    }
)


def polarity_for_verb(verb: str) -> Polarity:
    """Map a relation verb to its polarity; unknown verbs are NEUTRAL."""
    normalized = verb.strip().lower().replace(" ", "_")
    if normalized in _NEGATIVE_VERBS:
        return Polarity.NEGATIVE
    if normalized in _POSITIVE_VERBS:
        return Polarity.POSITIVE
    return Polarity.NEUTRAL


def compute_edge_weight(
    *,
    occurrences: int,
    confidence: float,
    last_reinforced_at: datetime,
    now: datetime,
) -> float:
    """``ln(1+occurrences) x confidence x recency-decay`` — hub ranking signal."""
    occ_component = math.log1p(max(1, occurrences))
    age_days = max(0.0, (now - last_reinforced_at).total_seconds() / 86_400.0)
    decay = math.exp(-age_days / 90.0)
    return round(occ_component * confidence * decay, 6)


def merge_edge(
    existing: RelationEdge,
    incoming: RelationEdge,
    *,
    now: datetime,
) -> RelationEdge:
    """Merge one more observation into an active edge (bitemporal, evidence-capped).

    Weight is monotonic: reinforcement never weakens a live edge.
    """
    evidence = dict.fromkeys(existing.evidence_fact_ids + incoming.evidence_fact_ids)
    merged_occurrences = existing.occurrences + 1
    recomputed = compute_edge_weight(
        occurrences=merged_occurrences,
        confidence=max(existing.confidence, incoming.confidence),
        last_reinforced_at=now,
        now=now,
    )
    return existing.model_copy(
        update={
            "occurrences": merged_occurrences,
            "weight": max(existing.weight, recomputed),
            "confidence": max(existing.confidence, incoming.confidence),
            "evidence_fact_ids": tuple(evidence)[-8:],
        }
    )
