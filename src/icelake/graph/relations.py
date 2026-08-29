from __future__ import annotations

import math
from datetime import datetime

from icelake.models.graph import Polarity, RelationEdge, RelationVerb

_NEGATIVE_VERBS = frozenset(
    {
        RelationVerb.DISLIKES,
        RelationVerb.HATES,
        RelationVerb.CALLED_OUT,
        RelationVerb.ARGUES_WITH,
        RelationVerb.DISTRUSTS,
        RelationVerb.FEUDS_WITH,
        RelationVerb.DISAPPROVES_OF,
        RelationVerb.AVOIDS,
    }
)

_POSITIVE_VERBS = frozenset(
    {
        RelationVerb.LIKES,
        RelationVerb.LOVES,
        RelationVerb.ENJOYS,
        RelationVerb.PREFERS,
        RelationVerb.FRIENDS_WITH,
        RelationVerb.FRIEND_OF,
        RelationVerb.SUPPORTS,
        RelationVerb.ADMIRES,
        RelationVerb.COLLABORATES_WITH,
        RelationVerb.TEAMMATE_OF,
        RelationVerb.FAN_OF,
    }
)


def polarity_for_verb(verb: str | RelationVerb) -> Polarity:
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
