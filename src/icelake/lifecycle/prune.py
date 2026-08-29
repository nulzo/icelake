"""Cap enforcement: which facts to invalidate when a subject exceeds its budget.

Shared by every store adapter so SQLite/Mongo cannot drift from the in-memory
reference (weakest first, manuals exempt, CORE last).
"""

from __future__ import annotations

from collections.abc import Sequence

from icelake.models.facts import AttributionType, FactRecord


def prune_sort_key(record: FactRecord) -> tuple[int, float, float]:
    """Ascending: prune these first. CORE is last; higher strength/confidence keep."""
    return (record.tier.prune_priority, record.strength, record.confidence)


def select_prune_victims(records: Sequence[FactRecord], *, cap: int) -> tuple[FactRecord, ...]:
    """Return the weakest non-manual facts that put ``records`` over ``cap``."""
    eligible = [
        record for record in records if record.attribution.type is not AttributionType.MANUAL
    ]
    excess = len(eligible) - cap
    if excess <= 0:
        return ()
    eligible.sort(key=prune_sort_key)
    return tuple(eligible[:excess])


def select_prune_victims_by_anchor(
    records: Sequence[FactRecord],
    *,
    max_per_user: int,
    max_server: int,
) -> tuple[FactRecord, ...]:
    """Group active facts by subject (``None`` = server) and apply per-bucket caps."""
    buckets: dict[str | None, list[FactRecord]] = {}
    for record in records:
        if not record.is_active:
            continue
        if record.attribution.type is AttributionType.MANUAL:
            continue
        anchor = None if record.is_server_fact else record.subject_id
        buckets.setdefault(anchor, []).append(record)
    victims: list[FactRecord] = []
    for anchor, group in buckets.items():
        cap = max_server if anchor is None else max_per_user
        victims.extend(select_prune_victims(group, cap=cap))
    return tuple(victims)
