"""Forgetting sweep: which active facts fall below the retention floor.

Shared by every store adapter so SQLite/Mongo cannot drift from the in-memory
reference (same contract as ``prune.py``): the pure function decides, the
store only executes the bulk invalidate. CORE and manual facts are exempt.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from icelake.lifecycle.strength import retention, should_forget
from icelake.models.facts import AttributionType, FactRecord


def select_forgotten_facts(
    records: Sequence[FactRecord],
    *,
    now: datetime,
    retention_floor: float,
    stability_days: float = 1.0,
) -> tuple[FactRecord, ...]:
    """Active, non-exempt facts whose live retention is below the floor."""
    victims: list[FactRecord] = []
    for record in records:
        if not record.is_active:
            continue
        last = record.last_reinforced_at or record.created_at
        if last is None:
            continue
        value = retention(
            last_reinforced_at=last,
            now=now,
            strength=record.strength,
            stability_days=stability_days,
        )
        if should_forget(
            retention_value=value,
            tier=record.tier,
            manual=record.attribution.type is AttributionType.MANUAL,
            forget_retention_floor=retention_floor,
        ):
            victims.append(record)
    return tuple(victims)


__all__ = ["select_forgotten_facts"]
