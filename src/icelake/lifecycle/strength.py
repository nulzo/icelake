"""Ebbinghaus-style strength and forgetting math.

``retention = exp(-Δdays / strength)``; reinforcement adds strength and resets the
clock. Pure functions — property-tested for monotonicity and reset behavior.
"""

from __future__ import annotations

import math
from datetime import datetime

from icelake.models.facts import MemoryTier

REINFORCE_STRENGTH_STEP = 1.0
MIN_STRENGTH = 1.0


def retention(
    *,
    last_reinforced_at: datetime,
    now: datetime,
    strength: float,
    stability_days: float = 1.0,
) -> float:
    """Retention fraction in [0, 1] since the last reinforcement.

    ``stability_days`` is the decay timescale per strength point — the sweep
    and the ranking path must pass the same configured value or they disagree
    about which facts are fading.
    """
    if strength < MIN_STRENGTH:
        strength = MIN_STRENGTH
    delta_days = max(0.0, (now - last_reinforced_at).total_seconds() / 86_400.0)
    return math.exp(-delta_days / (strength * stability_days))


def reinforced_strength(current_strength: float) -> float:
    """Strength after one reinforcement observation."""
    return current_strength + REINFORCE_STRENGTH_STEP


def should_forget(
    *,
    retention_value: float,
    tier: MemoryTier,
    manual: bool,
    forget_retention_floor: float,
) -> bool:
    """Forgetting gate: weak retention, non-core, never manual facts."""
    if tier is MemoryTier.CORE or manual:
        return False
    return retention_value < forget_retention_floor


def strength_signal(
    *,
    strength: float,
    retention_value: float,
) -> float:
    """Ranking component in [0, 1]: log-scaled strength times live retention."""
    log_component = min(1.0, math.log1p(max(0.0, strength)) / math.log1p(10))
    return round(log_component * max(0.0, min(1.0, retention_value)), 6)
