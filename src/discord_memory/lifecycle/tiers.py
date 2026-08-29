"""Tier assignment and TTL computation (PLAN.md §4.4, ported from the bot)."""

from __future__ import annotations

import re
from datetime import timedelta

from discord_memory.config import LifecycleConfig
from discord_memory.models.facts import FactCategory, MemoryTier

_PLAN_TIME_MARKERS = (
    "today",
    "tonight",
    "tomorrow",
    "this weekend",
    "this week",
    "next week",
    "soon",
    "currently",
    "temporarily",
)
_DURABLE_CATEGORIES = {
    FactCategory.PERSONAL,
    FactCategory.RELATIONSHIPS,
    FactCategory.PROFESSIONAL,
}
_CORE_CONFIDENCE = 0.95
_CORE_OCCURRENCES = 3


def _mentions_horizon(text: str, short_term_days: int) -> int | None:
    """Return a TTL horizon in days for explicit short-horizon phrasing."""
    lowered = text.lower()
    if re.search(r"\b(today|tonight)\b", lowered):
        return min(3, short_term_days)
    if re.search(r"\b(tomorrow|this weekend|next week|this week)\b", lowered):
        return short_term_days
    if re.search(r"\b(this month|upcoming)\b", lowered):
        return 21
    return None


def assign_tier(
    *,
    text: str,
    category: FactCategory,
    confidence: float,
    occurrences: int,
    manual: bool,
    is_server_fact: bool,
    lifecycle: LifecycleConfig,
) -> tuple[MemoryTier, timedelta | None]:
    """Pure tier assignment. Returns ``(tier, expires_after)``; ``None`` = never expires.

    Rules (in order):
    1. Manual facts are CORE.
    2. Server rules/culture facts with high confidence are CORE.
    3. Explicit time horizons ("tomorrow") are SHORT_TERM.
    4. Identity/durable categories at high confidence+occurrences graduate to CORE.
    5. Durable categories default to LONG_TERM.
    6. Everything else defaults to MID_TERM.
    """
    if manual:
        return MemoryTier.CORE, None

    lowered_confidence = confidence
    if is_server_fact and category in {FactCategory.RULES, FactCategory.CULTURE}:
        if lowered_confidence >= 0.85:
            return MemoryTier.CORE, None
        return MemoryTier.LONG_TERM, timedelta(days=lifecycle.long_term_days)

    horizon = _mentions_horizon(text, lifecycle.short_term_days)
    if horizon is not None:
        return MemoryTier.SHORT_TERM, timedelta(days=horizon)

    if (
        confidence >= _CORE_CONFIDENCE
        and occurrences >= _CORE_OCCURRENCES
        and category in _DURABLE_CATEGORIES
    ):
        return MemoryTier.CORE, None

    if category in _DURABLE_CATEGORIES or category is FactCategory.EXPERIENCES:
        return MemoryTier.LONG_TERM, timedelta(days=lifecycle.long_term_days)

    return MemoryTier.MID_TERM, timedelta(days=lifecycle.mid_term_days)
