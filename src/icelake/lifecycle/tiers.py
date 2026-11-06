"""Tier assignment and TTL computation."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

from icelake.config import LifecycleConfig
from icelake.identity.names import name_evidence
from icelake.models.common import FrozenModel, ensure_aware
from icelake.models.facts import AttributionType, FactCategory, FactRecord, MemoryTier

_DURABLE_CATEGORIES = {
    FactCategory.PERSONAL,
    FactCategory.RELATIONSHIPS,
    FactCategory.PROFESSIONAL,
    FactCategory.PREFERENCES,
    FactCategory.INTERESTS,
    FactCategory.PERSONALITY,
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
    name_is_grounded: bool = False,
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

    # A direct, sourced identity preference must not expire because extraction
    # omitted/misclassified its category or described its start as "today".
    if name_is_grounded:
        category = FactCategory.PERSONAL

    lowered_confidence = confidence
    if is_server_fact and category in {FactCategory.RULES, FactCategory.CULTURE}:
        if lowered_confidence >= 0.85:
            return MemoryTier.CORE, None
        return MemoryTier.LONG_TERM, _long_term_expiry(lifecycle)

    horizon = None if name_is_grounded else _mentions_horizon(text, lifecycle.short_term_days)
    if horizon is not None:
        return MemoryTier.SHORT_TERM, timedelta(days=horizon)

    if (
        confidence >= _CORE_CONFIDENCE
        and occurrences >= _CORE_OCCURRENCES
        and category in _DURABLE_CATEGORIES
    ):
        return MemoryTier.CORE, None

    if category in _DURABLE_CATEGORIES or category is FactCategory.EXPERIENCES:
        return MemoryTier.LONG_TERM, _long_term_expiry(lifecycle)

    return MemoryTier.MID_TERM, timedelta(days=lifecycle.mid_term_days)


def _long_term_expiry(lifecycle: LifecycleConfig) -> timedelta | None:
    return (
        timedelta(days=lifecycle.long_term_days) if lifecycle.long_term_days is not None else None
    )


class RetentionExtension(FrozenModel):
    """A monotonic policy update; content, evidence and validity are unchanged."""

    tier: MemoryTier
    expires_at: datetime | None


def retention_extension(
    record: FactRecord, *, now: datetime, lifecycle: LifecycleConfig
) -> RetentionExtension | None:
    """Extend an automatic memory without reviving retired evidence.

    Never shortens a deadline, makes an expired memory current, or changes a
    manual memory or short-term plan. Persist with a version check so a
    concurrent correction or deletion wins.
    """
    if (
        not record.is_active
        or not record.is_valid_at(now)
        or record.expires_at is None
        or record.tier is MemoryTier.SHORT_TERM
        or record.attribution.type is AttributionType.MANUAL
    ):
        return None
    tier, duration = assign_tier(
        text=record.text,
        category=record.category,
        confidence=record.confidence,
        occurrences=record.occurrences,
        manual=False,
        is_server_fact=record.is_server_fact,
        lifecycle=lifecycle,
        name_is_grounded=bool(name_evidence(record)),
    )
    if tier is MemoryTier.SHORT_TERM:
        return None
    anchor = record.last_reinforced_at or record.created_at or record.observed_at
    if duration is not None and anchor is None:
        return None
    expiry = (
        ensure_aware(anchor) + duration if anchor is not None and duration is not None else None
    )
    if expiry is not None and expiry <= ensure_aware(record.expires_at):
        return None
    return RetentionExtension(tier=tier, expires_at=expiry)
