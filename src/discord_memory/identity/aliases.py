"""Alias normalization, validation, and weight derivation (PLAN.md §3.2)."""

from __future__ import annotations

import re

from discord_memory.models.identity import AliasSource

_DIGIT_RUN = re.compile(r"^\d{9,}$")
_INTERNAL_TOKEN = re.compile(r"^[a-z0-9._-]{2,32}$")

_SOURCE_WEIGHTS: dict[AliasSource, float] = {
    AliasSource.DISCORD_USERNAME: 1.0,
    AliasSource.SUBJECT_USERNAME: 0.95,
    AliasSource.REAL_NAME: 0.85,
    AliasSource.DISPLAY_NAME: 0.7,
    AliasSource.MENTION: 0.6,
    AliasSource.BACKFILL: 0.5,
    AliasSource.ENTITY_TAG: 0.4,
}


def normalize_alias(alias: str) -> str:
    """Lowercase and collapse whitespace. Matching is case-insensitive by construction."""
    return " ".join(alias.strip().lower().split())


def alias_slug(alias: str) -> str:
    """URL-safe slug for entity nodes: ``[a-z0-9-]``."""
    normalized = normalize_alias(alias)
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return slug or "unknown"


def is_valid_alias(alias_norm: str) -> bool:
    """Reject empties, single chars, and snowflake-shaped digit runs (poison guard)."""
    if len(alias_norm) < 2 or len(alias_norm) > 64:
        return False
    return not _DIGIT_RUN.match(alias_norm)


def weight_for_source(source: AliasSource, surface: str = "") -> float:
    """Base confidence weight for an alias source; usernames keep internal tokens."""
    base = _SOURCE_WEIGHTS[source]
    if source in {AliasSource.DISCORD_USERNAME, AliasSource.SUBJECT_USERNAME}:
        cleaned = surface.strip().lower()
        if source is AliasSource.DISCORD_USERNAME and _INTERNAL_TOKEN.match(cleaned):
            return max(base, 0.95)
    return base


def combined_confidence(source_rank: int, weight: float, top_weight: float) -> float:
    """Map (source rank, relative weight) into [0, 1] confidence."""
    rank_component = min(1.0, source_rank / 100.0)
    weight_component = min(1.0, weight / max(top_weight, 0.0001))
    return round(0.6 * rank_component + 0.4 * weight_component, 4)
