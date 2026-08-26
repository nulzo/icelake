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


_SELF_NAME_PATTERNS = (
    (
        re.compile(
            r"\b[Mm]y name(?:'s| is) ([A-Z][a-zA-Z'-]{1,20})"
            r"(?: ([A-Z][a-zA-Z'-]{1,20}))?",
        ),
        0.88,
    ),
    (re.compile(r"\bcall me ([A-Z][a-zA-Z'-]{1,20})", re.IGNORECASE), 0.86),
    (re.compile(r"\bgoes by ([A-Z][a-zA-Z'-]{1,20})", re.IGNORECASE), 0.85),
)


def extract_self_name_aliases(text: str) -> list[tuple[str, float]]:
    """Mine self-referential name mentions from fact/message text.

    Matches capitalized 1-2-token names after markers like ``my name is X`` /
    ``call me X`` / ``goes by X``. Returns ``(surface, weight)`` pairs.
    """
    found: list[tuple[str, float]] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern, weight in _SELF_NAME_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span(1)
            if span in seen_spans:
                continue
            seen_spans.add(span)
            surface = match.group(1)
            if len(match.groups()) > 1 and match.group(2):
                surface = f"{surface} {match.group(2)}"
            found.append((surface, weight))
    return found


_KINSHIP_TERMS = frozenset(
    {
        "brother",
        "sister",
        "mom",
        "mother",
        "dad",
        "father",
        "cousin",
        "uncle",
        "aunt",
        "grandma",
        "grandpa",
        "friend",
        "roommate",
        "boss",
        "wife",
        "husband",
        "partner",
        "son",
        "daughter",
        "kid",
        "dog",
        "cat",
    }
)


def is_third_party_name_reference(text: str, name: str) -> bool:
    """Guard: possessive/kinship patterns must not become user aliases.

    ``"X's brother Ivan"`` or ``"someone named Ivan"`` refers to a third party,
    so the mined name must not bind to the subject.
    """
    lowered = " " + text.lower() + " "
    name_lower = name.lower()
    if f"'s {name_lower}" in lowered:
        return True
    for term in _KINSHIP_TERMS:
        if re.search(
            rf"\b{term}\b[^.]{{0,24}}\b{re.escape(name_lower)}\b",
            lowered,
        ):
            return True
    return f"someone named {name_lower}" in lowered
