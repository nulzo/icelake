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


# One-two token names; case-insensitive because chat is lowercase in practice
# and the LLM normalizes case in extracted facts.
_NAME = r"([a-z][a-z0-9'.-]*(?: [a-z][a-z0-9'.-]*)?)"

# Second-token captures like "Klim and" / "Nolan but" are conjunction bleed,
# not surnames — drop the trailing token.
_TRAILING_NOISE = frozenset(
    {"and", "but", "or", "so", "the", "a", "an", "is", "it", "to", "too", "also", "then"}
)

_SELF_NAME_PATTERNS = (
    (re.compile(rf"\bmy name(?:'s| is) {_NAME}", re.IGNORECASE), 0.88),
    (re.compile(rf"\bcall me {_NAME}", re.IGNORECASE), 0.86),
    (re.compile(rf"\bgoes by {_NAME}", re.IGNORECASE), 0.85),
)

# Third-person forms found in extracted fact text ("nulzo's name is Nolan
# Gregory", "Real name is Nolan Gregory"). Only safe on text whose subject
# attribution is already known (facts), never on raw messages.
_STATED_NAME_PATTERNS = (
    (re.compile(rf"\b(?:real |full )?name is {_NAME}", re.IGNORECASE), 0.85),
)


def _collect(
    text: str, patterns: tuple[tuple[re.Pattern[str], float], ...]
) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    seen_spans: set[tuple[int, int]] = set()
    for pattern, weight in patterns:
        for match in pattern.finditer(text):
            span = match.span(1)
            if span in seen_spans:
                continue
            seen_spans.add(span)
            surface = match.group(1)
            parts = surface.split()
            if len(parts) == 2 and parts[1].lower() in _TRAILING_NOISE:
                surface = parts[0]
            found.append((surface, weight))
    return found


def extract_self_name_aliases(text: str) -> list[tuple[str, float]]:
    """Mine self-referential name mentions from raw message text.

    Matches 1-2-token names after first-person markers like ``my name is X`` /
    ``call me X`` / ``goes by X``. Bind results to the message author only.
    """
    return _collect(text, _SELF_NAME_PATTERNS)


def extract_stated_name_aliases(text: str) -> list[tuple[str, float]]:
    """Mine name statements from extracted fact text (third-person safe).

    Matches ``name is X`` / ``real name is X`` in LLM-normalized facts; the
    caller must bind results to the fact's subject, never the speaker.
    """
    return _collect(text, _STATED_NAME_PATTERNS)


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
