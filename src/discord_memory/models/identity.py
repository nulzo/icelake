"""Identity models: alias sources, resolutions, and the ambiguity contract."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from discord_memory.models.common import FrozenModel


class AliasSource(StrEnum):
    """Provenance of an alias row. Higher ranks bind harder during resolution."""

    DISCORD_USERNAME = "discord_username"
    SUBJECT_USERNAME = "subject_username"
    REAL_NAME = "real_name"
    DISPLAY_NAME = "display_name"
    MENTION = "mention"
    BACKFILL = "backfill"
    ENTITY_TAG = "entity_tag"

    @property
    def rank(self) -> int:
        return _RANKS[self]


_RANKS: dict[AliasSource, int] = {
    AliasSource.DISCORD_USERNAME: 100,
    AliasSource.SUBJECT_USERNAME: 95,
    AliasSource.REAL_NAME: 85,
    AliasSource.DISPLAY_NAME: 70,
    AliasSource.MENTION: 60,
    AliasSource.BACKFILL: 50,
    AliasSource.ENTITY_TAG: 40,
}


class ResolvedCandidate(FrozenModel):
    """One candidate identity for an identifier."""

    user_id: str
    matched_alias: str
    source: AliasSource
    weight: float = Field(default=0.5, ge=0.1, le=3.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Resolution(FrozenModel):
    """Outcome of name→ID resolution. ``ambiguous`` never guesses (PLAN.md §3.2)."""

    identifier: str
    resolved: ResolvedCandidate | None = None
    candidates: tuple[ResolvedCandidate, ...] = ()
    ambiguous: bool = False

    @property
    def basis(self) -> str:
        if self.resolved is None:
            return "unresolved"
        if self.resolved.matched_alias == self.identifier.lower():
            return f"alias:{self.resolved.source.value}"
        return f"fuzzy:{self.resolved.source.value}"


class AliasRecord(FrozenModel):
    """Stored alias binding."""

    guild_id: str
    alias_norm: str
    user_id: str
    source: AliasSource
    weight: float = 0.5
    updated_at: datetime | None = None
