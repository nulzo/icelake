"""Fact records: the atomic unit of memory (PLAN.md Part 4.7 anchoring invariant)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from icelake.models.common import FrozenModel


class MemoryTier(StrEnum):
    """Lifecycle tier. CORE never expires; prune priority follows declaration order."""

    SHORT_TERM = "short_term"
    MID_TERM = "mid_term"
    LONG_TERM = "long_term"
    CORE = "core"

    @property
    def prune_priority(self) -> int:
        return _PRUNE_PRIORITY[self]


_PRUNE_PRIORITY: dict[MemoryTier, int] = {
    MemoryTier.SHORT_TERM: 0,
    MemoryTier.MID_TERM: 1,
    MemoryTier.LONG_TERM: 2,
    MemoryTier.CORE: 3,
}


class FactCategory(StrEnum):
    """Closed vocabulary for fact classification."""

    PERSONAL = "personal"
    PREFERENCES = "preferences"
    INTERESTS = "interests"
    PROFESSIONAL = "professional"
    RELATIONSHIPS = "relationships"
    GOALS = "goals"
    EXPERIENCES = "experiences"
    PERSONALITY = "personality"
    CULTURE = "culture"
    RULES = "rules"
    GENERAL = "general"


class AttributionType(StrEnum):
    """How a fact came to be attributed to its subject."""

    SELF = "self"
    THIRD_PARTY = "third_party"
    MANUAL = "manual"
    INFERRED = "inferred"
    AGENT = "agent"


class SourceRole(StrEnum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"


class Attribution(FrozenModel):
    """Who a fact is about and, when applicable, who stated it."""

    type: AttributionType = AttributionType.SELF
    speaker_id: str | None = None
    speaker_name: str | None = None
    actor_id: str | None = None


class SourceRef(FrozenModel):
    """Citation-ready snapshot of one supporting Discord message.

    Content is snapshotted at ingest so jump links and quotes survive message
    deletion or edits.
    """

    message_id: str
    channel_id: str
    guild_id: str
    author_id: str
    author_name: str = ""
    content_snippet: str = ""
    created_at: datetime | None = None
    message_url: str = ""
    role: SourceRole = SourceRole.SUPPORTING


class FactRecord(FrozenModel):
    """One stored memory. Exactly one anchor: ``subject_id`` or ``guild_id`` (server).

    ``subject_id is None`` marks a server-wide fact. Validity is bitemporal-lite:
    ``valid_from``/``valid_until`` describe world time; ``observed_at``/``updated_at``
    describe ingestion order. Nothing is ever hard-deleted except by explicit purge.
    """

    id: str
    guild_id: str
    subject_id: str | None = None
    text: str
    text_normalized: str = ""
    category: FactCategory = FactCategory.GENERAL
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    tier: MemoryTier = MemoryTier.SHORT_TERM
    scope: Literal["user", "server"] = "user"
    attribution: Attribution = Field(default_factory=Attribution)

    occurrences: int = Field(default=1, ge=1)
    strength: float = Field(default=1.0, ge=1.0)
    last_reinforced_at: datetime | None = None

    created_at: datetime | None = None
    updated_at: datetime | None = None
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    supersedes_id: str | None = None
    superseded_by_id: str | None = None

    citations: tuple[SourceRef, ...] = ()
    related_user_ids: tuple[str, ...] = ()
    entity_slugs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    expires_at: datetime | None = None
    version: int = 1

    @property
    def is_server_fact(self) -> bool:
        return self.subject_id is None

    @property
    def is_active(self) -> bool:
        return self.superseded_by_id is None and self.valid_until is None

    def with_updates(self, **fields: object) -> FactRecord:
        """Functional copy helper (records are frozen)."""
        return self.model_copy(update=dict(fields))


class FactHistoryEntry(FrozenModel):
    """One lineage event in a fact's audit trail."""

    at: datetime
    kind: Literal["created", "reinforced", "superseded", "invalidated", "reinstated"]
    detail: str = ""
    fact_version: int = 1


class ProfileSummary(FrozenModel):
    """Consolidated per-subject or per-guild digest (derived representation).

    Regenerated asynchronously by consolidation; keyed by ``subject_id`` with ``None``
    meaning the guild digest.
    """

    guild_id: str
    subject_id: str | None
    text: str
    generated_at: datetime | None = None
    source_fact_count: int = 0
