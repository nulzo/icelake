"""Extraction boundary schemas: roster tokens, proposed facts, reconcile decisions.

The LLM never sees or emits Discord snowflakes (PLAN.md §3.1). It references batch
participants by opaque tokens we mint (``p0``, ``p1``, ...) and unknown people by name
strings that become entity references. Every payload crossing the LLM boundary is a
strict Pydantic schema — parse failures fall back, they never store garbage.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from discord_memory.models.common import FrozenModel


class ProposedEntity(FrozenModel):
    """Named entity a fact is about; becomes/links an entity node."""

    name: str = Field(min_length=1, max_length=64)
    kind: str = "concept"

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: object) -> str:
        text = str(value or "concept").strip().lower()
        return text if text in {"person", "place", "concept", "org"} else "concept"


class ProposedRelation(FrozenModel):
    """Typed edge proposal. Endpoints reference roster tokens or entity names."""

    verb: str = Field(min_length=2, max_length=48)
    from_token: str | None = None
    to_token: str | None = None
    from_entity: str | None = None
    to_entity: str | None = None


class ProposedFact(FrozenModel):
    """One candidate fact extracted from a batch.

    ``subject_token`` must be a minted roster token (or ``server``). ``speaker_token``
    optionally records who said it when different from the subject.
    """

    subject_token: str = Field(min_length=1)
    speaker_token: str | None = None
    text: str = Field(min_length=1, max_length=400)
    category: str = "general"
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_message_indexes: tuple[int, ...] = ()
    entities: tuple[ProposedEntity, ...] = ()
    relations: tuple[ProposedRelation, ...] = ()

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: object) -> str:
        allowed = {
            "personal",
            "preferences",
            "interests",
            "professional",
            "relationships",
            "goals",
            "experiences",
            "personality",
            "culture",
            "rules",
            "general",
        }
        text = str(value or "general").strip().lower()
        return text if text in allowed else "general"


class ExtractionOutput(FrozenModel):
    """Root schema of the extraction LLM response."""

    operations: tuple[ProposedFact, ...] = ()

    @field_validator("operations", mode="before")
    @classmethod
    def _coerce_operations(cls, value: object) -> object:
        return [] if not isinstance(value, list) else value


class ReconcileKind(StrEnum):
    ADD = "add"
    UPDATE = "update"
    INVALIDATE = "invalidate"
    NOOP = "noop"


class ReconcileDecision(FrozenModel):
    """One reconciliation outcome against an existing neighbor fact."""

    kind: ReconcileKind
    target_id: str | None = None
    text: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("target_id", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        if value in ("", "null", None):
            return None
        return value


class ReconcileOutput(FrozenModel):
    decisions: tuple[ReconcileDecision, ...] = ()
