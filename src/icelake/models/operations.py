"""Extraction boundary schemas: roster tokens, proposed facts, reconcile decisions.

The LLM never sees or emits Discord snowflakes. It references batch
participants by opaque tokens we mint (``p0``, ``p1``, ...) and unknown people by name
strings that become entity references. Every payload crossing the LLM boundary is a
strict Pydantic schema — invalid output is never stored.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, field_validator

from icelake.models.common import FrozenModel
from icelake.models.facts import FactCategory
from icelake.models.graph import EntityKind, RelationVerb


class ProposedEntity(FrozenModel):
    """Named entity a fact is about; becomes/links an entity node."""

    name: str = Field(min_length=1, max_length=64)
    kind: EntityKind = EntityKind.CONCEPT

    @field_validator("kind", mode="before")
    @classmethod
    def _coerce_kind(cls, value: object) -> str:
        text = str(value or EntityKind.CONCEPT).strip().lower()
        return text if text in set(EntityKind) else EntityKind.CONCEPT


class ProposedRelation(FrozenModel):
    """Typed edge proposal. Endpoints reference roster tokens or entity names.

    ``verb`` accepts any string — extraction produces an open vocabulary —
    but prefer :class:`RelationVerb` members, which carry a polarity mapping.
    """

    verb: RelationVerb | str = Field(min_length=2, max_length=48)
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
    category: FactCategory = FactCategory.GENERAL
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    source_message_indexes: tuple[int, ...] = ()
    entities: tuple[ProposedEntity, ...] = ()
    relations: tuple[ProposedRelation, ...] = ()

    @field_validator("category", mode="before")
    @classmethod
    def _coerce_category(cls, value: object) -> str:
        text = str(value or FactCategory.GENERAL).strip().lower()
        return text if text in set(FactCategory) else FactCategory.GENERAL


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
    """One reconciliation outcome against an existing neighbor fact.

    ``candidate_index`` identifies which candidate in a batched reconcile call
    this decision belongs to; ``target_id`` carries the integer id remapping
    shown in the prompt (mapped back to real fact ids by the reconciler).
    """

    kind: ReconcileKind
    candidate_index: int = Field(default=0, ge=0)
    target_id: str | None = None
    text: str | None = None
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reason: str = ""

    @field_validator("target_id", mode="before")
    @classmethod
    def _empty_to_none(cls, value: object) -> object:
        if value in ("", "null", None):
            return None
        return str(value)


class ReconcileOutput(FrozenModel):
    decisions: tuple[ReconcileDecision, ...] = ()
