"""Governance and operations boundary models."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import model_validator

from icelake.models.common import FrozenModel
from icelake.models.facts import FactRecord, FactScope, MemoryTier
from icelake.models.graph import EntityRecord, RelationEdge


class PurgeReport(FrozenModel):
    """Snapshot removal counts, not proof of atomic or complete physical erasure.

    Concurrent writers and indirectly attributed data may remain; consent is preserved.
    """

    guild_id: str
    subject_id: str | None = None
    dry_run: bool = True
    facts_removed: int = 0
    links_removed: int = 0
    edges_removed: int = 0
    aliases_removed: int = 0
    summaries_removed: int = 0
    vectors_removed: int = 0
    messages_removed: int = 0


class ConsentState(FrozenModel):
    """Durable governance state for one user: consent flag plus erasure generation.

    ``generation`` increments on every executed purge. Queued messages and facts
    carry the generation they were created under; a mismatch against the current
    generation fences pre-purge data out of every read and commit, across
    processes, without relying on check-then-write atomicity.
    """

    opted_out: bool = False
    generation: int = 0


class GuildStats(FrozenModel):
    """Per-guild health and usage snapshot (``memory.stats``)."""

    guild_id: str
    total_facts: int = 0
    active_facts: int = 0
    by_tier: Mapping[MemoryTier, int] = {}
    by_scope: Mapping[FactScope, int] = {}
    user_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    pending_messages: int = 0
    in_flight_messages: int = 0  # claimed by a worker, not yet committed
    dead_letters: int = 0


class MemoryExport(FrozenModel):
    """Wire-stable data export for compliance flows (API.md Part 10)."""

    schema_version: Literal[1] = 1
    guild_id: str
    exported_at: datetime | None = None
    facts: tuple[FactRecord, ...] = ()
    entities: tuple[EntityRecord, ...] = ()
    relations: tuple[RelationEdge, ...] = ()

    @model_validator(mode="after")
    def validate_guild(self) -> MemoryExport:
        records: tuple[FactRecord | EntityRecord | RelationEdge, ...] = (
            *self.facts,
            *self.entities,
            *self.relations,
        )
        if any(record.guild_id != self.guild_id for record in records):
            raise ValueError("export records must belong to its declared guild")
        if len({record.id for record in self.facts}) != len(self.facts):
            raise ValueError("export contains duplicate fact IDs")
        return self


class HealthStatus(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


class ComponentHealth(FrozenModel):
    component: str
    status: HealthStatus
    detail: str = ""


class HealthReport(FrozenModel):
    components: tuple[ComponentHealth, ...] = ()
    pending_messages: int = 0
    dead_letters: int = 0

    @property
    def healthy(self) -> bool:
        return all(c.status is not HealthStatus.DOWN for c in self.components)


class BudgetStep(StrEnum):
    """Graceful-degradation ladder when budgets bind."""

    NONE = "none"
    SKIP_RECONCILE = "skip_reconcile"
    SKIP_EXTRACTION = "skip_extraction"
    SKIP_CONSOLIDATION = "skip_consolidation"


class MeterPurpose(StrEnum):
    """The library's own LLM call purposes — the keys used in meter snapshots.

    The set is open: ``ChatRequest.purpose`` accepts any string so consumers
    can attribute their own calls (e.g. ``"reply"``). These members cover
    every purpose the library itself charges.
    """

    GENERAL = "general"
    EXTRACTION = "extraction"
    RECONCILE = "reconcile"
    SUMMARIZE = "summarize"
    CLASSIFY_COMMAND = "classify_command"
    CITATION = "citation"


class MeterSnapshot(FrozenModel):
    """Cumulative counters keyed by purpose (see :class:`MeterPurpose`).

    Keys are strings because consumers can charge their own purposes; the
    library's own keys always match a :class:`MeterPurpose` value.
    """

    counters: Mapping[str, float] = {}
    calls: Mapping[str, int] = {}
    prompt_tokens: Mapping[str, int] = {}
    completion_tokens: Mapping[str, int] = {}
    estimated_cost_usd: Mapping[str, float] = {}

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
