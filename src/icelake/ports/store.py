"""Storage port: the single contract every persistence backend implements.

The protocol is deliberately backend-agnostic (no SQL/Mongo shapes leak). New backends
must pass the conformance suite in ``tests/integration/test_store_conformance.py``.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol, runtime_checkable

from icelake.models.admin import GuildStats, PurgeReport
from icelake.models.common import Page
from icelake.models.facts import (
    FactCategory,
    FactHistoryEntry,
    FactRecord,
    ProfileSummary,
    SourceRef,
)
from icelake.models.graph import (
    EdgeKind,
    EntityKind,
    EntityRecord,
    LinkRow,
    NodeType,
    Polarity,
    RelationEdge,
)
from icelake.models.identity import AliasRecord, AliasSource

NodeRef = tuple[NodeType, str]

__all__ = ["MemoryStore", "NodeRef"]


@runtime_checkable
class MemoryStore(Protocol):
    """Durable state: facts, identity, graph, derived summaries, governance flags.

    All methods are async; implementations must never block the event loop. Facts are
    addressed by ``(guild_id, id)``. ``subject_id=None`` denotes server-wide facts.
    """

    # -- lifecycle ------------------------------------------------------------
    async def setup(self) -> None: ...
    async def close(self) -> None: ...
    async def ping(self) -> bool: ...

    def transaction(self) -> AbstractAsyncContextManager[None]:
        """Unit-of-work scope for multi-statement fact commits.

        Backends without native transactions may yield a no-op; call sites must
        not rely on isolation beyond best-effort atomicity.
        """
        ...

    # -- aliases (identity layer 1) -------------------------------------------
    async def upsert_alias(
        self,
        guild_id: str,
        alias_norm: str,
        user_id: str,
        source: AliasSource,
        weight: float,
    ) -> None: ...

    async def resolve_alias_candidates(
        self,
        guild_id: str,
        alias_norm: str,
        limit: int = 8,
    ) -> tuple[AliasRecord, ...]: ...

    async def prefix_alias_candidates(
        self,
        guild_id: str,
        prefix_norm: str,
        limit: int = 8,
    ) -> tuple[AliasRecord, ...]: ...

    async def aliases_for_user(self, guild_id: str, user_id: str) -> tuple[AliasRecord, ...]: ...

    async def delete_aliases_for_user(self, guild_id: str, user_id: str) -> int: ...

    # -- facts ----------------------------------------------------------------
    async def insert_fact(self, record: FactRecord) -> None: ...

    async def get_fact(self, guild_id: str, fact_id: str) -> FactRecord | None: ...

    async def get_facts(
        self,
        guild_id: str,
        fact_ids: tuple[str, ...],
    ) -> tuple[FactRecord, ...]: ...

    async def find_duplicate(
        self,
        guild_id: str,
        subject_id: str | None,
        text_normalized: str,
    ) -> FactRecord | None:
        """Exact-normalized duplicate among active facts for the same anchor."""

    async def reinforce_fact(
        self,
        guild_id: str,
        fact_id: str,
        *,
        occurrences_delta: int,
        strength: float,
        last_reinforced_at: datetime,
        expires_at: datetime | None,
        tier: str,
        confidence: float,
        extra_citations: tuple[SourceRef, ...] = (),
    ) -> FactRecord | None:
        """Merge a reinforcement observation into an existing fact."""

    async def transition_fact(
        self,
        guild_id: str,
        fact_id: str,
        *,
        valid_until: datetime | None = None,
        superseded_by_id: str | None = None,
        updated_at: datetime,
    ) -> FactRecord | None:
        """Soft-invalidate / supersede. History stays queryable."""

    async def update_fact_fields(
        self,
        guild_id: str,
        fact_id: str,
        *,
        text: str | None = None,
        text_normalized: str | None = None,
        category: FactCategory | None = None,
        confidence: float | None = None,
        tier: str | None = None,
        expires_at: datetime | None = None,
        updated_at: datetime,
    ) -> FactRecord | None: ...

    async def list_facts(
        self,
        guild_id: str,
        *,
        subject_id: str | None,
        include_server: bool = False,
        active_only: bool = True,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Page[FactRecord]:
        """List facts for one subject (or server facts when ``subject_id=None``)."""

    async def top_strength_facts(
        self,
        guild_id: str,
        *,
        subject_ids: tuple[str, ...] | None,
        server_only: bool = False,
        limit: int = 10,
        as_of: datetime | None = None,
    ) -> tuple[FactRecord, ...]:
        """``as_of`` switches to point-in-time validity (bitemporal read)."""

    async def search_facts_text(
        self,
        guild_id: str,
        query: str,
        *,
        subject_ids: tuple[str, ...] | None = None,
        server_only: bool = False,
        limit: int = 20,
        as_of: datetime | None = None,
    ) -> tuple[tuple[FactRecord, float], ...]:
        """Lexical search with relevance scores in [0, 1].

        ``as_of`` returns facts valid at that instant instead of now.
        """

    async def append_history(
        self,
        guild_id: str,
        fact_id: str,
        entry: FactHistoryEntry,
    ) -> None: ...

    async def get_history(
        self,
        guild_id: str,
        fact_id: str,
    ) -> tuple[FactHistoryEntry, ...]: ...

    # -- incidence links (graph layer 2) ---------------------------------------
    async def add_links(self, rows: tuple[LinkRow, ...]) -> None: ...

    async def links_for_node(
        self,
        guild_id: str,
        node_type: NodeType,
        node_id: str,
        *,
        kinds: tuple[EdgeKind, ...] | None = None,
        active_only: bool = True,
        limit: int = 100,
        as_of: datetime | None = None,
    ) -> tuple[tuple[LinkRow, FactRecord], ...]:
        """``as_of`` switches the join to point-in-time validity."""

    async def nodes_for_fact(self, guild_id: str, memory_id: str) -> tuple[LinkRow, ...]: ...

    async def links_for_nodes(
        self,
        guild_id: str,
        nodes: tuple[NodeRef, ...],
        *,
        active_only: bool = True,
        limit_per_node: int = 50,
    ) -> tuple[tuple[LinkRow, FactRecord], ...]:
        """One-round-trip batch of ``links_for_node``; per-node caps applied.

        Discovery paths (graph hop, pair recall) fan out over many nodes —
        per-node queries are the N+1 this replaces.
        """

    # -- relations (graph layer 3) ----------------------------------------------
    async def upsert_relation(self, edge: RelationEdge) -> RelationEdge:
        """Merge into the currently-active edge between the same pair+verb.

        Increments occurrences/weight, unions evidence ids, keeps bitemporal validity.
        """

    async def edges_between(
        self,
        guild_id: str,
        src: NodeRef,
        dst: NodeRef,
    ) -> tuple[RelationEdge, ...]: ...

    async def incident_edges(
        self,
        guild_id: str,
        node: NodeRef,
        *,
        limit: int = 50,
    ) -> tuple[RelationEdge, ...]: ...

    async def incident_edges_many(
        self,
        guild_id: str,
        nodes: tuple[NodeRef, ...],
        *,
        limit_per_node: int = 50,
    ) -> tuple[RelationEdge, ...]:
        """One-round-trip batch of ``incident_edges``; per-node caps applied."""

    async def edges_to_nodes(
        self,
        guild_id: str,
        nodes: tuple[NodeRef, ...],
        *,
        limit: int = 500,
    ) -> tuple[RelationEdge, ...]:
        """Active edges whose DESTINATION is any of ``nodes``, weight-ranked."""

    async def drop_evidence_from_edges(
        self,
        guild_id: str,
        fact_id: str,
        until: datetime,
    ) -> int:
        """Remove a fact from edge evidence; expire edges left without evidence."""

    async def entity_stance_edges(
        self,
        guild_id: str,
        entity_slug: str,
        *,
        polarity: Polarity | None = None,
        limit: int = 25,
    ) -> tuple[RelationEdge, ...]:
        """Edges whose destination is this entity node, weight-ranked."""

    # -- entities ---------------------------------------------------------------
    async def upsert_entity(
        self,
        guild_id: str,
        slug: str,
        name: str,
        kind: EntityKind,
        aliases: tuple[str, ...] = (),
    ) -> EntityRecord: ...

    async def bump_entity_facts(self, guild_id: str, slug: str, delta: int = 1) -> None: ...

    async def get_entity(self, guild_id: str, slug: str) -> EntityRecord | None: ...

    async def resolve_entity_alias(self, guild_id: str, alias_norm: str) -> str | None:
        """Map a normalized surface name to a canonical entity slug."""

    async def merge_entities(
        self,
        guild_id: str,
        from_slugs: tuple[str, ...],
        to_slug: str,
    ) -> int: ...

    # -- derived summaries --------------------------------------------------------
    async def get_summary(
        self,
        guild_id: str,
        subject_id: str | None,
    ) -> ProfileSummary | None: ...

    async def put_summary(self, summary: ProfileSummary) -> None: ...

    async def delete_summary(self, guild_id: str, subject_id: str | None) -> int: ...

    # -- consent & governance ------------------------------------------------------
    async def set_opt_out(self, guild_id: str, user_id: str, opted_out: bool) -> None: ...

    async def get_opt_out(self, guild_id: str, user_id: str) -> bool: ...

    async def purge_user_data(self, guild_id: str, user_id: str, dry_run: bool) -> PurgeReport: ...

    async def export_guild(
        self, guild_id: str
    ) -> tuple[
        tuple[FactRecord, ...],
        tuple[EntityRecord, ...],
        tuple[RelationEdge, ...],
    ]: ...

    async def sweep_expired(self, guild_id: str, now: datetime) -> int:
        """Soft-invalidate facts whose ``expires_at`` has passed. Returns count."""

    async def prune_to_caps(
        self,
        guild_id: str,
        *,
        max_per_user: int,
        max_server: int,
        now: datetime,
    ) -> int:
        """Enforce profile-size caps, pruning weakest first (tier, strength, conf)."""

    async def apply_forgetting(
        self,
        guild_id: str,
        *,
        now: datetime,
        retention_floor: float,
    ) -> int:
        """Soft-invalidate non-core facts below the retention floor."""

    async def get_cursor(self, guild_id: str, key: str) -> str | None:
        """Durable high-water mark (e.g. last community-window message id)."""

    async def set_cursor(self, guild_id: str, key: str, value: str) -> None:
        """Persist a high-water mark."""

    async def list_guild_ids(self) -> tuple[str, ...]:
        """Guilds with any stored state (facts or queued messages).

        Derived, not tracked: no hot-path writes, and maintenance survives
        restarts because the source tables are the state.
        """

    async def charge_guild_tokens(
        self,
        guild_id: str,
        *,
        day_key: str,
        month_key: str,
        prompt_tokens: int,
    ) -> tuple[int, int]:
        """Atomically add prompt spend; returns (day_total, month_total).

        Store-backed so budget enforcement is correct across N processes —
        an in-process counter under-enforces by the worker count.
        """

    async def guild_token_usage(
        self,
        guild_id: str,
        *,
        day_key: str,
        month_key: str,
    ) -> tuple[int, int]:
        """Current (day_total, month_total) prompt spend."""

    async def touch_facts(
        self,
        guild_id: str,
        fact_ids: tuple[str, ...],
        *,
        accessed_at: datetime,
    ) -> int:
        """Access-time reinforcement: reset the decay clock on recalled facts
        (MemoryBank / mem0-decay pattern). One batched write."""

    async def import_guild(
        self,
        facts: tuple[FactRecord, ...],
        entities: tuple[EntityRecord, ...],
        relations: tuple[RelationEdge, ...],
    ) -> int:
        """Restore a previously exported guild. Returns fact count inserted."""

    async def guild_stats(self, guild_id: str) -> GuildStats: ...
