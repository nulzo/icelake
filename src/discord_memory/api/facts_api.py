"""Facts API group: CRUD, audit, explicit memory (API.md Part 7)."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from discord_memory.config import MemoryConfig
from discord_memory.errors import (
    FactNotFoundError,
    SchemaValidationError,
    SubjectNotAllowedError,
)
from discord_memory.graph.writes import DirectRoster, write_fact_graph
from discord_memory.identity.aliases import normalize_alias, weight_for_source
from discord_memory.identity.guards import SubjectGate
from discord_memory.ingest.gates import (
    normalize_text,
    text_hygiene_gate,
)
from discord_memory.models.common import Page, TokenUsage
from discord_memory.models.facts import (
    Attribution,
    AttributionType,
    FactCategory,
    FactHistoryEntry,
    FactRecord,
    MemoryTier,
)
from discord_memory.models.identity import AliasSource
from discord_memory.models.operations import ProposedEntity, ProposedRelation
from discord_memory.ports.clock import Clock, IdGen
from discord_memory.ports.llm import Embedder
from discord_memory.ports.store import MemoryStore
from discord_memory.ports.vectors import VectorIndex, VectorItem


async def _noop_gate_async() -> None:
    """Default no-op lifecycle gate."""


class FactsApi:
    """Programmatic fact management — the ``memory.facts.*`` namespace."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        vectors: VectorIndex | None,
        embedder: Embedder | None,
        clock: Clock,
        id_gen: IdGen,
        config: MemoryConfig,
        subject_gate: SubjectGate,
        startup_gate: Callable[[], Any] | None = None,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self._clock = clock
        self._id_gen = id_gen
        self._config = config
        self._gate = subject_gate
        self._startup_gate = startup_gate or _noop_gate_async

    async def remember(
        self,
        *,
        guild_id: str,
        subject_id: str | None,
        text: str,
        category: FactCategory = FactCategory.GENERAL,
        confidence: float = 1.0,
        actor_id: str | None = None,
        speaker_id: str | None = None,
        entities: tuple[ProposedEntity, ...] = (),
        relations: tuple[ProposedRelation, ...] = (),
        subject_username: str = "",
        attribution: AttributionType = AttributionType.MANUAL,
    ) -> FactRecord:
        """Manually remember a fact. Manual facts land in CORE tier (never expire).

        ``speaker_id`` records third-party attribution when the actor states
        something about someone else (subject stays the person it's about).

        ``entities`` and ``relations`` participate in the knowledge graph exactly
        like extracted facts — reference endpoints by raw user id tokens or entity
        names, e.g.::

            relations=(ProposedRelation(
                verb="owes", from_token=bob_id, to_entity="pizza"),
            )
        """
        await self._startup_gate()
        if not await self._gate.allows(guild_id, subject_id):
            raise SubjectNotAllowedError(f"subject {subject_id} is barred from memory")
        hygiene = text_hygiene_gate(text)
        if not hygiene.allowed:
            raise SchemaError(hygiene.reason)

        normalized = normalize_text(text)
        duplicate = await self._store.find_duplicate(guild_id, subject_id, normalized)
        now = self._clock.now()
        if duplicate is not None:
            updated = await self._store.reinforce_fact(
                guild_id,
                duplicate.id,
                occurrences_delta=1,
                strength=duplicate.strength + 1.0,
                last_reinforced_at=now,
                expires_at=duplicate.expires_at,
                tier=duplicate.tier.value,
                confidence=max(duplicate.confidence, confidence),
            )
            assert updated is not None
            return updated

        third_party = bool(speaker_id and speaker_id != subject_id)
        record = FactRecord(
            id=self._id_gen.new_id("fct"),
            guild_id=guild_id,
            subject_id=subject_id,
            text=text.strip(),
            text_normalized=normalized,
            category=category,
            confidence=confidence,
            tier=MemoryTier.CORE,
            scope="server" if subject_id is None else "user",
            attribution=Attribution(
                type=(AttributionType.THIRD_PARTY if third_party else attribution),
                actor_id=actor_id,
                speaker_id=speaker_id if third_party else None,
            ),
            strength=1.0,
            last_reinforced_at=now,
            created_at=now,
            updated_at=now,
            observed_at=now,
            valid_from=now,
        )
        if subject_username and subject_id is not None:
            normalized = normalize_alias(subject_username)
            if len(normalized) >= 2 and not normalized.isdigit():
                await self._store.upsert_alias(
                    guild_id,
                    normalized,
                    subject_id,
                    AliasSource.SUBJECT_USERNAME,
                    weight_for_source(AliasSource.SUBJECT_USERNAME),
                )
        await self._store.insert_fact(record)
        await self._index(record)
        roster = DirectRoster(*(uid for uid in (subject_id, actor_id, speaker_id) if uid))
        await write_fact_graph(
            store=self._store,
            clock_now=now,
            guild_id=guild_id,
            record=record,
            entities=entities,
            relations=relations,
            mentioned_ids=tuple(uid for uid in (actor_id,) if uid and uid != subject_id),
            roster=roster,
        )
        return record

    async def get_all(
        self,
        guild_id: str,
        subject_id: str | None,
        *,
        include_server: bool = False,
        limit: int = 100,
    ) -> tuple[FactRecord, ...]:
        """All active facts for a member (mem0 ``get_all`` parity)."""
        page = await self.list_for_subject(
            guild_id,
            subject_id,
            include_server=include_server,
            active_only=True,
            limit=limit,
        )
        return page.items

    async def get(self, guild_id: str, fact_id: str) -> FactRecord:
        await self._startup_gate()
        record = await self._store.get_fact(guild_id, fact_id)
        if record is None:
            raise FactNotFoundError(fact_id)
        return record

    async def update(
        self,
        fact_id: str,
        *,
        guild_id: str,
        text: str,
        reason: str = "",
        actor_id: str | None = None,
    ) -> FactRecord:
        """Refine a fact's text; history records the correction."""
        _ = await self.get(guild_id, fact_id)
        now = self._clock.now()
        updated_fields = await self._store.update_fact_fields(
            guild_id,
            fact_id,
            text=text,
            text_normalized=normalize_text(text),
            updated_at=now,
        )
        assert updated_fields is not None
        detail = f"update: {reason or 'no reason given'} by {actor_id or 'unknown'}"
        await self._store.append_history(
            guild_id,
            fact_id,
            FactHistoryEntry(at=now, kind="superseded", detail=detail),
        )
        await self._index(updated_fields)  # stale vector would misdirect recall
        return updated_fields

    async def forget(
        self, fact_id: str, *, guild_id: str, reason: str = "", actor_id: str | None = None
    ) -> None:
        """Soft-invalidate a fact (reversible via history; purge erases fully)."""
        await self.get(guild_id, fact_id)
        now = self._clock.now()
        await self._store.transition_fact(guild_id, fact_id, valid_until=now, updated_at=now)
        await self._store.drop_evidence_from_edges(guild_id, fact_id, until=now)
        await self._store.append_history(
            guild_id,
            fact_id,
            FactHistoryEntry(
                at=now,
                kind="invalidated",
                detail=reason or f"by {actor_id or 'admin'}",
            ),
        )

    async def reinforce(self, fact_id: str, *, guild_id: str) -> FactRecord:
        record = await self.get(guild_id, fact_id)
        now = self._clock.now()
        updated = await self._store.reinforce_fact(
            guild_id,
            fact_id,
            occurrences_delta=1,
            strength=record.strength + 1.0,
            last_reinforced_at=now,
            expires_at=record.expires_at,
            tier=record.tier.value,
            confidence=record.confidence,
        )
        assert updated is not None
        return updated

    async def history(self, fact_id: str, *, guild_id: str) -> tuple[FactHistoryEntry, ...]:
        await self.get(guild_id, fact_id)
        return await self._store.get_history(guild_id, fact_id)

    async def list_for_subject(
        self,
        guild_id: str,
        subject_id: str | None,
        *,
        include_server: bool = True,
        active_only: bool = True,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Page[FactRecord]:
        return await self._store.list_facts(
            guild_id,
            subject_id=subject_id,
            include_server=include_server,
            active_only=active_only,
            limit=limit,
            cursor=cursor,
        )

    async def search(
        self,
        guild_id: str,
        query: str,
        *,
        subject_ids: tuple[str, ...] | None = None,
        server_only: bool = False,
        limit: int = 20,
    ) -> tuple[tuple[FactRecord, float], ...]:
        return await self._store.search_facts_text(
            guild_id,
            query,
            subject_ids=subject_ids,
            server_only=server_only,
            limit=limit,
        )

    async def _index(self, record: FactRecord) -> None:
        if self._vectors is None or self._embedder is None:
            return
        (embedding,) = await self._embedder.embed((record.text,))
        await self._vectors.upsert(
            (
                VectorItem(
                    id=record.id,
                    guild_id=record.guild_id,
                    subject_id=record.subject_id,
                    embedding=embedding,
                ),
            )
        )


class SchemaError(SchemaValidationError):
    """Manual fact rejected by hygiene gates (catchable as SchemaValidationError)."""


__all__ = ["FactsApi", "SchemaError", "TokenUsage"]
