"""Facts API group: CRUD, audit, explicit memory (API.md Part 7).

Cohesion exception: the explicit keyword facade remains in this module to preserve
its public API. Remember is longer than the size guideline because validation,
request fingerprinting and the transactional receipt must be reviewed together;
the actual fact mutation and projection recovery have separate entry points.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from hashlib import sha256
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from icelake.config import MemoryConfig
from icelake.errors import (
    FactNotFoundError,
    SchemaValidationError,
    SubjectNotAllowedError,
)
from icelake.graph.writes import DirectRoster, write_fact_graph
from icelake.identity.aliases import normalize_alias, weight_for_source
from icelake.identity.guards import SubjectGate
from icelake.ingest.gates import (
    normalize_text,
    text_hygiene_gate,
)
from icelake.models.common import FrozenModel, Page, TokenUsage
from icelake.models.events import FactCommitted, FactSupersededEvent
from icelake.models.facts import (
    Attribution,
    AttributionType,
    FactCategory,
    FactHistoryEntry,
    FactHistoryKind,
    FactRecord,
    FactScope,
    MemoryTier,
    SourceRef,
)
from icelake.models.identity import AliasSource
from icelake.models.operations import ProposedEntity, ProposedRelation
from icelake.models.recovery import ProjectionJob
from icelake.ports.clock import Clock, IdGen
from icelake.ports.llm import Embedder
from icelake.ports.store import MemoryStore
from icelake.ports.vectors import VectorIndex
from icelake.recovery import recover_projections

if TYPE_CHECKING:
    from icelake.api.events import EventBus


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
        event_bus: EventBus | None = None,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self._clock = clock
        self._id_gen = id_gen
        self._config = config
        self._gate = subject_gate
        self._startup_gate = startup_gate or _noop_gate_async
        self._event_bus = event_bus

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
        source_refs: tuple[SourceRef, ...] = (),
        idempotency_key: str | None = None,
    ) -> FactRecord:
        """Manually remember a fact. Manual facts land in CORE tier (never expire).

        ``idempotency_key`` makes command retries return the original result.
        Reusing a key with different inputs raises; receipts survive process restarts.

        ``speaker_id`` records third-party attribution when the actor states
        something about someone else (subject stays the person it's about).

        ``entities`` and ``relations`` participate in the knowledge graph exactly
        like extracted facts — reference endpoints by raw user id tokens or entity
        names, e.g.::

            relations=(ProposedRelation(
                verb=RelationVerb.LIKES, from_token=bob_id, to_entity="pizza"),
            )
        """
        await self._startup_gate()
        if any(ref.guild_id != guild_id for ref in source_refs):
            raise SchemaError("source reference belongs to another guild")
        participants = tuple(
            dict.fromkeys(
                uid
                for uid in (
                    subject_id,
                    speaker_id,
                    actor_id,
                    *(ref.author_id for ref in source_refs),
                )
                if uid
            )
        )
        if not (await self._gate.allowed_ids(guild_id, participants)).issuperset(participants):
            raise SubjectNotAllowedError("a fact participant is barred from memory")
        governance = await self._store.governance_generations(guild_id, participants)
        hygiene = text_hygiene_gate(text)
        if not hygiene.allowed:
            raise SchemaError(hygiene.reason)

        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 256:
            raise SchemaError("idempotency_key must contain 1 to 256 characters")
        receipt_key = (
            "manual:" + sha256(idempotency_key.encode()).hexdigest()
            if idempotency_key is not None
            else None
        )
        payload = {
            "guild_id": guild_id,
            "subject_id": subject_id,
            "text": text,
            "category": category,
            "confidence": confidence,
            "actor_id": actor_id,
            "speaker_id": speaker_id,
            "subject_username": subject_username,
            "attribution": attribution,
            "entities": [e.model_dump(mode="json") for e in entities],
            "relations": [r.model_dump(mode="json") for r in relations],
            "source_refs": [r.model_dump(mode="json") for r in source_refs],
        }
        fingerprint = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        replayed = False
        async with self._store.transaction():
            await self._store.set_cursor(guild_id, "commit_fence", uuid4().hex)
            raw = await self._store.get_cursor(guild_id, receipt_key) if receipt_key else None
            if raw is not None:
                receipt = _RememberReceipt.model_validate_json(raw)
                if receipt.request_hash != fingerprint:
                    raise SchemaError("idempotency key was already used with different inputs")
                restored = await self._store.get_fact(guild_id, receipt.fact_id)
                if restored is None:
                    raise FactNotFoundError(receipt.fact_id)
                record = restored
                replayed = True
            else:
                record = await self._remember(
                    guild_id=guild_id,
                    subject_id=subject_id,
                    text=text,
                    category=category,
                    confidence=confidence,
                    actor_id=actor_id,
                    speaker_id=speaker_id,
                    entities=entities,
                    relations=relations,
                    subject_username=subject_username,
                    attribution=attribution,
                    source_refs=source_refs,
                    governance=governance,
                )
                if receipt_key is not None:
                    await self._store.set_cursor(
                        guild_id,
                        receipt_key,
                        _RememberReceipt(
                            request_hash=fingerprint, fact_id=record.id
                        ).model_dump_json(),
                    )
        await recover_projections(
            self._store,
            self._vectors,
            self._embedder,
            limit=self._config.recovery.projection_batch_size,
            timeout_seconds=self._config.recovery.projection_timeout_seconds,
            retry_delay_seconds=self._config.recovery.retry_initial_seconds,
        )
        if not replayed:
            self._publish_committed(record, was_reinforcement=record.occurrences > 1)
        return record

    async def _remember(
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
        source_refs: tuple[SourceRef, ...] = (),
        governance: dict[str, int],
    ) -> FactRecord:
        current = await self._store.governance_generations(guild_id, tuple(governance))
        if any(current.get(uid, 0) > generation for uid, generation in governance.items()):
            raise SubjectNotAllowedError("a participant was purged during the command")
        normalized = normalize_text(text)
        duplicate = await self._store.find_duplicate(guild_id, subject_id, normalized)
        now = self._clock.now()
        if duplicate is not None and {ref.channel_id for ref in duplicate.citations} != {
            ref.channel_id for ref in source_refs
        }:
            duplicate = None
        if duplicate is not None:
            from icelake.governance import fact_participants

            for user_id in fact_participants(duplicate):
                if not await self._gate.allows(guild_id, user_id):
                    raise SubjectNotAllowedError("a fact participant is barred from memory")
            updated = await self._store.reinforce_fact(
                guild_id,
                duplicate.id,
                occurrences_delta=1,
                strength=duplicate.strength + 1.0,
                last_reinforced_at=now,
                expires_at=duplicate.expires_at,
                tier=duplicate.tier,
                confidence=max(duplicate.confidence, confidence),
            )
            assert updated is not None
            await self._store.enqueue_projection(
                ProjectionJob(guild_id=guild_id, fact_id=updated.id, version=updated.version)
            )
            return updated

        third_party = bool(speaker_id and speaker_id != subject_id)
        stamped = governance
        record = FactRecord(
            id=self._id_gen.new_id("fct"),
            guild_id=guild_id,
            subject_id=subject_id,
            text=text.strip(),
            text_normalized=normalized,
            category=category,
            confidence=confidence,
            tier=MemoryTier.CORE,
            scope=FactScope.SERVER if subject_id is None else FactScope.USER,
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
            governance=stamped,
            citations=source_refs,
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
        async with self._store.transaction():
            await self._store.insert_fact(record)
            await self._store.enqueue_projection(
                ProjectionJob(guild_id=guild_id, fact_id=record.id, version=record.version)
            )
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
        if record is None or not await self._gate.allows_fact(record):
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
        async with self._store.transaction():
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
                FactHistoryEntry(at=now, kind=FactHistoryKind.SUPERSEDED, detail=detail),
            )
            await self._store.enqueue_projection(
                ProjectionJob(guild_id=guild_id, fact_id=fact_id, version=updated_fields.version)
            )
        await recover_projections(self._store, self._vectors, self._embedder, limit=32)
        self._publish_superseded(guild_id, fact_id, fact_id, reason)
        return updated_fields

    async def forget(
        self, fact_id: str, *, guild_id: str, reason: str = "", actor_id: str | None = None
    ) -> None:
        """Soft-invalidate a fact (reversible via history; purge erases fully)."""
        await self.get(guild_id, fact_id)
        now = self._clock.now()
        async with self._store.transaction():
            await self._store.transition_fact(guild_id, fact_id, valid_until=now, updated_at=now)
            await self._store.drop_evidence_from_edges(guild_id, fact_id, until=now)
            await self._store.append_history(
                guild_id,
                fact_id,
                FactHistoryEntry(
                    at=now,
                    kind=FactHistoryKind.INVALIDATED,
                    detail=reason or f"by {actor_id or 'admin'}",
                ),
            )
        self._publish_superseded(guild_id, fact_id, None, reason)

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
            tier=record.tier,
            confidence=record.confidence,
        )
        assert updated is not None
        self._publish_committed(updated, was_reinforcement=True)
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
        await self._startup_gate()
        page = await self._store.list_facts(
            guild_id,
            subject_id=subject_id,
            include_server=include_server,
            active_only=active_only,
            limit=limit,
            cursor=cursor,
        )
        return Page(
            items=await self._gate.filter_records(guild_id, page.items),
            next_cursor=page.next_cursor,
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
        await self._startup_gate()
        hits = await self._store.search_facts_text(
            guild_id,
            query,
            subject_ids=subject_ids,
            server_only=server_only,
            limit=limit,
        )
        allowed = {
            record.id
            for record in await self._gate.filter_records(
                guild_id, tuple(record for record, _score in hits)
            )
        }
        return tuple((record, score) for record, score in hits if record.id in allowed)

    def _publish_committed(self, record: FactRecord, *, was_reinforcement: bool) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                FactCommitted(
                    guild_id=record.guild_id,
                    fact_id=record.id,
                    subject_id=record.subject_id,
                    text=record.text,
                    was_reinforcement=was_reinforcement,
                )
            )

    def _publish_superseded(
        self, guild_id: str, old_fact_id: str, new_fact_id: str | None, reason: str
    ) -> None:
        if self._event_bus is not None:
            self._event_bus.publish(
                FactSupersededEvent(
                    guild_id=guild_id,
                    old_fact_id=old_fact_id,
                    new_fact_id=new_fact_id,
                    reason=reason,
                )
            )


class _RememberReceipt(FrozenModel):
    request_hash: str
    fact_id: str


class SchemaError(SchemaValidationError):
    """Manual fact rejected by hygiene gates (catchable as SchemaValidationError)."""


__all__ = ["FactsApi", "SchemaError", "TokenUsage"]
