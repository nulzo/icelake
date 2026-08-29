"""Fact committer: turns vetted candidates and reconcile decisions into storage.

Owns the transactional write rules: fact insert + embedding +
incidence links + entity nodes + relation edges + history, with supersede/invalidate
transitions that never hard-delete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from icelake.config import MemoryConfig
from icelake.ingest.gates import normalize_text
from icelake.lifecycle.strength import reinforced_strength
from icelake.lifecycle.tiers import assign_tier
from icelake.models.facts import (
    Attribution,
    AttributionType,
    FactHistoryEntry,
    FactHistoryKind,
    FactRecord,
    FactScope,
    SourceRef,
)
from icelake.models.operations import ProposedFact
from icelake.ports.clock import Clock, IdGen
from icelake.ports.llm import Embedder
from icelake.ports.store import MemoryStore
from icelake.ports.vectors import VectorIndex, VectorItem

logger = logging.getLogger(__name__)

MAX_CITATIONS_PER_FACT = 8


@dataclass(slots=True)
class CommitSummary:
    adds: int = 0
    reinforces: int = 0
    supersessions: int = 0
    invalidations: int = 0


class FactCommitter:
    """Applies extraction outcomes to the store under the pipeline's lease."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        vectors: VectorIndex | None,
        embedder: Embedder | None,
        clock: Clock,
        id_gen: IdGen,
        config: MemoryConfig,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self._clock = clock
        self._id_gen = id_gen
        self._config = config

    async def commit_add(
        self,
        *,
        proposal: ProposedFact,
        subject_id: str | None,
        speaker_id: str | None,
        guild_id: str,
        roster: RosterLike,
        supersedes_id: str | None = None,
        mentioned_ids: tuple[str, ...] = (),
        source_refs: tuple[SourceRef, ...] = (),
        skip_embedding: bool = False,
    ) -> FactRecord:
        now = self._clock.now()
        text = roster.bind_names(proposal.text.strip())
        third_party = bool(speaker_id and subject_id and speaker_id != subject_id)
        named_id = speaker_id if third_party else subject_id
        tier, expires_after = assign_tier(
            text=text,
            category=proposal.category,
            confidence=proposal.confidence,
            occurrences=1,
            manual=False,
            is_server_fact=subject_id is None,
            lifecycle=self._config.lifecycle,
        )
        record = FactRecord(
            id=self._id_gen.new_id("fct"),
            guild_id=guild_id,
            subject_id=subject_id,
            text=text,
            text_normalized=normalize_text(text),
            category=proposal.category,
            confidence=proposal.confidence,
            tier=tier,
            scope=FactScope.SERVER if subject_id is None else FactScope.USER,
            attribution=Attribution(
                type=AttributionType.THIRD_PARTY if third_party else AttributionType.SELF,
                speaker_id=speaker_id if third_party else None,
                speaker_name=roster.display_name(named_id) if named_id else None,
            ),
            strength=1.0,
            last_reinforced_at=now,
            created_at=now,
            updated_at=now,
            observed_at=now,
            valid_from=now,
            expires_at=(now + expires_after) if expires_after is not None else None,
            supersedes_id=supersedes_id,
            citations=source_refs,
        )
        record = record.model_copy(
            update={
                "related_user_ids": tuple(
                    dict.fromkeys(
                        uid
                        for uid in (*record.related_user_ids, *mentioned_ids)
                        if uid != record.subject_id
                    )
                ),
            }
        )
        # Embed BEFORE the unit of work: network I/O never happens inside a
        # database transaction (a held write lock on a single-writer store
        # would stall every other guild's commits for the embed's latency).
        embedding = await self._embed(record) if not skip_embedding else None
        async with self._store.transaction():
            await self._store.insert_fact(record)
            if embedding is not None:
                await self._upsert_embedding(record, embedding)
            await self._write_graph(
                guild_id=guild_id,
                record=record,
                proposal=proposal,
                roster=roster,
                mentioned_ids=mentioned_ids,
            )
            await self._store.append_history(
                guild_id,
                record.id,
                FactHistoryEntry(
                    at=now,
                    kind=FactHistoryKind.CREATED,
                    detail=f"extracted conf={proposal.confidence:.2f}",
                ),
            )
        return record

    async def commit_reinforce(self, existing: FactRecord, proposal: ProposedFact) -> FactRecord:
        now = self._clock.now()
        new_strength = reinforced_strength(existing.strength)
        new_confidence = max(existing.confidence, proposal.confidence)
        tier, expires_after = assign_tier(
            text=existing.text,
            category=existing.category,
            confidence=new_confidence,
            occurrences=existing.occurrences + 1,
            manual=existing.attribution.type is AttributionType.MANUAL,
            is_server_fact=existing.is_server_fact,
            lifecycle=self._config.lifecycle,
        )
        updated = await self._store.reinforce_fact(
            existing.guild_id,
            existing.id,
            occurrences_delta=1,
            strength=new_strength,
            last_reinforced_at=now,
            expires_at=(now + expires_after) if expires_after is not None else None,
            tier=tier,
            confidence=new_confidence,
        )
        return updated or existing

    async def commit_supersede(
        self,
        *,
        old_record: FactRecord,
        proposal: ProposedFact,
        subject_id: str | None,
        speaker_id: str | None,
        reason: str,
        guild_id: str,
        roster: RosterLike,
        mentioned_ids: tuple[str, ...] = (),
        source_refs: tuple[SourceRef, ...] = (),
    ) -> tuple[FactRecord, FactRecord]:
        """Insert refined fact linked to the old one; old stays queryable history."""
        fresh = await self.commit_add(
            proposal=proposal,
            subject_id=subject_id or old_record.subject_id,
            speaker_id=speaker_id,
            guild_id=guild_id,
            roster=roster,
            supersedes_id=old_record.id,
            mentioned_ids=mentioned_ids,
            source_refs=source_refs,
        )
        now = self._clock.now()
        # Second unit of work: retire the old fact. Kept separate from the
        # insert above so the embed never runs inside a transaction; a crash
        # between the two leaves the old fact active, which reconcile will
        # re-collide with the fresh one — self-healing, not corrupt.
        async with self._store.transaction():
            await self._store.transition_fact(
                guild_id,
                old_record.id,
                superseded_by_id=fresh.id,
                valid_until=now,
                updated_at=now,
            )
            # The old fact is dead knowledge; its evidence must stop holding
            # relation edges alive (same treatment as commit_invalidate).
            await self._store.drop_evidence_from_edges(guild_id, old_record.id, until=now)
            await self._store.append_history(
                guild_id,
                old_record.id,
                FactHistoryEntry(
                    at=now,
                    kind=FactHistoryKind.SUPERSEDED,
                    detail=reason or "refined",
                ),
            )
        return fresh, old_record

    async def commit_invalidate(self, *, old_record: FactRecord, reason: str) -> FactRecord | None:
        now = self._clock.now()
        async with self._store.transaction():
            updated = await self._store.transition_fact(
                old_record.guild_id,
                old_record.id,
                valid_until=now,
                updated_at=now,
            )
            detached = await self._store.drop_evidence_from_edges(
                old_record.guild_id,
                old_record.id,
                until=now,
            )
            if updated is not None:
                await self._store.append_history(
                    old_record.guild_id,
                    old_record.id,
                    FactHistoryEntry(
                        at=now,
                        kind=FactHistoryKind.INVALIDATED,
                        detail=reason or "contradicted",
                    ),
                )
        logger.debug("Invalidated %s; detached %d edge evidences", old_record.id, detached)
        return updated

    async def _embed(self, record: FactRecord) -> tuple[float, ...] | None:
        if self._vectors is None or self._embedder is None:
            return None
        (embedding,) = await self._embedder.embed((record.text,))
        return embedding

    async def _upsert_embedding(self, record: FactRecord, embedding: tuple[float, ...]) -> None:
        assert self._vectors is not None
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

    async def _write_graph(
        self,
        *,
        guild_id: str,
        record: FactRecord,
        proposal: ProposedFact,
        roster: RosterLike,
        mentioned_ids: tuple[str, ...] = (),
    ) -> None:
        from icelake.graph.writes import write_fact_graph

        await write_fact_graph(
            store=self._store,
            clock_now=self._clock.now(),
            guild_id=guild_id,
            record=record,
            entities=proposal.entities,
            relations=proposal.relations,
            mentioned_ids=mentioned_ids,
            roster=roster,
        )


class RosterLike(Protocol):
    """Structural subset of :class:`~icelake.ingest.roster.Roster`."""

    def knows(self, token: str) -> bool: ...

    def user_id_for(self, token: str) -> str | None: ...

    def bind_names(self, text: str) -> str: ...

    def display_name(self, user_id: str) -> str | None: ...


__all__ = ["CommitSummary", "FactCommitter", "RosterLike"]
