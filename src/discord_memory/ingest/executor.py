"""Fact committer: turns vetted candidates and reconcile decisions into storage.

Owns the transactional write rules of PLAN.md §4.3/§4.7: fact insert + embedding +
incidence links + entity nodes + relation edges + history, with supersede/invalidate
transitions that never hard-delete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from discord_memory.config import MemoryConfig
from discord_memory.identity.aliases import (
    extract_self_name_aliases,
    is_third_party_name_reference,
    normalize_alias,
    weight_for_source,
)
from discord_memory.ingest.extraction import category_of
from discord_memory.ingest.gates import normalize_text
from discord_memory.lifecycle.strength import reinforced_strength
from discord_memory.lifecycle.tiers import assign_tier
from discord_memory.models.facts import (
    Attribution,
    AttributionType,
    FactHistoryEntry,
    FactRecord,
    SourceRef,
)
from discord_memory.models.identity import AliasSource
from discord_memory.models.operations import ProposedFact
from discord_memory.ports.clock import Clock, IdGen
from discord_memory.ports.llm import Embedder
from discord_memory.ports.store import MemoryStore
from discord_memory.ports.vectors import VectorIndex, VectorItem

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
        tier, expires_after = assign_tier(
            text=proposal.text,
            category=category_of(proposal),
            confidence=proposal.confidence,
            occurrences=1,
            manual=False,
            is_server_fact=subject_id is None,
            lifecycle=self._config.lifecycle,
        )
        third_party = bool(speaker_id and subject_id and speaker_id != subject_id)
        record = FactRecord(
            id=self._id_gen.new_id("fct"),
            guild_id=guild_id,
            subject_id=subject_id,
            text=proposal.text.strip(),
            text_normalized=normalize_text(proposal.text),
            category=category_of(proposal),
            confidence=proposal.confidence,
            tier=tier,
            scope="server" if subject_id is None else "user",
            attribution=Attribution(
                type=AttributionType.THIRD_PARTY if third_party else AttributionType.SELF,
                speaker_id=speaker_id if third_party else None,
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
        await self._store.insert_fact(record)
        if not skip_embedding:
            await self._index_embedding(record)
        await self._write_graph(
            guild_id=guild_id,
            record=record,
            proposal=proposal,
            roster=roster,
            mentioned_ids=mentioned_ids,
        )
        await self._mine_aliases(guild_id=guild_id, record=record)
        await self._store.append_history(
            guild_id,
            record.id,
            FactHistoryEntry(
                at=now,
                kind="created",
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
            tier=tier.value,
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
    ) -> tuple[FactRecord, FactRecord]:
        """Insert refined fact linked to the old one; old stays queryable history."""
        fresh = await self.commit_add(
            proposal=proposal,
            subject_id=subject_id or old_record.subject_id,
            speaker_id=speaker_id,
            guild_id=guild_id,
            roster=roster,
            supersedes_id=old_record.id,
        )
        now = self._clock.now()
        await self._store.transition_fact(
            guild_id,
            old_record.id,
            superseded_by_id=fresh.id,
            updated_at=now,
        )
        await self._store.append_history(
            guild_id,
            old_record.id,
            FactHistoryEntry(
                at=now,
                kind="superseded",
                detail=reason or "refined",
            ),
        )
        return fresh, old_record

    async def commit_invalidate(self, *, old_record: FactRecord, reason: str) -> FactRecord | None:
        now = self._clock.now()
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
        logger.debug("Invalidated %s; detached %d edge evidences", old_record.id, detached)
        if updated is not None:
            await self._store.append_history(
                old_record.guild_id,
                old_record.id,
                FactHistoryEntry(
                    at=now,
                    kind="invalidated",
                    detail=reason or "contradicted",
                ),
            )
        return updated

    async def _index_embedding(self, record: FactRecord) -> None:
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

    async def _write_graph(
        self,
        *,
        guild_id: str,
        record: FactRecord,
        proposal: ProposedFact,
        roster: RosterLike,
        mentioned_ids: tuple[str, ...] = (),
    ) -> None:
        from discord_memory.graph.writes import write_fact_graph

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

    async def _mine_aliases(self, *, guild_id: str, record: FactRecord) -> None:
        """Register subject aliases mined from fact text (PLAN.md §3.2 write-time).

        Skips third-party facts and kinship/possessive references so a speaker's
        statement about someone else never teaches us the subject's name.
        """
        if record.subject_id is None:
            return
        if record.attribution.type is AttributionType.THIRD_PARTY:
            return
        for surface, _weight in extract_self_name_aliases(record.text):
            if is_third_party_name_reference(record.text, surface):
                continue
            await self._store.upsert_alias(
                guild_id,
                normalize_alias(surface),
                record.subject_id,
                AliasSource.REAL_NAME,
                weight_for_source(AliasSource.REAL_NAME),
            )


class RosterLike(Protocol):
    """Structural subset of :class:`~discord_memory.ingest.roster.Roster`."""

    def knows(self, token: str) -> bool: ...

    def user_id_for(self, token: str) -> str | None: ...


__all__ = ["CommitSummary", "FactCommitter", "RosterLike"]
