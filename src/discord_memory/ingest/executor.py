"""Fact committer: turns vetted candidates and reconcile decisions into storage.

Owns the transactional write rules of PLAN.md §4.3/§4.7: fact insert + embedding +
incidence links + entity nodes + relation edges + history, with supersede/invalidate
transitions that never hard-delete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from discord_memory.config import MemoryConfig
from discord_memory.graph.relations import compute_edge_weight, merge_edge, polarity_for_verb
from discord_memory.identity.aliases import alias_slug, normalize_alias
from discord_memory.ingest.extraction import category_of
from discord_memory.ingest.gates import normalize_text
from discord_memory.lifecycle.strength import reinforced_strength
from discord_memory.lifecycle.tiers import assign_tier
from discord_memory.models.facts import (
    Attribution,
    AttributionType,
    FactHistoryEntry,
    FactRecord,
)
from discord_memory.models.graph import EdgeKind, LinkRow, NodeType, RelationEdge
from discord_memory.models.operations import ProposedEntity, ProposedFact, ProposedRelation
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
        )
        await self._store.insert_fact(record)
        await self._index_embedding(record)
        await self._write_graph(guild_id=guild_id, record=record, proposal=proposal, roster=roster)
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
    ) -> None:
        now = self._clock.now()
        links: list[LinkRow] = []
        if record.subject_id is not None:
            links.append(
                _link(
                    guild_id, record.id, NodeType.USER, record.subject_id, EdgeKind.SUBJECT_OF, now
                )
            )
        speaker = record.attribution.speaker_id
        if speaker and speaker != record.subject_id:
            links.append(
                _link(guild_id, record.id, NodeType.USER, speaker, EdgeKind.SPEAKER_OF, now)
            )
        slugs: list[str] = []
        for entity in proposal.entities:
            slug = await self._resolve_entity_slug(guild_id, entity)
            slugs.append(slug)
            links.append(
                _link(guild_id, record.id, NodeType.ENTITY, slug, EdgeKind.ABOUT_ENTITY, now)
            )
        if links:
            await self._store.add_links(tuple(links))
        for slug in dict.fromkeys(slugs):
            await self._store.bump_entity_facts(guild_id, slug)
        for relation in proposal.relations:
            await self._write_relation(
                guild_id=guild_id,
                relation=relation,
                fact=record,
                roster=roster,
            )

    async def _resolve_entity_slug(self, guild_id: str, entity: ProposedEntity) -> str:
        alias_norm = normalize_alias(entity.name)
        existing = await self._store.resolve_entity_alias(guild_id, alias_norm)
        if existing:
            return existing
        slug = alias_slug(entity.name)
        await self._store.upsert_entity(
            guild_id,
            slug,
            entity.name,
            entity.kind,  # type: ignore[arg-type]
            aliases=(alias_norm,),
        )
        return slug

    async def _write_relation(
        self,
        *,
        guild_id: str,
        relation: ProposedRelation,
        fact: FactRecord,
        roster: RosterLike,
    ) -> None:
        endpoints: list[tuple[NodeType, str]] = []
        for token, name in (
            (relation.from_token, relation.from_entity),
            (relation.to_token, relation.to_entity),
        ):
            if token and roster.knows(token) and token != "server":
                user_id = roster.user_id_for(token)
                if user_id:
                    endpoints.append((NodeType.USER, user_id))
            elif name:
                slug = await self._resolve_entity_slug(guild_id, ProposedEntity(name=name))
                endpoints.append((NodeType.ENTITY, slug))
        if len(endpoints) < 2 or endpoints[0] == endpoints[1]:
            return
        src, dst = endpoints[0], endpoints[1]
        verb = relation.verb.strip().lower().replace(" ", "_")
        now = self._clock.now()
        active_edges = [
            e for e in await self._store.edges_between(guild_id, src, dst) if e.verb == verb
        ]
        matching = active_edges[0] if active_edges else None
        if matching is not None:
            incoming = matching.model_copy(update={"evidence_fact_ids": (fact.id,)})
            await self._store.upsert_relation(merge_edge(matching, incoming, now=now))
            return
        weight = compute_edge_weight(
            occurrences=1,
            confidence=fact.confidence,
            last_reinforced_at=now,
            now=now,
        )
        await self._store.upsert_relation(
            RelationEdge(
                guild_id=guild_id,
                src_type=src[0],
                src_id=src[1],
                dst_type=dst[0],
                dst_id=dst[1],
                verb=verb,
                polarity=polarity_for_verb(verb),
                weight=weight,
                confidence=fact.confidence,
                evidence_fact_ids=(fact.id,),
                valid_from=now,
            )
        )


class RosterLike(Protocol):
    """Structural subset of :class:`~discord_memory.ingest.roster.Roster`."""

    def knows(self, token: str) -> bool: ...

    def user_id_for(self, token: str) -> str | None: ...


def _link(
    guild_id: str,
    memory_id: str,
    node_type: NodeType,
    node_id: str,
    kind: EdgeKind,
    now: datetime,
) -> LinkRow:
    return LinkRow(
        guild_id=guild_id,
        memory_id=memory_id,
        node_type=node_type,
        node_id=node_id,
        kind=kind,
        created_at=now,
    )


__all__ = ["CommitSummary", "FactCommitter", "RosterLike"]
