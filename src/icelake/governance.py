"""Distributed governance fencing for pre-purge data.

Standalone Mongo offers no cross-document transactions, so check-then-write
cannot fence in-flight writers. Instead every durable fact and queued message
stamps a snapshot of each participant's erasure generation captured at
enqueue/claim time, and every read compares that per-participant snapshot
against the CURRENT durable generation: any participant whose generation has
since advanced fences the row out, regardless of which process wrote it.
"""

from __future__ import annotations

from typing import Protocol

from icelake.models.facts import FactRecord
from icelake.models.graph import EntityRecord, NodeType, RelationEdge


class GovernanceReader(Protocol):
    """Batch generation reads used by the fence."""

    async def governance_generations(
        self,
        guild_id: str,
        user_ids: tuple[str, ...],
    ) -> dict[str, int]: ...


def fact_participants(record: FactRecord) -> tuple[str, ...]:
    """Every user whose consent state governs this fact's visibility."""
    users: list[str] = []
    for user_id in (
        record.subject_id,
        record.attribution.speaker_id,
        record.attribution.actor_id,
        *record.related_user_ids,
        *(citation.author_id for citation in record.citations),
    ):
        if user_id and user_id not in users:
            users.append(user_id)
    return tuple(users)


async def generation_map(
    store: GovernanceReader,
    guild_id: str,
    records: tuple[FactRecord, ...],
) -> dict[str, int]:
    """Current erasure generation for every user associated with ``records``."""
    users: set[str] = set()
    for record in records:
        users.update(fact_participants(record))
    if not users:
        return {}
    return await store.governance_generations(guild_id, tuple(users))


def fact_fenced(record: FactRecord, generations: dict[str, int]) -> bool:
    """True when any participant has purged after the fact's stamped snapshot."""
    return any(
        generations.get(user_id, 0) > record.governance.get(user_id, 0)
        for user_id in fact_participants(record)
    )


async def participants_current(
    store: GovernanceReader,
    guild_id: str,
    user_ids: tuple[str, ...],
) -> dict[str, int]:
    """Current generations for a bounded participant set (batched, deduped)."""
    unique = tuple(dict.fromkeys(user_ids))
    if not unique:
        return {}
    return await store.governance_generations(guild_id, unique)


class ExportGovernanceReader(GovernanceReader, Protocol):
    async def get_opt_out(self, guild_id: str, user_id: str) -> bool: ...


async def filter_export(
    store: ExportGovernanceReader,
    guild_id: str,
    facts: tuple[FactRecord, ...],
    entities: tuple[EntityRecord, ...],
    relations: tuple[RelationEdge, ...],
) -> tuple[tuple[FactRecord, ...], tuple[EntityRecord, ...], tuple[RelationEdge, ...]]:
    """Exclude erased/blocked evidence so a portable export cannot resurrect it."""
    generations = await generation_map(store, guild_id, facts)
    participants = {uid for fact in facts for uid in fact_participants(fact)}
    participants.update(entity.linked_user_id for entity in entities if entity.linked_user_id)
    participants.update(
        uid
        for edge in relations
        for kind, uid in ((edge.src_type, edge.src_id), (edge.dst_type, edge.dst_id))
        if kind is NodeType.USER
    )
    blocked = {uid for uid in participants if await store.get_opt_out(guild_id, uid)}
    visible = tuple(
        fact
        for fact in facts
        if not fact_fenced(fact, generations) and not blocked.intersection(fact_participants(fact))
    )
    visible_ids = {fact.id for fact in visible}
    edges = tuple(
        edge
        for edge in relations
        if edge.evidence_fact_ids
        and visible_ids.issuperset(edge.evidence_fact_ids)
        and not (edge.src_type is NodeType.USER and edge.src_id in blocked)
        and not (edge.dst_type is NodeType.USER and edge.dst_id in blocked)
    )
    # Entity digests have no source revision contract. Never export their text.
    safe_entities = tuple(
        entity.model_copy(update={"summary": ""})
        for entity in entities
        if entity.linked_user_id not in blocked
    )
    return visible, safe_entities, edges
