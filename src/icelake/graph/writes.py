"""Shared fact-graph writes: incidence rows, entity nodes, relation edges.

Single implementation used by BOTH write paths — pipeline extraction and the
manual ``facts.remember`` API — so programmatic commands participate in the
knowledge graph identically to passive learning (DRY; one concern per module).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol

from icelake.graph.relations import compute_edge_weight, polarity_for_verb
from icelake.identity.aliases import alias_slug, normalize_alias
from icelake.models.facts import FactRecord
from icelake.models.graph import EdgeKind, LinkRow, NodeType, RelationEdge
from icelake.models.operations import ProposedEntity, ProposedRelation
from icelake.ports.store import MemoryStore

logger = logging.getLogger(__name__)


class TokenRoster(Protocol):
    """Token->user resolution surface (satisfied by ingest.roster.Roster)."""

    def knows(self, token: str) -> bool: ...

    def user_id_for(self, token: str) -> str | None: ...


class DirectRoster:
    """Trivial roster over explicit user ids (for manual API writes).

    Tokens are the raw user IDs themselves, so callers can reference endpoints
    directly without minting p0/p1 tokens.
    """

    def __init__(self, *user_ids: str) -> None:
        self._ids = set(user_ids)

    def knows(self, token: str) -> bool:
        return token in self._ids

    def user_id_for(self, token: str) -> str | None:
        return token if token in self._ids else None

    def bind_names(self, text: str) -> str:
        return text

    def display_name(self, user_id: str) -> str | None:
        return None


async def resolve_entity_slug(
    store: MemoryStore,
    guild_id: str,
    name: str,
    kind: str = "concept",
) -> str:
    """Canonicalize a surface name into an existing-or-new entity slug."""
    alias_norm = normalize_alias(name)
    existing = await store.resolve_entity_alias(guild_id, alias_norm)
    if existing:
        return existing
    slug = alias_slug(name)
    await store.upsert_entity(guild_id, slug, name, kind, aliases=(alias_norm,))  # type: ignore[arg-type]
    return slug


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


async def write_fact_graph(
    *,
    store: MemoryStore,
    clock_now: datetime,
    guild_id: str,
    record: FactRecord,
    entities: tuple[ProposedEntity, ...] = (),
    relations: tuple[ProposedRelation, ...] = (),
    mentioned_ids: tuple[str, ...] = (),
    roster: TokenRoster | None = None,
) -> tuple[str, ...]:
    """Materialize layer-2 incidence + layer-3 relations for a committed fact.

    Returns the touched entity slugs. Ownership never changes hands: linking is
    additive around the fact's single anchor.
    """
    now = clock_now
    links: list[LinkRow] = []
    if record.subject_id is not None:
        links.append(
            _link(guild_id, record.id, NodeType.USER, record.subject_id, EdgeKind.SUBJECT_OF, now)
        )
    speaker = record.attribution.speaker_id
    if speaker and speaker != record.subject_id:
        links.append(_link(guild_id, record.id, NodeType.USER, speaker, EdgeKind.SPEAKER_OF, now))

    linked_already = {(link.node_type, link.node_id) for link in links}
    for mention_id in mentioned_ids:
        identity = (NodeType.USER, mention_id)
        if identity in linked_already or mention_id == record.subject_id:
            continue
        kind = EdgeKind.ABOUT_USER if speaker and mention_id == speaker else EdgeKind.MENTIONED_WITH
        links.append(_link(guild_id, record.id, NodeType.USER, mention_id, kind, now))

    slugs: list[str] = []
    for entity in entities:
        slug = await resolve_entity_slug(store, guild_id, entity.name, entity.kind)
        slugs.append(slug)
        links.append(_link(guild_id, record.id, NodeType.ENTITY, slug, EdgeKind.ABOUT_ENTITY, now))

    if links:
        await store.add_links(tuple(links))
    for slug in dict.fromkeys(slugs):
        await store.bump_entity_facts(guild_id, slug)

    for relation in relations:
        await _write_relation(
            store=store, guild_id=guild_id, relation=relation, fact=record, roster=roster, now=now
        )
    return tuple(dict.fromkeys(slugs))


async def _write_relation(
    *,
    store: MemoryStore,
    guild_id: str,
    relation: ProposedRelation,
    fact: FactRecord,
    roster: TokenRoster | None,
    now: datetime,
) -> None:
    endpoints: list[tuple[NodeType, str]] = []
    for token, name in (
        (relation.from_token, relation.from_entity),
        (relation.to_token, relation.to_entity),
    ):
        resolved = False
        if token and token != "server":
            user_id = roster.user_id_for(token) if roster else None
            if user_id:
                endpoints.append((NodeType.USER, user_id))
                resolved = True
        if not resolved and name:
            slug = await resolve_entity_slug(store, guild_id, name)
            endpoints.append((NodeType.ENTITY, slug))
    if len(endpoints) < 2 or endpoints[0] == endpoints[1]:
        return
    src, dst = endpoints[0], endpoints[1]
    verb = relation.verb.strip().lower().replace(" ", "_")
    active = [e for e in await store.edges_between(guild_id, src, dst) if e.verb == verb]
    matching = active[0] if active else None
    if matching is not None:
        incoming = matching.model_copy(update={"evidence_fact_ids": (fact.id,)})
        evidence = dict.fromkeys(matching.evidence_fact_ids + incoming.evidence_fact_ids)
        occurrences = matching.occurrences + 1
        recomputed = compute_edge_weight(
            occurrences=occurrences,
            confidence=max(matching.confidence, fact.confidence),
            last_reinforced_at=now,
            now=now,
        )
        await store.upsert_relation(
            matching.model_copy(
                update={
                    "occurrences": occurrences,
                    "weight": max(matching.weight, recomputed),
                    "confidence": max(matching.confidence, fact.confidence),
                    "evidence_fact_ids": tuple(evidence)[-8:],
                }
            )
        )
        return
    await store.upsert_relation(
        RelationEdge(
            guild_id=guild_id,
            src_type=src[0],
            src_id=src[1],
            dst_type=dst[0],
            dst_id=dst[1],
            verb=verb,
            polarity=polarity_for_verb(verb),
            weight=compute_edge_weight(
                occurrences=1,
                confidence=fact.confidence,
                last_reinforced_at=now,
                now=now,
            ),
            confidence=fact.confidence,
            evidence_fact_ids=(fact.id,),
            valid_from=now,
        )
    )


__all__ = ["DirectRoster", "TokenRoster", "resolve_entity_slug", "write_fact_graph"]
