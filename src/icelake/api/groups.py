"""Identity, graph and admin API groups (API.md Parts 8-10)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from icelake.graph.collapse import collapse_edges, twin_refs
from icelake.identity.aliases import normalize_alias, weight_for_source
from icelake.identity.resolver import IdentityResolver
from icelake.models.admin import MemoryExport, PurgeReport
from icelake.models.common import Page
from icelake.models.graph import (
    EntityKind,
    EntityUsers,
    NeighborInfo,
    NodeType,
    Polarity,
    RelationEdge,
    SimilarUser,
    StanceSummary,
)
from icelake.models.identity import AliasRecord, AliasSource, Resolution
from icelake.ports.store import MemoryStore, NodeRef


async def _noop_gate() -> None:
    """Default gate: no-op (used when the facade wires no lifecycle)."""


class IdentityApi:
    """Name ↔ hardened ID resolution — the ``memory.identity.*`` namespace."""

    def __init__(
        self,
        store: MemoryStore,
        startup_gate: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._store = store
        self._gate = startup_gate or _noop_gate
        self._resolver = IdentityResolver(store)

    async def resolve(self, guild_id: str, identifier: str) -> Resolution:
        await self._gate()
        return await self._resolver.resolve(guild_id, identifier)

    async def register_alias(
        self,
        guild_id: str,
        user_id: str,
        alias: str,
        source: AliasSource = AliasSource.DISPLAY_NAME,
    ) -> None:
        await self._gate()
        alias_norm = normalize_alias(alias)
        if not alias_norm:
            return
        weight = weight_for_source(source, surface=alias)
        await self._store.upsert_alias(guild_id, alias_norm, user_id, source, weight)
        await self._reconcile_person_entity(guild_id, alias_norm, user_id)

    async def _reconcile_person_entity(self, guild_id: str, alias_norm: str, user_id: str) -> None:
        """Bridge (or unbridge) a PERSON entity twin as the name's resolution changes.

        Extraction can mint a person-entity from a third-party mention before
        the member ever speaks. When the alias uniquely resolves to a member,
        the entity IS that member — link it so graph reads collapse the twin
        onto the user node. When the name is ambiguous (a second member
        registers it), any existing bridge is cleared: an ambiguous name owns
        no twin (grounded-or-silent — never guess which member it is).
        """
        slug = await self._store.resolve_entity_alias(guild_id, alias_norm)
        if slug is None:
            return
        entity = await self._store.get_entity(guild_id, slug)
        if entity is None or entity.kind is not EntityKind.PERSON:
            return
        resolution = await self._resolver.resolve(guild_id, alias_norm)
        if resolution.ambiguous or resolution.resolved is None:
            if entity.linked_user_id is not None:
                await self._store.link_entity_to_user(guild_id, slug, None)
            return
        if entity.linked_user_id is None and resolution.resolved.user_id == user_id:
            await self._store.link_entity_to_user(guild_id, slug, user_id)

    async def handle_member_rename(
        self,
        guild_id: str,
        user_id: str,
        new_display_name: str,
    ) -> None:
        """Re-index a rename: new name binds strongly; old aliases stop accruing."""
        normalized = normalize_alias(new_display_name)
        if not normalized:
            return
        existing = await self._store.resolve_alias_candidates(guild_id, normalized)
        if not any(record.user_id == user_id for record in existing):
            from icelake.identity.aliases import weight_for_source

            await self._store.upsert_alias(
                guild_id,
                normalized,
                user_id,
                AliasSource.DISPLAY_NAME,
                weight_for_source(AliasSource.DISPLAY_NAME, surface=normalized),
            )
        await self._reconcile_person_entity(guild_id, normalized, user_id)

    async def aliases_of(self, guild_id: str, user_id: str) -> tuple[AliasRecord, ...]:
        return await self._store.aliases_for_user(guild_id, user_id)

    async def display_name(self, guild_id: str, user_id: str) -> str | None:
        """Strongest known alias for operator-facing labels (None if unknown)."""
        from icelake.identity.aliases import strongest_alias

        return strongest_alias(await self._store.aliases_for_user(guild_id, user_id))


class GraphApi:
    """Relations, stances and discovery — the ``memory.graph.*`` namespace."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        startup_gate: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._store = store
        self._gate = startup_gate or _noop_gate

    async def _collapse(
        self, guild_id: str, edges: tuple[RelationEdge, ...]
    ) -> tuple[RelationEdge, ...]:
        """Rewrite entity endpoints that are bridged members into user nodes."""
        return collapse_edges(edges, await self._store.entities_linked_to_users(guild_id))

    async def _incident(
        self, guild_id: str, user_id: str, *, limit: int
    ) -> tuple[RelationEdge, ...]:
        """All edges touching a member: their user node plus every entity twin.

        This is the single read seam that makes person queries complete — a
        member mentioned by name before they spoke lives as an entity twin,
        and those edges count too.
        """
        linked = await self._store.entities_linked_to_users(guild_id)
        refs = twin_refs(user_id, linked)
        if len(refs) == 1:
            raw = await self._store.incident_edges(guild_id, refs[0], limit=limit)
        else:
            raw = await self._store.incident_edges_many(guild_id, refs, limit_per_node=limit)
        return collapse_edges(raw, linked)

    async def between(
        self,
        guild_id: str,
        src_user_id: str,
        dst_user_id: str,
    ) -> tuple[RelationEdge, ...]:
        """Active edges between two users in BOTH directions, weight-ranked.

        Relationships are asymmetric in storage (a→b and b→a are distinct
        edges) but symmetric in user expectation: "between alice and bob"
        means everything connecting them. Entity twins collapse to their
        linked member before the pair filter, so a stored user→entity edge
        counts when the entity is the other person.
        """
        left, right = await asyncio.gather(
            self._incident(guild_id, src_user_id, limit=200),
            self._incident(guild_id, dst_user_id, limit=200),
        )
        pair = {src_user_id, dst_user_id}
        seen: set[tuple[str, str, str]] = set()
        edges: list[RelationEdge] = []
        for e in (*left, *right):
            if (
                e.src_type is NodeType.USER
                and e.dst_type is NodeType.USER
                and {e.src_id, e.dst_id} == pair
            ):
                key = (e.src_id, e.dst_id, e.verb)
                if key not in seen:
                    seen.add(key)
                    edges.append(e)
        return tuple(sorted(edges, key=lambda e: -e.weight))

    async def relations_of(
        self,
        guild_id: str,
        user_id: str,
        *,
        limit: int = 50,
    ) -> tuple[RelationEdge, ...]:
        return await self._incident(guild_id, user_id, limit=limit)

    async def entity_stances(
        self,
        guild_id: str,
        entity_name_or_slug: str,
        *,
        limit: int = 100,
    ) -> StanceSummary:
        from icelake.identity.aliases import alias_slug

        slug = await self._store.resolve_entity_alias(
            guild_id,
            normalize_alias(entity_name_or_slug),
        )
        if slug is None:
            slug = alias_slug(entity_name_or_slug)
        edges = await self._store.entity_stance_edges(guild_id, slug, limit=limit)
        edges = await self._collapse(guild_id, edges)
        entity = await self._store.get_entity(guild_id, slug)
        return StanceSummary(
            entity_slug=slug,
            entity_name=entity.name if entity else entity_name_or_slug,
            positive=tuple(e for e in edges if e.polarity is Polarity.POSITIVE),
            negative=tuple(e for e in edges if e.polarity is Polarity.NEGATIVE),
            other=tuple(e for e in edges if e.polarity is Polarity.NEUTRAL),
            total_evidence=sum(e.occurrences for e in edges),
        )

    async def shared(
        self,
        guild_id: str,
        left_user_id: str,
        right_user_id: str,
        *,
        limit: int = 10,
    ) -> tuple[RelationEdge, ...]:
        """Left member's entity edges that the right member also touches.

        Identity-collapsed and weight-ranked. Prefer ``shared_attributions``
        when the consumer needs both stances (agreement vs disagreement).
        """
        left, _right = await self.shared_attributions(
            guild_id, left_user_id, right_user_id, limit=limit
        )
        return left

    async def shared_attributions(
        self,
        guild_id: str,
        left_user_id: str,
        right_user_id: str,
        *,
        limit: int = 10,
    ) -> tuple[tuple[RelationEdge, ...], tuple[RelationEdge, ...]]:
        """Both members' edges over the entities they share (pair form of
        ``shared_n``). Polarity is preserved so callers can show agreement
        vs disagreement."""
        left, right = await self.shared_n(guild_id, (left_user_id, right_user_id), limit=limit)
        return left, right

    async def shared_n(
        self,
        guild_id: str,
        user_ids: tuple[str, ...],
        *,
        limit: int = 10,
    ) -> tuple[tuple[RelationEdge, ...], ...]:
        """Entity edges over the intersection of ALL members' hubs (N-way).

        A hub survives only when every member touches it — this is the true
        intersection, not a union of pairwise overlaps. One incident fetch per
        member, identity-collapsed so a member named as an entity never counts
        as a shared hobby. Output aligns with ``user_ids`` order.
        """
        ids = tuple(dict.fromkeys(user_ids))
        if len(ids) < 2:
            return tuple(() for _ in ids)
        per_user = await asyncio.gather(
            *(self._incident(guild_id, user_id, limit=200) for user_id in ids)
        )
        entity_sets = [
            {e.dst_id for e in edges if e.dst_type is NodeType.ENTITY} for edges in per_user
        ]
        common = set.intersection(*entity_sets)
        if not common:
            return tuple(() for _ in ids)
        ranked = tuple(dict.fromkeys(e.dst_id for e in per_user[0] if e.dst_id in common))[:limit]
        keep = set(ranked)
        return tuple(
            tuple(e for e in edges if e.dst_type is NodeType.ENTITY and e.dst_id in keep)
            for edges in per_user
        )

    async def users_for_entity(
        self,
        guild_id: str,
        entity_name_or_slug: str,
        *,
        limit: int = 100,
    ) -> EntityUsers:
        """Every member connected to a thing — the "who likes golf?" read.

        Stance buckets come from typed relation edges (``likes`` → positive,
        ``dislikes`` → negative, neutral verbs like ``plays`` → other).
        ``mentioned`` covers members whose facts touch the entity through
        incidence links without a typed stance. Identity-collapsed: a member
        bridged to an entity twin appears as their user ID, never both.
        """
        from icelake.identity.aliases import alias_slug

        slug = await self._store.resolve_entity_alias(
            guild_id,
            normalize_alias(entity_name_or_slug),
        )
        if slug is None:
            slug = alias_slug(entity_name_or_slug)
        edges, linked_rows, entity = await asyncio.gather(
            self._store.entity_stance_edges(guild_id, slug, limit=limit),
            self._store.links_for_node(
                guild_id, NodeType.ENTITY, slug, active_only=True, limit=limit
            ),
            self._store.get_entity(guild_id, slug),
        )
        edges = await self._collapse(guild_id, edges)

        def bucket(polarity: Polarity) -> tuple[str, ...]:
            return tuple(
                dict.fromkeys(
                    edge.src_id
                    for edge in edges
                    if edge.polarity is polarity and edge.src_type is NodeType.USER
                )
            )

        mentioned = tuple(
            dict.fromkeys(
                user_id
                for _row, record in linked_rows
                for user_id in (record.subject_id, record.attribution.speaker_id)
                if user_id
            )
        )
        return EntityUsers(
            entity_slug=slug,
            entity_name=entity.name if entity else entity_name_or_slug,
            positive=bucket(Polarity.POSITIVE),
            negative=bucket(Polarity.NEGATIVE),
            other=bucket(Polarity.NEUTRAL),
            mentioned=mentioned,
            total_evidence=sum(edge.occurrences for edge in edges),
        )

    async def similar_users(
        self,
        guild_id: str,
        user_id: str,
        *,
        limit: int = 10,
    ) -> tuple[SimilarUser, ...]:
        """Members sharing entity traits with this user, Jaccard-ranked (capped).

        Three round-trips total regardless of graph size: seed adjacency,
        reverse lookup of who touches those entities, candidate adjacency.
        """
        from icelake.graph.traversal import jaccard_similarity

        seed_edges = await self._incident(guild_id, user_id, limit=200)
        seed_entities = {edge.dst_id for edge in seed_edges if edge.dst_type is NodeType.ENTITY}
        if not seed_entities:
            return ()

        inbound = await self._store.edges_to_nodes(
            guild_id,
            tuple((NodeType.ENTITY, slug) for slug in list(seed_entities)[:50]),
        )
        inbound = await self._collapse(guild_id, inbound)
        candidates = {
            edge.src_id
            for edge in inbound
            if edge.src_type is NodeType.USER and edge.src_id != user_id
        }
        if not candidates:
            return ()

        candidate_edges = await self._store.incident_edges_many(
            guild_id,
            tuple((NodeType.USER, c) for c in list(candidates)[:100]),
            limit_per_node=200,
        )
        candidate_edges = await self._collapse(guild_id, candidate_edges)
        entities_by_user: dict[str, set[str]] = {}
        for edge in candidate_edges:
            if edge.src_type is NodeType.USER and edge.dst_type is NodeType.ENTITY:
                entities_by_user.setdefault(edge.src_id, set()).add(edge.dst_id)

        scored = [
            SimilarUser(user_id=candidate, score=round(score, 4))
            for candidate, entity_set in entities_by_user.items()
            if (score := jaccard_similarity(frozenset(seed_entities), frozenset(entity_set))) > 0
        ]
        scored.sort(key=lambda hit: -hit.score)
        return tuple(scored[:limit])

    async def neighbors(
        self,
        guild_id: str,
        user_id: str,
        *,
        depth: int = 2,
        limit_per_hop: int = 24,
    ) -> tuple[NeighborInfo, ...]:
        from icelake.graph.traversal import hop_neighbors, node_key

        seed: NodeRef = (NodeType.USER, user_id)
        adjacency: dict[str, list[RelationEdge]] = {}
        frontier = [seed]
        visited = {seed}
        for _ in range(max(1, depth)):
            next_frontier: list[NodeRef] = []
            for node in frontier:
                if node[0] is NodeType.USER:
                    edges = list(await self._incident(guild_id, node[1], limit=limit_per_hop))
                else:
                    raw = await self._store.incident_edges(guild_id, node, limit=limit_per_hop)
                    edges = list(await self._collapse(guild_id, raw))
                key = node_key(node[0].value, node[1])
                adjacency[key] = sorted(edges, key=lambda e: -e.weight)
                for edge in edges[:limit_per_hop]:
                    child = (edge.dst_type, edge.dst_id)
                    if child not in visited:
                        visited.add(child)
                        next_frontier.append(child)
            frontier = next_frontier
        del seed
        return hop_neighbors(
            node_key(NodeType.USER.value, user_id),
            adjacency,
            depth=max(1, depth),
            fan_out_per_node=limit_per_hop,
        )


class AdminApi:
    """Consent, purge, export — the ``memory.admin.*`` namespace."""

    def __init__(
        self,
        store: MemoryStore,
        startup_gate: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._store = store
        self._gate = startup_gate or _noop_gate

    async def set_opt_out(self, guild_id: str, user_id: str, opted_out: bool) -> None:
        await self._store.set_opt_out(guild_id, user_id, opted_out)

    async def get_opt_out(self, guild_id: str, user_id: str) -> bool:
        return await self._store.get_opt_out(guild_id, user_id)

    async def purge_user(
        self,
        guild_id: str,
        user_id: str,
        *,
        dry_run: bool = True,
    ) -> PurgeReport:

        await self._gate()
        return await self._store.purge_user_data(guild_id, user_id, dry_run=dry_run)

    async def import_guild(self, export: MemoryExport) -> int:
        """Restore a previously exported guild. Returns fact count inserted."""
        result = await self._store.import_guild(
            export.facts,
            export.entities,
            export.relations,
        )
        return result

    async def export_guild(self, guild_id: str) -> MemoryExport:
        from datetime import datetime

        facts, entities, relations = await self._store.export_guild(guild_id)
        return MemoryExport(
            guild_id=guild_id,
            exported_at=datetime.now().astimezone(),
            facts=facts,
            entities=entities,
            relations=relations,
        )


__all__ = ["AdminApi", "GraphApi", "IdentityApi", "Page"]
