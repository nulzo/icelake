"""Identity, graph and admin API groups (API.md Parts 8-10)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from icelake.identity.aliases import normalize_alias, weight_for_source
from icelake.identity.resolver import IdentityResolver
from icelake.models.admin import MemoryExport, PurgeReport
from icelake.models.common import Page
from icelake.models.graph import (
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

    async def between(
        self,
        guild_id: str,
        src_user_id: str,
        dst_user_id: str,
    ) -> tuple[RelationEdge, ...]:
        """Active edges between two users in BOTH directions, weight-ranked.

        Relationships are asymmetric in storage (a→b and b→a are distinct
        edges) but symmetric in user expectation: "between alice and bob"
        means everything connecting them.
        """
        forward, backward = await asyncio.gather(
            self._store.edges_between(
                guild_id,
                (NodeType.USER, src_user_id),
                (NodeType.USER, dst_user_id),
            ),
            self._store.edges_between(
                guild_id,
                (NodeType.USER, dst_user_id),
                (NodeType.USER, src_user_id),
            ),
        )
        return tuple(sorted((*forward, *backward), key=lambda e: -e.weight))

    async def relations_of(
        self,
        guild_id: str,
        user_id: str,
        *,
        limit: int = 50,
    ) -> tuple[RelationEdge, ...]:
        return await self._store.incident_edges(
            guild_id,
            (NodeType.USER, user_id),
            limit=limit,
        )

    async def entity_stances(
        self,
        guild_id: str,
        entity_name_or_slug: str,
    ) -> StanceSummary:
        from icelake.identity.aliases import alias_slug

        slug = await self._store.resolve_entity_alias(
            guild_id,
            normalize_alias(entity_name_or_slug),
        )
        if slug is None:
            slug = alias_slug(entity_name_or_slug)
        edges = await self._store.entity_stance_edges(guild_id, slug, limit=100)
        entity = await self._store.get_entity(guild_id, slug)
        return StanceSummary(
            entity_slug=slug,
            entity_name=entity.name if entity else entity_name_or_slug,
            positive=tuple(e for e in edges if e.polarity is Polarity.POSITIVE),
            negative=tuple(e for e in edges if e.polarity is Polarity.NEGATIVE),
            other=tuple(e for e in edges if e.polarity is Polarity.NEUTRAL),
            total_evidence=sum(e.occurrences for e in edges),
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

        seed_edges = await self._store.incident_edges(
            guild_id,
            (NodeType.USER, user_id),
            limit=200,
        )
        seed_entities = {edge.dst_id for edge in seed_edges if edge.dst_type is NodeType.ENTITY}
        if not seed_entities:
            return ()

        inbound = await self._store.edges_to_nodes(
            guild_id,
            tuple((NodeType.ENTITY, slug) for slug in list(seed_entities)[:50]),
        )
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
                edges = list(await self._store.incident_edges(guild_id, node, limit=limit_per_hop))
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
