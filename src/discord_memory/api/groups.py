"""Identity, graph and admin API groups (API.md Parts 8-10)."""

from __future__ import annotations

from discord_memory.identity.aliases import normalize_alias, weight_for_source
from discord_memory.identity.resolver import IdentityResolver
from discord_memory.models.admin import MemoryExport, PurgeReport
from discord_memory.models.common import Page
from discord_memory.models.graph import (
    NeighborInfo,
    NodeType,
    RelationEdge,
    StanceSummary,
)
from discord_memory.models.identity import AliasRecord, AliasSource, Resolution
from discord_memory.ports.store import MemoryStore, NodeRef


class IdentityApi:
    """Name ↔ hardened ID resolution — the ``memory.identity.*`` namespace."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._resolver = IdentityResolver(store)

    async def resolve(self, guild_id: str, identifier: str) -> Resolution:
        return await self._resolver.resolve(guild_id, identifier)

    async def register_alias(
        self,
        guild_id: str,
        user_id: str,
        alias: str,
        source: AliasSource = AliasSource.DISPLAY_NAME,
    ) -> None:
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
            from discord_memory.identity.aliases import weight_for_source

            await self._store.upsert_alias(
                guild_id,
                normalized,
                user_id,
                AliasSource.DISPLAY_NAME,
                weight_for_source(AliasSource.DISPLAY_NAME, surface=normalized),
            )

    async def aliases_of(self, guild_id: str, user_id: str) -> tuple[AliasRecord, ...]:
        return await self._store.aliases_for_user(guild_id, user_id)


class GraphApi:
    """Relations, stances and discovery — the ``memory.graph.*`` namespace."""

    def __init__(self, *, store: MemoryStore) -> None:
        self._store = store

    async def between(
        self,
        guild_id: str,
        src_user_id: str,
        dst_user_id: str,
    ) -> tuple[RelationEdge, ...]:
        return await self._store.edges_between(
            guild_id,
            (NodeType.USER, src_user_id),
            (NodeType.USER, dst_user_id),
        )

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
        from discord_memory.identity.aliases import alias_slug

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
            positive=tuple(e for e in edges if e.polarity.value == "positive"),
            negative=tuple(e for e in edges if e.polarity.value == "negative"),
            other=tuple(e for e in edges if e.polarity.value == "neutral"),
            total_evidence=sum(e.occurrences for e in edges),
        )

    async def similar_users(
        self,
        guild_id: str,
        user_id: str,
        *,
        limit: int = 10,
    ) -> tuple[tuple[str, float], ...]:
        """Members sharing entity traits with this user, Jaccard-ranked (capped)."""
        from discord_memory.graph.traversal import jaccard_similarity

        seed_edges = await self._store.incident_edges(
            guild_id,
            (NodeType.USER, user_id),
            limit=200,
        )
        seed_entities = {edge.dst_id for edge in seed_edges if edge.dst_type is NodeType.ENTITY}
        if not seed_entities:
            return ()

        candidates: set[str] = set()
        for slug in list(seed_entities)[:50]:
            for edge in await self._store.entity_stance_edges(guild_id, slug, limit=100):
                if edge.src_type is NodeType.USER and edge.src_id != user_id:
                    candidates.add(edge.src_id)

        scored: list[tuple[str, float]] = []
        for candidate in list(candidates)[:100]:
            candidate_entities = {
                edge.dst_id
                for edge in await self._store.incident_edges(
                    guild_id,
                    (NodeType.USER, candidate),
                    limit=200,
                )
                if edge.dst_type is NodeType.ENTITY
            }
            score = jaccard_similarity(
                frozenset(seed_entities),
                frozenset(candidate_entities),
            )
            if score > 0:
                scored.append((candidate, round(score, 4)))

        scored.sort(key=lambda pair: -pair[1])
        return tuple(scored[:limit])

    async def neighbors(
        self,
        guild_id: str,
        user_id: str,
        *,
        depth: int = 2,
        limit_per_hop: int = 24,
    ) -> tuple[NeighborInfo, ...]:
        from discord_memory.graph.traversal import hop_neighbors, node_key

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

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

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
        return await self._store.purge_user_data(guild_id, user_id, dry_run=dry_run)

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
