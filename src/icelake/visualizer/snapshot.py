"""Build a graph-explorer snapshot from ``DiscordMemory`` (read-only, no LLM).

Incidence links (``dm_links``) are not drawn — they are an index, not a
relationship. Typed ``dm_relations`` plus identity ``is`` edges are the canvas.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict

from icelake.api.client import DiscordMemory
from icelake.identity.aliases import alias_slug, strongest_alias
from icelake.models.admin import GuildStats, MemoryExport
from icelake.models.facts import FactRecord
from icelake.models.graph import EntityRecord, NodeType
from icelake.models.identity import AliasRecord
from icelake.visualizer.models import (
    GraphSnapshot,
    VizAlias,
    VizCitation,
    VizEdge,
    VizEdgeKind,
    VizFact,
    VizNode,
    VizNodeType,
    VizStats,
)

SERVER_NODE_ID = "server:guild"
_SERVER_REFS = frozenset({"server", "guild", "community", "the server"})
_SEARCH_CHARS = 2000


class CenterError(ValueError):
    """``--center`` did not resolve to a node."""


class CenterAmbiguousError(CenterError):
    """Name matched more than one member — never guess."""

    def __init__(self, identifier: str, candidates: tuple[str, ...]) -> None:
        self.identifier = identifier
        self.candidates = candidates
        listed = ", ".join(candidates) or "(none)"
        super().__init__(f"{identifier!r} is ambiguous: {listed}")


def user_node_id(user_id: str) -> str:
    return f"user:{user_id}"


def entity_node_id(slug: str) -> str:
    return f"entity:{slug}"


def compact_fact(record: FactRecord) -> VizFact:
    return VizFact(
        id=record.id,
        text=record.text,
        category=record.category.value,
        tier=record.tier.value,
        subject_id=record.subject_id,
        confidence=record.confidence,
        occurrences=record.occurrences,
        active=record.is_active,
        entity_slugs=record.entity_slugs,
        related_user_ids=record.related_user_ids,
        citations=tuple(
            VizCitation(
                message_url=cite.message_url,
                author_name=cite.author_name,
                content_snippet=cite.content_snippet,
            )
            for cite in record.citations
        ),
    )


def collect_user_ids(export: MemoryExport) -> set[str]:
    users: set[str] = set()
    for fact in export.facts:
        if fact.subject_id:
            users.add(fact.subject_id)
        users.update(fact.related_user_ids)
    for entity in export.entities:
        if entity.linked_user_id:
            users.add(entity.linked_user_id)
    for edge in export.relations:
        if edge.src_type is NodeType.USER:
            users.add(edge.src_id)
        if edge.dst_type is NodeType.USER:
            users.add(edge.dst_id)
    return users


def _endpoint(node_type: NodeType, node_id: str) -> str:
    if node_type is NodeType.USER:
        return user_node_id(node_id)
    return entity_node_id(node_id)


def _search_blob(label: str, aliases: tuple[VizAlias, ...], texts: list[str]) -> str:
    parts = [label, *(item.alias for item in aliases), *texts]
    blob = " ".join(parts).strip()
    return blob if len(blob) <= _SEARCH_CHARS else blob[: _SEARCH_CHARS - 1] + "…"


def ego_keep(edges: tuple[VizEdge, ...], center_id: str, depth: int) -> set[str]:
    """Undirected BFS over relation + identity edges."""
    adj: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        adj[edge.source].add(edge.target)
        adj[edge.target].add(edge.source)
    keep = {center_id}
    frontier = {center_id}
    for _ in range(max(0, depth)):
        nxt: set[str] = set()
        for node_id in frontier:
            nxt |= adj.get(node_id, set())
        nxt -= keep
        keep |= nxt
        frontier = nxt
        if not frontier:
            break
    return keep


def snapshot_from_export(
    export: MemoryExport,
    stats: GuildStats,
    labels: dict[str, tuple[str, tuple[VizAlias, ...]]],
    *,
    center_id: str | None = None,
    depth: int = 2,
) -> GraphSnapshot:
    """Pure projection: export + identity labels → explorer payload."""
    facts = tuple(compact_fact(record) for record in export.facts)
    facts_by_subject: dict[str | None, list[str]] = defaultdict(list)
    fact_text: dict[str, str] = {}
    for fact in facts:
        facts_by_subject[fact.subject_id].append(fact.id)
        fact_text[fact.id] = fact.text

    nodes = _build_nodes(export, labels, facts_by_subject, fact_text)
    edges = _build_edges(export)
    node_ids = {node.id for node in nodes}
    edges = tuple(edge for edge in edges if edge.source in node_ids and edge.target in node_ids)

    if center_id is not None:
        if center_id not in node_ids:
            raise CenterError(f"center {center_id!r} is not in this guild graph")
        keep = ego_keep(edges, center_id, depth)
        nodes = tuple(node for node in nodes if node.id in keep)
        edges = tuple(edge for edge in edges if edge.source in keep and edge.target in keep)
        cited = {fid for node in nodes for fid in node.fact_ids}
        cited.update(fid for edge in edges for fid in edge.evidence_fact_ids)
        facts = tuple(fact for fact in facts if fact.id in cited)

    return GraphSnapshot(
        guild_id=export.guild_id,
        center=center_id,
        depth=depth if center_id is not None else None,
        stats=_viz_stats(stats),
        nodes=nodes,
        edges=edges,
        facts=facts,
    )


def _viz_stats(stats: GuildStats) -> VizStats:
    return VizStats(
        total_facts=stats.total_facts,
        active_facts=stats.active_facts,
        user_count=stats.user_count,
        entity_count=stats.entity_count,
        relation_count=stats.relation_count,
        pending_messages=stats.pending_messages,
    )


def _build_nodes(
    export: MemoryExport,
    labels: dict[str, tuple[str, tuple[VizAlias, ...]]],
    facts_by_subject: dict[str | None, list[str]],
    fact_text: dict[str, str],
) -> tuple[VizNode, ...]:
    nodes: list[VizNode] = [
        _server_node(export.guild_id, facts_by_subject.get(None, []), fact_text)
    ]
    for user_id in sorted(collect_user_ids(export)):
        label, aliases = labels.get(user_id, (user_id, ()))
        fact_ids = tuple(facts_by_subject.get(user_id, ()))
        texts = [fact_text[fid] for fid in fact_ids if fid in fact_text]
        nodes.append(
            VizNode(
                id=user_node_id(user_id),
                type=VizNodeType.USER,
                label=label,
                user_id=user_id,
                aliases=aliases,
                fact_ids=fact_ids,
                search_text=_search_blob(label, aliases, [*texts, user_id]),
            )
        )
    for entity in export.entities:
        texts = [entity.name, entity.summary, *entity.aliases]
        nodes.append(
            VizNode(
                id=entity_node_id(entity.slug),
                type=VizNodeType.ENTITY,
                label=entity.name or entity.slug,
                entity_slug=entity.slug,
                entity_kind=entity.kind.value,
                linked_user_id=entity.linked_user_id,
                aliases=tuple(VizAlias(alias=alias, source="entity") for alias in entity.aliases),
                search_text=_search_blob(entity.name or entity.slug, (), texts),
            )
        )
    return tuple(nodes)


def _server_node(guild_id: str, fact_ids: list[str], fact_text: dict[str, str]) -> VizNode:
    texts = [fact_text[fid] for fid in fact_ids if fid in fact_text]
    return VizNode(
        id=SERVER_NODE_ID,
        type=VizNodeType.SERVER,
        label="server",
        fact_ids=tuple(fact_ids),
        search_text=_search_blob("server guild community", (), [*texts, guild_id]),
    )


def _build_edges(export: MemoryExport) -> tuple[VizEdge, ...]:
    edges: list[VizEdge] = []
    for index, rel in enumerate(export.relations):
        edges.append(
            VizEdge(
                id=f"rel:{index}",
                source=_endpoint(rel.src_type, rel.src_id),
                target=_endpoint(rel.dst_type, rel.dst_id),
                verb=rel.verb,
                polarity=rel.polarity.value,
                weight=rel.weight,
                occurrences=rel.occurrences,
                confidence=rel.confidence,
                evidence_fact_ids=rel.evidence_fact_ids,
                kind=VizEdgeKind.RELATION,
            )
        )
    for entity in export.entities:
        if not entity.linked_user_id:
            continue
        edges.append(
            VizEdge(
                id=f"is:{entity.slug}",
                source=entity_node_id(entity.slug),
                target=user_node_id(entity.linked_user_id),
                verb="is",
                polarity="neutral",
                weight=1.0,
                occurrences=1,
                confidence=1.0,
                kind=VizEdgeKind.IDENTITY,
            )
        )
    return tuple(edges)


def match_entity(entities: tuple[EntityRecord, ...], needle: str) -> str | None:
    slug = alias_slug(needle)
    lowered = needle.strip().lower()
    for entity in entities:
        names = {entity.slug, entity.name.lower(), *(alias.lower() for alias in entity.aliases)}
        if slug in names or lowered in names:
            return entity_node_id(entity.slug)
    return None


async def resolve_center(
    memory: DiscordMemory,
    guild_id: str,
    center: str,
    entities: tuple[EntityRecord, ...],
) -> str:
    if center.strip().lower() in _SERVER_REFS:
        return SERVER_NODE_ID
    resolution = await memory.identity.resolve(guild_id, center)
    if resolution.ambiguous:
        raise CenterAmbiguousError(
            center,
            tuple(f"{c.user_id} ({c.matched_alias})" for c in resolution.candidates),
        )
    if resolution.resolved is not None:
        return user_node_id(resolution.resolved.user_id)
    entity_id = match_entity(entities, center)
    if entity_id is not None:
        return entity_id
    raise CenterError(f"could not resolve {center!r} to a user, entity, or the server")


async def _load_labels(
    memory: DiscordMemory, guild_id: str, user_ids: set[str]
) -> dict[str, tuple[str, tuple[VizAlias, ...]]]:
    async def one(user_id: str) -> tuple[str, str, tuple[VizAlias, ...]]:
        records: tuple[AliasRecord, ...] = await memory.identity.aliases_of(guild_id, user_id)
        aliases = tuple(
            VizAlias(alias=row.alias_norm, source=row.source.value, weight=row.weight)
            for row in records
        )
        return user_id, strongest_alias(records) or user_id, aliases

    loaded = await asyncio.gather(*(one(uid) for uid in user_ids))
    return {user_id: (label, aliases) for user_id, label, aliases in loaded}


async def build_snapshot(
    memory: DiscordMemory,
    guild_id: str,
    *,
    center: str | None = None,
    depth: int = 2,
) -> GraphSnapshot:
    """Read-only snapshot via ``admin.export_guild`` + identity labels."""
    export = await memory.admin.export_guild(guild_id)
    stats = await memory.stats(guild_id)
    labels = await _load_labels(memory, guild_id, collect_user_ids(export))
    center_id = None
    if center:
        center_id = await resolve_center(memory, guild_id, center, export.entities)
    return snapshot_from_export(export, stats, labels, center_id=center_id, depth=depth)
