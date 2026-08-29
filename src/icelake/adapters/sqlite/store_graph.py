"""SQLite MemoryStore mixin: aliases, links, relations, entities (graph layers)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from icelake.adapters.sqlite.connection import SqliteConnection, dumps, iso
from icelake.adapters.sqlite.store_facts import parse_moment
from icelake.models.facts import FactRecord
from icelake.models.graph import (
    EdgeKind,
    EntityKind,
    EntityRecord,
    LinkRow,
    NodeType,
    Polarity,
    RelationEdge,
)
from icelake.models.identity import AliasRecord, AliasSource
from icelake.ports.store import NodeRef


def _alias_from_row(row: sqlite3.Row) -> AliasRecord:
    return AliasRecord(
        guild_id=row["guild_id"],
        alias_norm=row["alias_norm"],
        user_id=row["user_id"],
        source=AliasSource(row["source"]),
        weight=float(row["weight"]),
        updated_at=parse_moment(row["updated_at"]),
    )


class IdentityGraphMixin:
    """Identity + graph persistence over ``self._db``."""

    _db: SqliteConnection

    # -- aliases -----------------------------------------------------------------

    async def upsert_alias(
        self,
        guild_id: str,
        alias_norm: str,
        user_id: str,
        source: AliasSource,
        weight: float,
    ) -> None:
        await self._db.execute(
            """INSERT INTO dm_aliases (guild_id, alias_norm, user_id, source, weight,
                                       updated_at)
               VALUES (?, ?, ?, ?, ?, datetime('now'))
               ON CONFLICT(guild_id, alias_norm, user_id) DO UPDATE SET
                 weight=MAX(weight, excluded.weight),
                 source=excluded.source""",
            (guild_id, alias_norm, user_id, source.value, weight),
        )

    async def resolve_alias_candidates(
        self,
        guild_id: str,
        alias_norm: str,
        limit: int = 8,
    ) -> tuple[AliasRecord, ...]:
        rows = await self._db.query(
            """SELECT * FROM dm_aliases WHERE guild_id=? AND alias_norm=?
               ORDER BY weight DESC LIMIT ?""",
            (guild_id, alias_norm, limit),
        )
        return tuple(_alias_from_row(r) for r in rows)

    async def prefix_alias_candidates(
        self,
        guild_id: str,
        prefix_norm: str,
        limit: int = 8,
    ) -> tuple[AliasRecord, ...]:
        rows = await self._db.query(
            """SELECT * FROM dm_aliases
               WHERE guild_id=? AND alias_norm LIKE ? ESCAPE '\\'
               ORDER BY weight DESC LIMIT ?""",
            (guild_id, prefix_norm.replace("\\", "\\\\").replace("%", r"\%") + "%", limit),
        )
        return tuple(_alias_from_row(r) for r in rows)

    async def aliases_for_user(self, guild_id: str, user_id: str) -> tuple[AliasRecord, ...]:
        rows = await self._db.query(
            "SELECT * FROM dm_aliases WHERE guild_id=? AND user_id=?",
            (guild_id, user_id),
        )
        return tuple(_alias_from_row(r) for r in rows)

    async def delete_aliases_for_user(self, guild_id: str, user_id: str) -> int:
        existing = await self.aliases_for_user(guild_id, user_id)
        await self._db.execute(
            "DELETE FROM dm_aliases WHERE guild_id=? AND user_id=?",
            (guild_id, user_id),
        )
        return len(existing)

    # -- links ---------------------------------------------------------------------

    async def add_links(self, rows: tuple[LinkRow, ...]) -> None:
        for row in rows:
            await self._db.execute(
                """INSERT OR IGNORE INTO dm_links
                   (memory_id, guild_id, node_type, node_id, kind, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    row.memory_id,
                    row.guild_id,
                    row.node_type.value,
                    row.node_id,
                    row.kind.value,
                    iso(row.created_at),
                ),
            )

    async def links_for_node(
        self,
        guild_id: str,
        node_type: NodeType,
        node_id: str,
        *,
        kinds: tuple[EdgeKind, ...] | None = None,
        active_only: bool = True,
        limit: int = 100,
        as_of: datetime | None = None,
    ) -> tuple[tuple[LinkRow, FactRecord], ...]:
        from icelake.adapters.sqlite.store_facts import record_from_row

        conditions = ["l.guild_id=?", "l.node_type=?", "l.node_id=?"]
        params: list[object] = [guild_id, node_type.value, node_id]
        if active_only:
            if as_of is not None:
                conditions.append(
                    "(f.valid_from IS NULL OR f.valid_from <= ?) "
                    "AND (f.valid_until IS NULL OR f.valid_until > ?)"
                )
                params.extend([iso(as_of), iso(as_of)])
            else:
                conditions.append("f.valid_until IS NULL AND f.superseded_by_id IS NULL")
        params.append(limit)
        rows = await self._db.query(
            f"""SELECT l.*, f.* AS fact_.*
                FROM dm_links l JOIN dm_facts f ON f.id=l.memory_id AND f.guild_id=l.guild_id
                WHERE {" AND ".join(conditions)}
                LIMIT ?""".replace("l.*, f.* AS fact_.*", "l.*, f.id AS f_id, f.*"),
            tuple(params),
        )
        results: list[tuple[LinkRow, FactRecord]] = []
        for row in rows:
            kind = EdgeKind(row["kind"])
            if kinds is not None and kind not in kinds:
                continue
            link = LinkRow(
                guild_id=row["guild_id"],
                memory_id=row["memory_id"],
                node_type=NodeType(row["node_type"]),
                node_id=row["node_id"],
                kind=kind,
                created_at=parse_moment(row["created_at"]),
            )
            record = record_from_row(row)
            results.append((link, record))
        return tuple(results[:limit])

    async def nodes_for_fact(self, guild_id: str, memory_id: str) -> tuple[LinkRow, ...]:
        rows = await self._db.query(
            "SELECT * FROM dm_links WHERE guild_id=? AND memory_id=?",
            (guild_id, memory_id),
        )
        return tuple(
            LinkRow(
                guild_id=r["guild_id"],
                memory_id=r["memory_id"],
                node_type=NodeType(r["node_type"]),
                node_id=r["node_id"],
                kind=EdgeKind(r["kind"]),
                created_at=parse_moment(r["created_at"]),
            )
            for r in rows
        )

    async def links_for_nodes(
        self,
        guild_id: str,
        nodes: tuple[NodeRef, ...],
        *,
        active_only: bool = True,
        limit_per_node: int = 50,
    ) -> tuple[tuple[LinkRow, FactRecord], ...]:
        from icelake.adapters.sqlite.store_facts import record_from_row

        if not nodes:
            return ()
        unique = tuple(dict.fromkeys(nodes))
        marks = ",".join("(?,?)" for _ in unique)
        params: list[object] = [guild_id]
        for node_type, node_id in unique:
            params.extend([node_type.value, node_id])
        conditions = ["l.guild_id=?", f"(l.node_type, l.node_id) IN ({marks})"]
        if active_only:
            conditions.append("f.valid_until IS NULL AND f.superseded_by_id IS NULL")
        # Fetch bounded by the worst case, then apply per-node caps in Python —
        # portable across backends and the pools here are small by config.
        params.append(limit_per_node * len(unique))
        rows = await self._db.query(
            f"""SELECT l.*, f.id AS f_id, f.*
                FROM dm_links l JOIN dm_facts f ON f.id=l.memory_id AND f.guild_id=l.guild_id
                WHERE {" AND ".join(conditions)}
                LIMIT ?""",
            tuple(params),
        )
        results: list[tuple[LinkRow, FactRecord]] = []
        per_node: dict[tuple[str, str], int] = {}
        for row in rows:
            key = (row["node_type"], row["node_id"])
            if per_node.get(key, 0) >= limit_per_node:
                continue
            per_node[key] = per_node.get(key, 0) + 1
            link = LinkRow(
                guild_id=row["guild_id"],
                memory_id=row["memory_id"],
                node_type=NodeType(row["node_type"]),
                node_id=row["node_id"],
                kind=EdgeKind(row["kind"]),
                created_at=parse_moment(row["created_at"]),
            )
            results.append((link, record_from_row(row)))
        return tuple(results)

    # -- relations --------------------------------------------------------------------

    def _edge_from_row(self, row: sqlite3.Row) -> RelationEdge:
        return RelationEdge(
            guild_id=row["guild_id"],
            src_type=NodeType(row["src_type"]),
            src_id=row["src_id"],
            dst_type=NodeType(row["dst_type"]),
            dst_id=row["dst_id"],
            verb=row["verb"],
            polarity=Polarity(row["polarity"]),
            weight=float(row["weight"]),
            occurrences=int(row["occurrences"]),
            confidence=float(row["confidence"]),
            evidence_fact_ids=tuple(json.loads(row["evidence_ids"] or "[]")),
            valid_from=parse_moment(row["valid_from"]),
            valid_until=parse_moment(row["valid_until"]),
        )

    async def upsert_relation(self, edge: RelationEdge) -> RelationEdge:
        existing_rows = await self._db.query(
            """SELECT * FROM dm_relations
               WHERE guild_id=? AND src_type=? AND src_id=? AND dst_type=? AND dst_id=?
                 AND verb=? AND valid_until IS NULL""",
            (
                edge.guild_id,
                edge.src_type.value,
                edge.src_id,
                edge.dst_type.value,
                edge.dst_id,
                edge.verb,
            ),
        )
        now_iso = iso(edge.valid_from or edge.valid_until)
        if not existing_rows:
            await self._db.execute(
                """INSERT INTO dm_relations
                   (guild_id, src_type, src_id, dst_type, dst_id, verb, polarity,
                    weight, occurrences, confidence, evidence_ids, valid_from,
                    valid_until)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)""",
                (
                    edge.guild_id,
                    edge.src_type.value,
                    edge.src_id,
                    edge.dst_type.value,
                    edge.dst_id,
                    edge.verb,
                    edge.polarity.value,
                    edge.weight,
                    edge.occurrences,
                    edge.confidence,
                    dumps(list(edge.evidence_fact_ids)),
                    now_iso,
                ),
            )
            rows = await self._db.query(
                """SELECT * FROM dm_relations
                   WHERE guild_id=? AND src_type=? AND src_id=? AND dst_type=? AND dst_id=?
                     AND verb=? AND valid_until IS NULL""",
                (
                    edge.guild_id,
                    edge.src_type.value,
                    edge.src_id,
                    edge.dst_type.value,
                    edge.dst_id,
                    edge.verb,
                ),
            )
            return self._edge_from_row(rows[0])
        current = self._edge_from_row(existing_rows[0])
        evidence = list(dict.fromkeys(current.evidence_fact_ids + edge.evidence_fact_ids))[-8:]
        occurrences = current.occurrences + 1
        weight = max(current.weight, edge.weight)
        await self._db.execute(
            """UPDATE dm_relations SET occurrences=?, weight=?, confidence=?, evidence_ids=?
               WHERE edge_id=?""",
            (
                occurrences,
                weight,
                max(current.confidence, edge.confidence),
                dumps(evidence),
                existing_rows[0]["edge_id"],
            ),
        )
        return await self.get_edge_by_id(existing_rows[0]["edge_id"])

    async def get_edge_by_id(self, edge_id: int) -> RelationEdge:
        row = await self._db.query_one(
            "SELECT * FROM dm_relations WHERE edge_id=?",
            (edge_id,),
        )
        assert row is not None
        return self._edge_from_row(row)

    async def edges_between(
        self,
        guild_id: str,
        src: NodeRef,
        dst: NodeRef,
    ) -> tuple[RelationEdge, ...]:
        rows = await self._db.query(
            """SELECT * FROM dm_relations
               WHERE guild_id=? AND src_type=? AND src_id=? AND dst_type=? AND dst_id=?
                 AND valid_until IS NULL""",
            (guild_id, src[0].value, src[1], dst[0].value, dst[1]),
        )
        return tuple(self._edge_from_row(r) for r in rows)

    async def incident_edges(
        self,
        guild_id: str,
        node: NodeRef,
        *,
        limit: int = 50,
    ) -> tuple[RelationEdge, ...]:
        rows = await self._db.query(
            """SELECT * FROM dm_relations
               WHERE guild_id=? AND valid_until IS NULL
                 AND ((src_type=? AND src_id=?) OR (dst_type=? AND dst_id=?))
               ORDER BY weight DESC LIMIT ?""",
            (guild_id, node[0].value, node[1], node[0].value, node[1], limit),
        )
        return tuple(self._edge_from_row(r) for r in rows)

    async def incident_edges_many(
        self,
        guild_id: str,
        nodes: tuple[NodeRef, ...],
        *,
        limit_per_node: int = 50,
    ) -> tuple[RelationEdge, ...]:
        if not nodes:
            return ()
        unique = tuple(dict.fromkeys(nodes))
        marks = ",".join("(?,?)" for _ in unique)
        params: list[object] = [guild_id]
        for _ in range(2):  # src IN (...) first, then dst IN (...)
            for node_type, node_id in unique:
                params.extend([node_type.value, node_id])
        rows = await self._db.query(
            f"""SELECT * FROM dm_relations
                WHERE guild_id=? AND valid_until IS NULL
                  AND ((src_type, src_id) IN ({marks}) OR (dst_type, dst_id) IN ({marks}))
                ORDER BY weight DESC LIMIT ?""",
            (*params, limit_per_node * len(unique)),
        )
        results: list[RelationEdge] = []
        per_node: dict[tuple[str, str], int] = {}
        wanted = {(t.value, i) for t, i in unique}
        for row in rows:
            edge = self._edge_from_row(row)
            touches = [
                key
                for key in ((edge.src_type.value, edge.src_id), (edge.dst_type.value, edge.dst_id))
                if key in wanted
            ]
            if any(per_node.get(key, 0) >= limit_per_node for key in touches):
                continue
            for key in touches:
                per_node[key] = per_node.get(key, 0) + 1
            results.append(edge)
        return tuple(results)

    async def edges_to_nodes(
        self,
        guild_id: str,
        nodes: tuple[NodeRef, ...],
        *,
        limit: int = 500,
    ) -> tuple[RelationEdge, ...]:
        if not nodes:
            return ()
        unique = tuple(dict.fromkeys(nodes))
        marks = ",".join("(?,?)" for _ in unique)
        params: list[object] = [guild_id]
        for node_type, node_id in unique:
            params.extend([node_type.value, node_id])
        params.append(limit)
        rows = await self._db.query(
            f"""SELECT * FROM dm_relations
                WHERE guild_id=? AND valid_until IS NULL
                  AND (dst_type, dst_id) IN ({marks})
                ORDER BY weight DESC LIMIT ?""",
            tuple(params),
        )
        return tuple(self._edge_from_row(r) for r in rows)

    async def drop_evidence_from_edges(
        self,
        guild_id: str,
        fact_id: str,
        until: datetime,
    ) -> int:
        rows = await self._db.query(
            """SELECT edge_id, evidence_ids FROM dm_relations
               WHERE guild_id=? AND valid_until IS NULL AND evidence_ids LIKE ?""",
            (guild_id, f'%"{fact_id}"%'),
        )
        changed = 0
        for row in rows:
            evidence = [fid for fid in json.loads(row["evidence_ids"] or "[]") if fid != fact_id]
            if evidence:
                await self._db.execute(
                    "UPDATE dm_relations SET evidence_ids=?, weight=weight*0.8 WHERE edge_id=?",
                    (dumps(evidence), row["edge_id"]),
                )
            else:
                await self._db.execute(
                    "UPDATE dm_relations SET valid_until=? WHERE edge_id=?",
                    (iso(until), row["edge_id"]),
                )
            changed += 1
        return changed

    async def entity_stance_edges(
        self,
        guild_id: str,
        entity_slug: str,
        *,
        polarity: Polarity | None = None,
        limit: int = 25,
    ) -> tuple[RelationEdge, ...]:
        conditions = [
            "guild_id=?",
            "valid_until IS NULL",
            "dst_type='entity'",
            "dst_id=?",
        ]
        params: list[object] = [guild_id, entity_slug]
        if polarity is not None:
            conditions.append("polarity=?")
            params.append(polarity.value)
        params.append(limit)
        rows = await self._db.query(
            f"""SELECT * FROM dm_relations WHERE {" AND ".join(conditions)}
                ORDER BY weight DESC LIMIT ?""",
            tuple(params),
        )
        return tuple(self._edge_from_row(r) for r in rows)

    # -- entities ----------------------------------------------------------------------

    async def upsert_entity(
        self,
        guild_id: str,
        slug: str,
        name: str,
        kind: EntityKind,
        aliases: tuple[str, ...] = (),
    ) -> EntityRecord:
        await self._db.execute(
            """INSERT INTO dm_entities (guild_id, slug, name, kind, aliases_json)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(guild_id, slug) DO UPDATE SET name=excluded.name""",
            (guild_id, slug, name, kind, dumps(list(aliases))),
        )
        for alias in dict.fromkeys(aliases):
            await self._db.execute(
                "INSERT OR IGNORE INTO dm_entity_aliases (guild_id, alias_norm, slug)"
                " VALUES (?, ?, ?)",
                (guild_id, alias, slug),
            )
        return await self.get_entity(guild_id, slug) or EntityRecord(
            guild_id=guild_id,
            slug=slug,
            name=name,
            kind=kind,
            aliases=aliases,
        )

    async def bump_entity_facts(self, guild_id: str, slug: str, delta: int = 1) -> None:
        await self._db.execute(
            "UPDATE dm_entities SET fact_count=fact_count+? WHERE guild_id=? AND slug=?",
            (delta, guild_id, slug),
        )

    async def get_entity(self, guild_id: str, slug: str) -> EntityRecord | None:
        row = await self._db.query_one(
            "SELECT * FROM dm_entities WHERE guild_id=? AND slug=?",
            (guild_id, slug),
        )
        if row is None:
            return None
        return EntityRecord(
            guild_id=guild_id,
            slug=row["slug"],
            name=row["name"],
            kind=row["kind"],
            aliases=tuple(json.loads(row["aliases_json"] or "[]")),
            fact_count=int(row["fact_count"]),
            linked_user_id=row["linked_user_id"],
            summary=row["summary"] or "",
        )

    async def resolve_entity_alias(self, guild_id: str, alias_norm: str) -> str | None:
        row = await self._db.query_one(
            "SELECT slug FROM dm_entity_aliases WHERE guild_id=? AND alias_norm=?",
            (guild_id, alias_norm),
        )
        return row["slug"] if row else None

    async def merge_entities(
        self,
        guild_id: str,
        from_slugs: tuple[str, ...],
        to_slug: str,
    ) -> int:
        moved = 0
        for slug in from_slugs:
            source = await self.get_entity(guild_id, slug)
            if source is None:
                continue
            target = await self.get_entity(guild_id, to_slug)
            merged_aliases = tuple(
                dict.fromkeys(
                    (target.aliases if target else ()) + source.aliases + (slug,),
                )
            )
            await self._db.execute(
                """INSERT INTO dm_entities (guild_id, slug, name, kind, aliases_json,
                                            fact_count)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(guild_id, slug) DO UPDATE SET
                     fact_count=fact_count+?,
                     aliases_json=excluded.aliases_json""",
                (
                    guild_id,
                    to_slug,
                    target.name if target else source.name,
                    source.kind,
                    dumps(list(merged_aliases)),
                    source.fact_count,
                    source.fact_count,
                ),
            )
            await self._db.execute(
                "DELETE FROM dm_entities WHERE guild_id=? AND slug=?",
                (guild_id, slug),
            )
            moved += 1
        return moved
