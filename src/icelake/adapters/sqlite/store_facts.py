"""SQLite MemoryStore mixin: facts, history, summaries, consent, purge, stats."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from icelake.adapters.sqlite.connection import SqliteConnection, dumps, iso
from icelake.lifecycle.prune import select_prune_victims_by_anchor
from icelake.models.admin import GuildStats, PurgeReport
from icelake.models.common import Page
from icelake.models.facts import (
    Attribution,
    AttributionType,
    FactCategory,
    FactHistoryEntry,
    FactRecord,
    FactScope,
    MemoryTier,
    ProfileSummary,
    SourceRef,
)
from icelake.models.graph import EntityRecord, RelationEdge


def parse_moment(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _validity_sql(as_of: datetime | None) -> tuple[str, list[object]]:
    """Active-now predicate, or point-in-time validity when ``as_of`` is set."""
    if as_of is None:
        return ("valid_until IS NULL AND superseded_by_id IS NULL", [])
    return (
        "(valid_from IS NULL OR valid_from <= ?) AND (valid_until IS NULL OR valid_until > ?)",
        [iso(as_of), iso(as_of)],
    )


def record_from_row(row: sqlite3.Row) -> FactRecord:
    attribution = json.loads(row["attribution"] or "{}")
    return FactRecord(
        id=row["id"],
        guild_id=row["guild_id"],
        subject_id=row["subject_id"],
        text=row["text"],
        text_normalized=row["text_normalized"],
        category=FactCategory(row["category"]),
        confidence=float(row["confidence"]),
        tier=MemoryTier(row["tier"]),
        scope=row["scope"],
        attribution=Attribution(
            type=AttributionType(attribution.get("type", "self")),
            speaker_id=attribution.get("speaker_id"),
            speaker_name=attribution.get("speaker_name"),
            actor_id=attribution.get("actor_id"),
        ),
        occurrences=int(row["occurrences"]),
        strength=float(row["strength"]),
        last_reinforced_at=parse_moment(row["last_reinforced_at"]),
        created_at=parse_moment(row["created_at"]),
        updated_at=parse_moment(row["updated_at"]),
        observed_at=parse_moment(row["observed_at"]),
        valid_from=parse_moment(row["valid_from"]),
        valid_until=parse_moment(row["valid_until"]),
        supersedes_id=row["supersedes_id"],
        superseded_by_id=row["superseded_by_id"],
        citations=tuple(SourceRef.model_validate(c) for c in json.loads(row["citations"] or "[]")),
        related_user_ids=tuple(json.loads(row["related_user_ids"] or "[]")),
        entity_slugs=tuple(json.loads(row["entity_slugs"] or "[]")),
        tags=tuple(json.loads(row["tags"] or "[]")),
        expires_at=parse_moment(row["expires_at"]),
        version=int(row["version"]),
    )


def fact_insert_params(record: FactRecord) -> tuple[object, ...]:
    attribution = {
        "type": record.attribution.type.value,
        "speaker_id": record.attribution.speaker_id,
        "speaker_name": record.attribution.speaker_name,
        "actor_id": record.attribution.actor_id,
    }
    return (
        record.id,
        record.guild_id,
        record.subject_id,
        record.text,
        record.text_normalized,
        record.category.value,
        record.confidence,
        record.tier.value,
        record.scope.value,
        json.dumps(attribution),
        record.occurrences,
        record.strength,
        iso(record.last_reinforced_at),
        iso(record.created_at),
        iso(record.updated_at),
        iso(record.observed_at),
        iso(record.valid_from),
        iso(record.valid_until),
        record.supersedes_id,
        record.superseded_by_id,
        dumps([c.model_dump(mode="json") for c in record.citations]),
        dumps(list(record.related_user_ids)),
        dumps(list(record.entity_slugs)),
        dumps(list(record.tags)),
        iso(record.expires_at),
        record.version,
    )


class FactsMixin:
    """Facts/history/summaries/consent/purge/stats over ``self._db``."""

    _db: SqliteConnection

    async def insert_fact(self, record: FactRecord) -> None:
        await self._db.execute(
            """INSERT INTO dm_facts (
                 id, guild_id, subject_id, text, text_normalized, category, confidence,
                 tier, scope, attribution, occurrences, strength, last_reinforced_at,
                 created_at, updated_at, observed_at, valid_from, valid_until,
                 supersedes_id, superseded_by_id, citations, related_user_ids,
                 entity_slugs, tags, expires_at, version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            fact_insert_params(record),
        )
        await self._db.execute(
            "INSERT INTO dm_facts_fts(rowid, text) VALUES "
            "((SELECT seq FROM dm_facts WHERE id=?), ?)",
            (record.id, record.text),
        )

    async def get_fact(self, guild_id: str, fact_id: str) -> FactRecord | None:
        row = await self._db.query_one(
            "SELECT * FROM dm_facts WHERE guild_id=? AND id=?",
            (guild_id, fact_id),
        )
        return record_from_row(row) if row else None

    async def get_facts(
        self,
        guild_id: str,
        fact_ids: tuple[str, ...],
    ) -> tuple[FactRecord, ...]:
        found: list[FactRecord] = []
        for fact_id in fact_ids:
            record = await self.get_fact(guild_id, fact_id)
            if record is not None:
                found.append(record)
        return tuple(found)

    async def find_duplicate(
        self,
        guild_id: str,
        subject_id: str | None,
        text_normalized: str,
    ) -> FactRecord | None:
        row = await self._db.query_one(
            """SELECT * FROM dm_facts
               WHERE guild_id=? AND subject_id IS ? AND text_normalized=?
                 AND valid_until IS NULL AND superseded_by_id IS NULL
               LIMIT 1""",
            (guild_id, subject_id, text_normalized),
        )
        return record_from_row(row) if row else None

    async def reinforce_fact(
        self,
        guild_id: str,
        fact_id: str,
        *,
        occurrences_delta: int,
        strength: float,
        last_reinforced_at: datetime,
        expires_at: datetime | None,
        tier: MemoryTier,
        confidence: float,
        extra_citations: tuple[SourceRef, ...] = (),
    ) -> FactRecord | None:
        current = await self.get_fact(guild_id, fact_id)
        if current is None:
            return None
        merged_citations: list[SourceRef] = []
        if extra_citations:
            merged_citations = (list(current.citations) + list(extra_citations))[:8]
        citation_sql = "citations=?, " if extra_citations else ""
        citation_param = (
            [dumps([c.model_dump(mode="json") for c in merged_citations])]
            if extra_citations
            else []
        )
        # occurrences increments server-side: concurrent reinforcements cannot
        # lose counts (optimistic concurrency on a read-modify-write would).
        await self._db.execute(
            f"""UPDATE dm_facts SET occurrences=occurrences+?, strength=?,
                 last_reinforced_at=?, expires_at=?, tier=?, confidence=?,
                 {citation_sql}version=version+1, updated_at=?
               WHERE guild_id=? AND id=? AND valid_until IS NULL
                 AND superseded_by_id IS NULL""",
            (
                occurrences_delta,
                strength,
                iso(last_reinforced_at),
                iso(expires_at),
                tier.value,
                confidence,
                *citation_param,
                iso(last_reinforced_at),
                guild_id,
                fact_id,
            ),
        )
        return await self.get_fact(guild_id, fact_id)

    async def transition_fact(
        self,
        guild_id: str,
        fact_id: str,
        *,
        valid_until: datetime | None = None,
        superseded_by_id: str | None = None,
        updated_at: datetime,
    ) -> FactRecord | None:
        sets = ["updated_at=?", "version=version+1"]
        params: list[object] = [iso(updated_at)]
        if valid_until is not None:
            sets.append("valid_until=?")
            params.append(iso(valid_until))
        if superseded_by_id is not None:
            sets.append("superseded_by_id=?")
            params.append(superseded_by_id)
        params.extend([guild_id, fact_id])
        existing = await self.get_fact(guild_id, fact_id)
        if existing is None:
            return None
        await self._db.execute(
            f"UPDATE dm_facts SET {', '.join(sets)} WHERE guild_id=? AND id=?",
            tuple(params),
        )
        return await self.get_fact(guild_id, fact_id)

    async def update_fact_fields(
        self,
        guild_id: str,
        fact_id: str,
        *,
        text: str | None = None,
        text_normalized: str | None = None,
        category: FactCategory | None = None,
        confidence: float | None = None,
        tier: MemoryTier | None = None,
        expires_at: datetime | None = None,
        updated_at: datetime,
    ) -> FactRecord | None:
        existing = await self.get_fact(guild_id, fact_id)
        if existing is None:
            return None
        sets = ["updated_at=?", "version=version+1"]
        params: list[object] = [iso(updated_at)]
        if text is not None:
            sets.append("text=?")
            params.append(text)
        if text_normalized is not None:
            sets.append("text_normalized=?")
            params.append(text_normalized)
        if category is not None:
            sets.append("category=?")
            params.append(category.value)
        if confidence is not None:
            sets.append("confidence=?")
            params.append(confidence)
        if tier is not None:
            sets.append("tier=?")
            params.append(tier.value)
        if expires_at is not None:
            sets.append("expires_at=?")
            params.append(iso(expires_at))
        params.extend([guild_id, fact_id])
        await self._db.execute(
            f"UPDATE dm_facts SET {', '.join(sets)} WHERE guild_id=? AND id=?",
            tuple(params),
        )
        return await self.get_fact(guild_id, fact_id)

    async def list_facts(
        self,
        guild_id: str,
        *,
        subject_id: str | None,
        include_server: bool = False,
        active_only: bool = True,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Page[FactRecord]:
        conditions = ["guild_id=?"]
        params: list[object] = [guild_id]
        if subject_id is None:
            conditions.append("subject_id IS NULL")
        else:
            conditions.append("(subject_id=? OR (? AND subject_id IS NULL))")
            params.extend([subject_id, int(include_server)])
        if active_only:
            conditions.append("valid_until IS NULL AND superseded_by_id IS NULL")
        if cursor:
            conditions.append("id>?")
            params.append(cursor)
        params.append(limit)
        rows = await self._db.query(
            f"SELECT * FROM dm_facts WHERE {' AND '.join(conditions)} ORDER BY id LIMIT ?",
            tuple(params),
        )
        items = tuple(record_from_row(r) for r in rows)
        next_cursor = items[-1].id if len(items) == limit and items else None
        return Page(items=items, next_cursor=next_cursor)

    async def top_strength_facts(
        self,
        guild_id: str,
        *,
        subject_ids: tuple[str, ...] | None,
        server_only: bool = False,
        limit: int = 10,
        as_of: datetime | None = None,
    ) -> tuple[FactRecord, ...]:
        """Strength-ranked anchors with the scope predicate pushed into SQL.

        Post-filtering after a global LIMIT starved subjects once a guild grew
        past the cap — every user's anchors must be found regardless of guild
        size.
        """
        conditions = ["guild_id=?"]
        params: list[object] = [guild_id]
        validity_sql, validity_params = _validity_sql(as_of)
        conditions.append(validity_sql)
        params.extend(validity_params)
        if server_only:
            conditions.append("scope='server'")
        elif subject_ids is not None and subject_ids:
            placeholders = ",".join("?" * len(subject_ids))
            conditions.append(
                f"(subject_id IN ({placeholders})"
                f" OR EXISTS (SELECT 1 FROM dm_links l"
                f" WHERE l.memory_id=dm_facts.id AND l.node_type='user'"
                f" AND l.node_id IN ({placeholders})))"
            )
            params.extend([*subject_ids, *subject_ids])
        params.append(limit)
        rows = await self._db.query(
            f"SELECT * FROM dm_facts WHERE {' AND '.join(conditions)} "
            f"ORDER BY strength DESC, confidence DESC LIMIT ?",
            tuple(params),
        )
        return tuple(record_from_row(r) for r in rows)

    async def search_facts_text(
        self,
        guild_id: str,
        query: str,
        *,
        subject_ids: tuple[str, ...] | None = None,
        server_only: bool = False,
        limit: int = 20,
        as_of: datetime | None = None,
    ) -> tuple[tuple[FactRecord, float], ...]:
        """One joined FTS query — the old per-hit N+1 fetched up to 201 rows."""
        safe_query = " ".join(
            token for token in "".join(ch if ch.isalnum() else " " for ch in query).split()
        )
        if not safe_query:
            return ()
        conditions = ["f.guild_id=?"]
        params: list[object] = [guild_id]
        if as_of is not None:
            conditions.append(
                "(f.valid_from IS NULL OR f.valid_from <= ?)"
                " AND (f.valid_until IS NULL OR f.valid_until > ?)"
            )
            params.extend([iso(as_of), iso(as_of)])
        else:
            conditions.append("f.valid_until IS NULL")
            conditions.append("f.superseded_by_id IS NULL")
        if server_only:
            conditions.append("f.scope='server'")
        elif subject_ids is not None and subject_ids:
            placeholders = ",".join("?" * len(subject_ids))
            conditions.append(
                f"(f.subject_id IN ({placeholders})"
                f" OR EXISTS (SELECT 1 FROM dm_links l"
                f" WHERE l.memory_id=f.id AND l.node_type='user'"
                f" AND l.node_id IN ({placeholders})))"
            )
            params.extend([*subject_ids, *subject_ids])
        params.append(limit)
        rows = await self._db.query(
            f"""SELECT f.*, bm25(dm_facts_fts) AS rank
                FROM dm_facts_fts j JOIN dm_facts f ON f.seq=j.rowid
                WHERE dm_facts_fts MATCH ? AND {" AND ".join(conditions)}
                ORDER BY rank LIMIT ?""",
            (safe_query, *params),
        )
        scored = [(record_from_row(r), min(1.0, 1.0 / (1.0 + abs(float(r["rank"]))))) for r in rows]
        scored.sort(key=lambda pair: -pair[1])
        return tuple(scored[:limit])

    async def append_history(
        self,
        guild_id: str,
        fact_id: str,
        entry: FactHistoryEntry,
    ) -> None:
        await self._db.execute(
            """INSERT INTO dm_history (guild_id, fact_id, at, kind, detail, fact_version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                guild_id,
                fact_id,
                iso(entry.at),
                entry.kind.value,
                entry.detail,
                entry.fact_version,
            ),
        )

    async def get_history(
        self,
        guild_id: str,
        fact_id: str,
    ) -> tuple[FactHistoryEntry, ...]:
        rows = await self._db.query(
            "SELECT * FROM dm_history WHERE guild_id=? AND fact_id=? ORDER BY seq",
            (guild_id, fact_id),
        )
        return tuple(
            FactHistoryEntry(
                at=parse_moment(r["at"]) or datetime.now(),
                kind=r["kind"],
                detail=r["detail"],
                fact_version=int(r["fact_version"]),
            )
            for r in rows
        )

    async def get_cursor(self, guild_id: str, key: str) -> str | None:
        row = await self._db.query_one(
            "SELECT value FROM dm_cursors WHERE guild_id=? AND key=?",
            (guild_id, key),
        )
        return row["value"] if row else None

    async def set_cursor(self, guild_id: str, key: str, value: str) -> None:
        await self._db.execute(
            """INSERT INTO dm_cursors (guild_id, key, value) VALUES (?, ?, ?)
               ON CONFLICT(guild_id, key) DO UPDATE SET value=excluded.value""",
            (guild_id, key, value),
        )

    async def list_guild_ids(self) -> tuple[str, ...]:
        rows = await self._db.query(
            "SELECT guild_id FROM dm_facts UNION SELECT guild_id FROM dm_messages"
        )
        return tuple(row["guild_id"] for row in rows)

    async def charge_guild_tokens(
        self,
        guild_id: str,
        *,
        day_key: str,
        month_key: str,
        prompt_tokens: int,
    ) -> tuple[int, int]:
        totals: list[int] = []
        for period in (day_key, month_key):
            rows = await self._db.execute_returning(
                """INSERT INTO dm_budgets (guild_id, period_key, prompt_tokens)
                   VALUES (?, ?, ?)
                   ON CONFLICT(guild_id, period_key)
                   DO UPDATE SET prompt_tokens = prompt_tokens + excluded.prompt_tokens
                   RETURNING prompt_tokens""",
                (guild_id, period, prompt_tokens),
            )
            totals.append(int(rows[0]["prompt_tokens"]))
        return totals[0], totals[1]

    async def guild_token_usage(
        self,
        guild_id: str,
        *,
        day_key: str,
        month_key: str,
    ) -> tuple[int, int]:
        rows = await self._db.query(
            "SELECT period_key, prompt_tokens FROM dm_budgets "
            "WHERE guild_id=? AND period_key IN (?, ?)",
            (guild_id, day_key, month_key),
        )
        by_period = {row["period_key"]: int(row["prompt_tokens"]) for row in rows}
        return by_period.get(day_key, 0), by_period.get(month_key, 0)

    async def get_summary(
        self,
        guild_id: str,
        subject_id: str | None,
    ) -> ProfileSummary | None:
        row = await self._db.query_one(
            "SELECT * FROM dm_summaries WHERE guild_id=? AND subject_key=?",
            (guild_id, subject_id or "__server__"),
        )
        if row is None:
            return None
        return ProfileSummary(
            guild_id=guild_id,
            subject_id=None if row["subject_key"] == "__server__" else row["subject_key"],
            text=row["text"],
            generated_at=parse_moment(row["generated_at"]),
            source_fact_count=int(row["source_fact_count"]),
        )

    async def put_summary(self, summary: ProfileSummary) -> None:
        await self._db.execute(
            """INSERT INTO dm_summaries (guild_id, subject_key, text, generated_at,
                                         source_fact_count)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(guild_id, subject_key) DO UPDATE SET
                 text=excluded.text, generated_at=excluded.generated_at,
                 source_fact_count=excluded.source_fact_count""",
            (
                summary.guild_id,
                summary.subject_id or "__server__",
                summary.text,
                iso(summary.generated_at),
                summary.source_fact_count,
            ),
        )

    async def delete_summary(self, guild_id: str, subject_id: str | None) -> int:
        key = subject_id or "__server__"
        row = await self._db.query_one(
            "SELECT subject_key FROM dm_summaries WHERE guild_id=? AND subject_key=?",
            (guild_id, key),
        )
        if row is None:
            return 0
        await self._db.execute(
            "DELETE FROM dm_summaries WHERE guild_id=? AND subject_key=?",
            (guild_id, key),
        )
        return 1

    async def set_opt_out(self, guild_id: str, user_id: str, opted_out: bool) -> None:
        if opted_out:
            await self._db.execute(
                "INSERT OR IGNORE INTO dm_optouts (guild_id, user_id) VALUES (?, ?)",
                (guild_id, user_id),
            )
        else:
            await self._db.execute(
                "DELETE FROM dm_optouts WHERE guild_id=? AND user_id=?",
                (guild_id, user_id),
            )

    async def get_opt_out(self, guild_id: str, user_id: str) -> bool:
        row = await self._db.query_one(
            "SELECT user_id FROM dm_optouts WHERE guild_id=? AND user_id=?",
            (guild_id, user_id),
        )
        return row is not None

    async def purge_user_data(self, guild_id: str, user_id: str, dry_run: bool) -> PurgeReport:
        fact_rows = await self._db.query(
            "SELECT id, seq FROM dm_facts WHERE guild_id=? AND subject_id=?",
            (guild_id, user_id),
        )
        link_rows = await self._db.query(
            """SELECT memory_id FROM dm_links WHERE guild_id=?
                 AND (node_type='user' AND node_id=?)
                    OR memory_id IN (SELECT id FROM dm_facts WHERE guild_id=? AND subject_id=?)""",
            (guild_id, user_id, guild_id, user_id),
        )
        edge_rows = await self._db.query(
            """SELECT edge_id FROM dm_relations WHERE guild_id=?
                 AND ((src_type='user' AND src_id=?) OR (dst_type='user' AND dst_id=?))""",
            (guild_id, user_id, user_id),
        )
        alias_rows = await self._db.query(
            "SELECT alias_norm FROM dm_aliases WHERE guild_id=? AND user_id=?",
            (guild_id, user_id),
        )
        report = PurgeReport(
            guild_id=guild_id,
            subject_id=user_id,
            dry_run=dry_run,
            facts_removed=len(fact_rows),
            links_removed=len(link_rows),
            edges_removed=len(edge_rows),
            aliases_removed=len(alias_rows),
            summaries_removed=0,
            vectors_removed=len(fact_rows),
        )
        if dry_run:
            return report
        for row in fact_rows:
            await self._db.execute(
                "DELETE FROM dm_vectors WHERE fact_id=?",
                (row["id"],),
            )
            await self._db.execute("DELETE FROM dm_facts_fts WHERE rowid=?", (row["seq"],))
        await self._db.execute(
            "DELETE FROM dm_facts WHERE guild_id=? AND subject_id=?",
            (guild_id, user_id),
        )
        await self._db.execute(
            "DELETE FROM dm_links WHERE guild_id=? AND node_type='user' AND node_id=?",
            (guild_id, user_id),
        )
        await self._db.execute(
            """DELETE FROM dm_links WHERE guild_id=? AND memory_id NOT IN
                 (SELECT id FROM dm_facts WHERE guild_id=?)""",
            (guild_id, guild_id),
        )
        await self._db.execute(
            """DELETE FROM dm_relations WHERE guild_id=?
                 AND ((src_type='user' AND src_id=?) OR (dst_type='user' AND dst_id=?))""",
            (guild_id, user_id, user_id),
        )
        await self._db.execute(
            "DELETE FROM dm_aliases WHERE guild_id=? AND user_id=?",
            (guild_id, user_id),
        )
        await self._db.execute(
            "DELETE FROM dm_summaries WHERE guild_id=? AND subject_key=?",
            (guild_id, user_id),
        )
        await self._db.execute(
            "DELETE FROM dm_optouts WHERE guild_id=? AND user_id=?",
            (guild_id, user_id),
        )
        return report

    async def export_guild(
        self, guild_id: str
    ) -> tuple[
        tuple[FactRecord, ...],
        tuple[EntityRecord, ...],
        tuple[RelationEdge, ...],
    ]:
        fact_rows = await self._db.query(
            "SELECT * FROM dm_facts WHERE guild_id=? ORDER BY created_at",
            (guild_id,),
        )
        entity_rows = await self._db.query(
            "SELECT * FROM dm_entities WHERE guild_id=?",
            (guild_id,),
        )
        relation_rows = await self._db.query(
            "SELECT * FROM dm_relations WHERE guild_id=?",
            (guild_id,),
        )
        from icelake.models.graph import EntityRecord, RelationEdge

        entities = tuple(
            EntityRecord(
                guild_id=guild_id,
                slug=r["slug"],
                name=r["name"],
                kind=r["kind"],
                aliases=tuple(json.loads(r["aliases_json"] or "[]")),
                fact_count=int(r["fact_count"]),
                linked_user_id=r["linked_user_id"],
                summary=r["summary"] or "",
            )
            for r in entity_rows
        )
        relations = tuple(
            RelationEdge(
                guild_id=guild_id,
                src_type=r["src_type"],
                src_id=r["src_id"],
                dst_type=r["dst_type"],
                dst_id=r["dst_id"],
                verb=r["verb"],
                polarity=r["polarity"],
                weight=float(r["weight"]),
                occurrences=int(r["occurrences"]),
                confidence=float(r["confidence"]),
                evidence_fact_ids=tuple(json.loads(r["evidence_ids"] or "[]")),
                valid_from=parse_moment(r["valid_from"]),
                valid_until=parse_moment(r["valid_until"]),
            )
            for r in relation_rows
        )
        facts = tuple(record_from_row(r) for r in fact_rows)
        return facts, entities, relations

    async def touch_facts(
        self,
        guild_id: str,
        fact_ids: tuple[str, ...],
        *,
        accessed_at: datetime,
    ) -> int:
        """Reset the decay clock on recalled facts in one batched statement."""
        ids = [fid for fid in fact_ids if fid]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        await self._db.execute(
            f"""UPDATE dm_facts SET last_reinforced_at=?, last_accessed_at=?,
                 version=version+1
               WHERE guild_id=? AND id IN ({placeholders})""",
            (iso(accessed_at), iso(accessed_at), guild_id, *ids),
        )
        return len(ids)

    async def sweep_expired(self, guild_id: str, now: datetime) -> int:
        await self._db.execute(
            """UPDATE dm_facts SET valid_until=?, updated_at=?, version=version+1
               WHERE guild_id=? AND expires_at IS NOT NULL AND expires_at<=?
                 AND valid_until IS NULL AND superseded_by_id IS NULL""",
            (iso(now), iso(now), guild_id, iso(now)),
        )
        row = await self._db.query_one(
            """SELECT COUNT(*) AS n FROM dm_facts WHERE guild_id=?
                 AND valid_until=?""",
            (guild_id, iso(now)),
        )
        return int(row["n"]) if row else 0

    async def prune_to_caps(
        self,
        guild_id: str,
        *,
        max_per_user: int,
        max_server: int,
        now: datetime,
    ) -> int:
        rows = await self._db.query(
            """SELECT * FROM dm_facts WHERE guild_id=?
                 AND valid_until IS NULL AND superseded_by_id IS NULL""",
            (guild_id,),
        )
        victims = select_prune_victims_by_anchor(
            tuple(record_from_row(row) for row in rows),
            max_per_user=max_per_user,
            max_server=max_server,
        )
        for victim in victims:
            await self._db.execute(
                """UPDATE dm_facts SET valid_until=?, updated_at=?,
                     version=version+1 WHERE id=?""",
                (iso(now), iso(now), victim.id),
            )
        return len(victims)

    async def apply_forgetting(
        self,
        guild_id: str,
        *,
        now: datetime,
        retention_floor: float,
    ) -> int:
        import math

        rows = await self._db.query(
            """SELECT id, tier, attribution, strength, last_reinforced_at,
                      created_at FROM dm_facts
               WHERE guild_id=? AND valid_until IS NULL
                 AND superseded_by_id IS NULL""",
            (guild_id,),
        )
        forgotten = 0
        for row in rows:
            tier = row["tier"]
            if tier == "core":
                continue
            attribution = json.loads(row["attribution"] or "{}")
            if attribution.get("type") == "manual":
                continue
            last = parse_moment(row["last_reinforced_at"]) or parse_moment(
                row["created_at"],
            )
            if last is None:
                continue
            delta_days = max(
                0.0,
                (now - last).total_seconds() / 86_400.0,
            )
            retention_value = math.exp(-delta_days / max(1.0, float(row["strength"])))
            if retention_value >= retention_floor:
                continue
            await self._db.execute(
                """UPDATE dm_facts SET valid_until=?, updated_at=?,
                     version=version+1 WHERE id=?""",
                (iso(now), iso(now), row["id"]),
            )
            forgotten += 1
        return forgotten

    async def guild_stats(self, guild_id: str) -> GuildStats:
        total_row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM dm_facts WHERE guild_id=?",
            (guild_id,),
        )
        active_row = await self._db.query_one(
            """SELECT COUNT(*) AS n FROM dm_facts
               WHERE guild_id=? AND valid_until IS NULL AND superseded_by_id IS NULL""",
            (guild_id,),
        )
        tier_rows = await self._db.query(
            """SELECT tier, COUNT(*) AS n FROM dm_facts WHERE guild_id=?
               GROUP BY tier""",
            (guild_id,),
        )
        scope_rows = await self._db.query(
            """SELECT scope, COUNT(*) AS n FROM dm_facts WHERE guild_id=?
               GROUP BY scope""",
            (guild_id,),
        )
        users_row = await self._db.query_one(
            """SELECT COUNT(DISTINCT subject_id) AS n FROM dm_facts
               WHERE guild_id=? AND subject_id IS NOT NULL""",
            (guild_id,),
        )
        entity_row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM dm_entities WHERE guild_id=?",
            (guild_id,),
        )
        relation_row = await self._db.query_one(
            """SELECT COUNT(*) AS n FROM dm_relations
               WHERE guild_id=? AND valid_until IS NULL""",
            (guild_id,),
        )
        pending_row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM dm_messages WHERE guild_id=? AND status='pending'",
            (guild_id,),
        )
        claimed_row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM dm_messages WHERE guild_id=? AND status='claimed'",
            (guild_id,),
        )
        dead_row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM dm_messages WHERE guild_id=? AND status='dead'",
            (guild_id,),
        )

        def count_of(row: object) -> int:
            return int(row["n"]) if row is not None else 0  # type: ignore[index]

        return GuildStats(
            guild_id=guild_id,
            total_facts=count_of(total_row),
            active_facts=count_of(active_row),
            by_tier={MemoryTier(r["tier"]): int(r["n"]) for r in tier_rows},
            by_scope={FactScope(r["scope"]): int(r["n"]) for r in scope_rows},
            user_count=count_of(users_row),
            entity_count=count_of(entity_row),
            relation_count=count_of(relation_row),
            pending_messages=count_of(pending_row),
            in_flight_messages=count_of(claimed_row),
            dead_letters=count_of(dead_row),
        )


async def _noop() -> None:
    pass
