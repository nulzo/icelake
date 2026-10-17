"""SQLite backend: MemoryStore + IngestQueue + VectorIndex over one connection.

Single-file cohesion note: the store is split into focused mixins (facts/graph) but
composed here so consumers get one object implementing both ports. SQLite is the
zero-dependency default backend.
"""

from __future__ import annotations

import json
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path

from icelake.adapters.sqlite.connection import SqliteConnection
from icelake.adapters.sqlite.llm_cache import SqliteLlmCache
from icelake.adapters.sqlite.queue import SqliteIngestQueue
from icelake.adapters.sqlite.store_facts import FactsMixin
from icelake.adapters.sqlite.store_graph import IdentityGraphMixin
from icelake.adapters.sqlite.vectors import SqliteVectorIndex
from icelake.models.backup import BackupManifest
from icelake.models.facts import FactRecord
from icelake.models.graph import EntityRecord, RelationEdge
from icelake.models.recovery import ProjectionJob


class SqliteStore(FactsMixin, IdentityGraphMixin):
    """Implements :class:`~icelake.ports.MemoryStore` and hosts the queue."""

    def __init__(self, url: str = "sqlite:///icelake.db") -> None:
        self._db = SqliteConnection(url)
        self.queue = SqliteIngestQueue(self._db)
        self.vectors = SqliteVectorIndex(self._db)
        self.llm_cache = SqliteLlmCache(self._db)

    def transaction(self) -> AbstractAsyncContextManager[SqliteConnection]:
        """Async CM: BEGIN IMMEDIATE .. COMMIT/ROLLBACK around fact commits."""
        return self._db.transaction_scope()

    async def setup(self) -> None:
        await self._db.connect()
        await self._db.ensure_schema()
        await self.vectors.setup()

    async def close(self) -> None:
        await self._db.close()

    async def ping(self) -> bool:
        row = await self._db.query_one("SELECT 1 AS ok")
        return row is not None

    async def pending_projections(
        self, limit: int, *, now: datetime | None = None
    ) -> tuple[ProjectionJob, ...]:
        rows = await self._db.query(
            "SELECT * FROM dm_projection_jobs "
            "WHERE next_attempt_at IS NULL OR next_attempt_at <= ? "
            "ORDER BY rowid LIMIT ?",
            ((now or datetime.now(UTC)).isoformat(), limit),
        )
        return tuple(
            ProjectionJob.model_validate(
                {
                    **dict(row),
                    "embedding": json.loads(row["embedding"]) if row["embedding"] else None,
                }
            )
            for row in rows
        )

    async def enqueue_projection(self, job: ProjectionJob) -> None:
        await self._db.execute(
            "INSERT INTO dm_projection_jobs(guild_id, fact_id, version, embedding) "
            "VALUES(?, ?, ?, ?) "
            "ON CONFLICT(fact_id) DO UPDATE SET version=excluded.version, "
            "embedding=excluded.embedding, next_attempt_at=NULL "
            "WHERE dm_projection_jobs.version <= excluded.version",
            (
                job.guild_id,
                job.fact_id,
                job.version,
                json.dumps(job.embedding) if job.embedding is not None else None,
            ),
        )

    async def defer_projection(self, job: ProjectionJob, next_attempt_at: datetime) -> None:
        await self._db.execute(
            "UPDATE dm_projection_jobs SET next_attempt_at=? "
            "WHERE guild_id=? AND fact_id=? AND version=?",
            (next_attempt_at.isoformat(), job.guild_id, job.fact_id, job.version),
        )

    async def complete_projection(self, job: ProjectionJob) -> None:
        await self._db.execute(
            "DELETE FROM dm_projection_jobs WHERE guild_id=? AND fact_id=? AND version=?",
            (job.guild_id, job.fact_id, job.version),
        )

    async def backup(self, directory: Path) -> BackupManifest:
        """Snapshot this database, including its built-in queue and vector index."""
        from icelake.adapters.sqlite.backup import create_backup

        return await create_backup(self._db, directory)

    async def import_guild(
        self,
        facts: tuple[FactRecord, ...],
        entities: tuple[EntityRecord, ...],
        relations: tuple[RelationEdge, ...],
    ) -> int:
        """Bulk restore from a MemoryExport."""
        for record in facts:
            await self.insert_fact(record)
        for entity in entities:
            await self.upsert_entity(
                entity.guild_id,
                entity.slug,
                entity.name,
                entity.kind,
                aliases=entity.aliases,
            )
        for edge in relations:
            await self.upsert_relation(edge)
        return len(facts)


__all__ = ["SqliteStore"]
