"""SQLite backend: MemoryStore + IngestQueue + VectorIndex over one connection.

Single-file cohesion note: the store is split into focused mixins (facts/graph) but
composed here so consumers get one object implementing both ports. SQLite is the
zero-dependency default backend (PLAN.md D1); the Postgres adapter implements the
identical ports.
"""

from __future__ import annotations

from icelake.adapters.sqlite.connection import SqliteConnection
from icelake.adapters.sqlite.llm_cache import SqliteLlmCache
from icelake.adapters.sqlite.queue import SqliteIngestQueue
from icelake.adapters.sqlite.store_facts import FactsMixin
from icelake.adapters.sqlite.store_graph import IdentityGraphMixin
from icelake.adapters.sqlite.vectors import SqliteVectorIndex
from icelake.models.facts import FactRecord
from icelake.models.graph import EntityRecord, RelationEdge


class SqliteStore(FactsMixin, IdentityGraphMixin):
    """Implements :class:`~icelake.ports.MemoryStore` and hosts the queue."""

    def __init__(self, url: str = "sqlite:///icelake.db") -> None:
        self._db = SqliteConnection(url)
        self.queue = SqliteIngestQueue(self._db)
        self.vectors = SqliteVectorIndex(self._db)
        self.llm_cache = SqliteLlmCache(self._db)

    def transaction(self):  # type: ignore[no-untyped-def]
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
