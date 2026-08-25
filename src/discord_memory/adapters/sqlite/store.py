"""SQLite backend: MemoryStore + IngestQueue + VectorIndex over one connection.

Single-file cohesion note: the store is split into focused mixins (facts/graph) but
composed here so consumers get one object implementing both ports. SQLite is the
zero-dependency default backend (PLAN.md D1); the Postgres adapter implements the
identical ports.
"""

from __future__ import annotations

from discord_memory.adapters.sqlite.connection import SqliteConnection
from discord_memory.adapters.sqlite.queue import SqliteIngestQueue
from discord_memory.adapters.sqlite.store_facts import FactsMixin
from discord_memory.adapters.sqlite.store_graph import IdentityGraphMixin
from discord_memory.adapters.sqlite.vectors import SqliteVectorIndex


class SqliteStore(FactsMixin, IdentityGraphMixin):
    """Implements :class:`~discord_memory.ports.MemoryStore` and hosts the queue."""

    def __init__(self, url: str = "sqlite:///discord_memory.db") -> None:
        self._db = SqliteConnection(url)
        self.queue = SqliteIngestQueue(self._db)
        self.vectors = SqliteVectorIndex(self._db)

    async def setup(self) -> None:
        await self._db.connect()
        await self._db.ensure_schema()
        await self.vectors.setup()

    async def close(self) -> None:
        await self._db.close()

    async def ping(self) -> bool:
        row = await self._db.query_one("SELECT 1 AS ok")
        return row is not None


__all__ = ["SqliteStore"]
