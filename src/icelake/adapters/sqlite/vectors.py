"""SQLite VectorIndex: brute-force cosine over guild-scoped candidates.

Scoped exact search in bounded working batches, without an age cutoff. Scoring
runs off the event loop; the recall service applies a deadline. Deployments exceeding
that measured budget can replace the VectorIndex port with an ANN adapter.
"""

from __future__ import annotations

import asyncio
import heapq
import sqlite3
import struct

from icelake.adapters.sqlite.connection import SqliteConnection
from icelake.ports.vectors import VectorHit, VectorItem, cosine


def _pack(vector: tuple[float, ...]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> tuple[float, ...]:
    count = len(blob) // 4
    return struct.unpack(f"{count}f", blob)


class SqliteVectorIndex:
    """Embedding store + brute-force search sharing the SQLite connection."""

    def __init__(self, connection: SqliteConnection) -> None:
        self._db = connection

    async def setup(self) -> None:
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS dm_vectors (
                 fact_id TEXT PRIMARY KEY,
                 guild_id TEXT NOT NULL,
                 subject_id TEXT,
                 dim INTEGER NOT NULL,
                 embedding BLOB NOT NULL
               )""",
        )
        columns = await self._db.query("PRAGMA table_info(dm_vectors)")
        if not any(row["name"] == "version" for row in columns):
            await self._db.execute(
                "ALTER TABLE dm_vectors ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
            )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_dm_vectors_guild ON dm_vectors (guild_id)",
        )

    async def upsert(self, items: tuple[VectorItem, ...]) -> None:
        for item in items:
            await self._db.execute(
                """INSERT INTO dm_vectors (fact_id, guild_id, subject_id, dim, embedding, version)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(fact_id) DO UPDATE SET
                     subject_id=excluded.subject_id,
                     dim=excluded.dim,
                     embedding=excluded.embedding, version=excluded.version
                   WHERE dm_vectors.version <= excluded.version""",
                (
                    item.id,
                    item.guild_id,
                    item.subject_id,
                    len(item.embedding),
                    _pack(item.embedding),
                    item.version,
                ),
            )

    async def delete(self, ids: tuple[str, ...]) -> int:
        removed = 0
        for fact_id in ids:
            row = await self._db.query_one(
                "SELECT fact_id FROM dm_vectors WHERE fact_id=?",
                (fact_id,),
            )
            if row is not None:
                await self._db.execute("DELETE FROM dm_vectors WHERE fact_id=?", (fact_id,))
                removed += 1
        return removed

    async def search(
        self,
        embedding: tuple[float, ...],
        *,
        guild_id: str,
        subject_ids: tuple[str, ...] | None = None,
        server_only: bool = False,
        limit: int = 20,
        candidate_cap: int = 500,
    ) -> tuple[VectorHit, ...]:
        if not embedding:
            return ()
        # Scope filter goes in the WHERE clause, before the cap. Capping first
        # and filtering after starves quiet members on busy servers.
        where = ["guild_id = ?"]
        params: list[object] = [guild_id]
        if server_only:
            where.append("subject_id IS NULL")
        elif subject_ids is not None:
            placeholders = ",".join("?" for _ in subject_ids)
            where.append(f"(subject_id IN ({placeholders}) OR subject_id IS NULL)")
            params.extend(subject_ids)
        batch_size = max(1, min(candidate_cap, 512))
        after = ""
        best: list[VectorHit] = []
        while True:
            rows = await self._db.query(
                "SELECT fact_id, embedding FROM dm_vectors "
                f"WHERE {' AND '.join(where)} AND fact_id > ? ORDER BY fact_id LIMIT ?",
                (*params, after, batch_size),
            )
            if not rows:
                break
            scored = await asyncio.to_thread(_score_batch, embedding, rows, limit)
            best = heapq.nlargest(limit, (*best, *scored), key=lambda hit: hit.score)
            after = rows[-1]["fact_id"]
        return tuple(best)

    async def count(self, guild_id: str) -> int:
        row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM dm_vectors WHERE guild_id=?",
            (guild_id,),
        )
        return int(row["n"]) if row is not None else 0


def _score_batch(
    embedding: tuple[float, ...],
    rows: list[sqlite3.Row],
    limit: int,
) -> list[VectorHit]:
    return heapq.nlargest(
        limit,
        (
            VectorHit(id=row["fact_id"], score=score)
            for row in rows
            if (score := cosine(embedding, _unpack(row["embedding"]))) > 0
        ),
        key=lambda hit: hit.score,
    )


__all__ = ["SqliteVectorIndex", "cosine"]
