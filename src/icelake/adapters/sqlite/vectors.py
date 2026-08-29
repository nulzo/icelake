"""SQLite VectorIndex: brute-force cosine over guild-scoped candidates.

O(candidates) per query with candidates capped by ``candidate_cap`` — appropriate for
single-node deployments; swap the Postgres/pgvector adapter for ANN at scale. The port
keeps this swap invisible.
"""

from __future__ import annotations

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
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS ix_dm_vectors_guild ON dm_vectors (guild_id)",
        )

    async def upsert(self, items: tuple[VectorItem, ...]) -> None:
        for item in items:
            await self._db.execute(
                """INSERT INTO dm_vectors (fact_id, guild_id, subject_id, dim, embedding)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(fact_id) DO UPDATE SET
                     subject_id=excluded.subject_id,
                     dim=excluded.dim,
                     embedding=excluded.embedding""",
                (
                    item.id,
                    item.guild_id,
                    item.subject_id,
                    len(item.embedding),
                    _pack(item.embedding),
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
        # Recency-biased slice: an arbitrary unordered LIMIT slice silently
        # degraded recall quality once a guild exceeded the candidate cap.
        rows = await self._db.query(
            "SELECT fact_id, subject_id, dim, embedding FROM dm_vectors "
            "WHERE guild_id=? ORDER BY fact_id DESC LIMIT ?",
            (guild_id, candidate_cap),
        )
        allowed = set(subject_ids) if subject_ids is not None else None
        scored: list[VectorHit] = []
        for row in rows:
            candidate_subject = row["subject_id"]
            if server_only and candidate_subject is not None:
                continue
            if (
                not server_only
                and allowed is not None
                and (candidate_subject is not None and candidate_subject not in allowed)
            ):
                continue
            vector = _unpack(row["embedding"])
            score = cosine(embedding, vector)
            if score > 0.0:
                scored.append(VectorHit(id=row["fact_id"], score=score))
        scored.sort(key=lambda hit: -hit.score)
        return tuple(scored[:limit])

    async def count(self, guild_id: str) -> int:
        row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM dm_vectors WHERE guild_id=?",
            (guild_id,),
        )
        return int(row["n"]) if row is not None else 0


__all__ = ["SqliteVectorIndex", "cosine"]
