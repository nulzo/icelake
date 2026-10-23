"""Mongo VectorIndex: guild-scoped brute-force cosine over stored embeddings."""

from __future__ import annotations

import asyncio
import heapq
import math
import struct
from typing import Any

import numpy as np

from icelake.ports.vectors import VectorHit, VectorItem

MongoAsyncDatabase = "AsyncDatabase"


def _pack(vector: tuple[float, ...]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _unpack(blob: bytes) -> tuple[float, ...]:
    count = len(blob) // 4
    return struct.unpack(f"{count}f", blob)


class MongoVectorIndex:
    """Embedding store + candidate-cap cosine scan sharing the Mongo database.

    Atlas deployments can swap in ``$vectorSearch`` behind the same port without
    consumer changes.
    """

    def __init__(self, db: Any) -> None:
        self.col = db["dm_vectors"]

    async def setup(self) -> None:
        await self.col.create_index([("guild_id", 1)])
        # Compound so the subject scope filter is index-served, not a scan.
        await self.col.create_index([("guild_id", 1), ("subject_id", 1)])

    async def upsert(self, items: tuple[VectorItem, ...]) -> None:
        for item in items:
            # A pipeline update keeps insert-or-ignore atomic without duplicate-key
            # failures when an older worker races a newer projection.
            payload = {
                "_id": item.id,
                "guild_id": item.guild_id,
                "subject_id": item.subject_id,
                "dim": len(item.embedding),
                "embedding": _pack(item.embedding),
                "version": item.version,
            }
            await self.col.update_one(
                {"_id": item.id},
                [
                    {
                        "$replaceWith": {
                            "$cond": [
                                {"$lte": [{"$ifNull": ["$version", 0]}, item.version]},
                                {"$literal": payload},
                                "$$ROOT",
                            ]
                        }
                    }
                ],
                upsert=True,
            )

    async def delete(self, ids: tuple[str, ...]) -> int:
        result = await self.col.delete_many({"_id": {"$in": list(ids)}})
        return int(result.deleted_count)

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
        query: dict[str, Any] = {"guild_id": guild_id}
        if server_only:
            query["subject_id"] = None
        elif subject_ids is not None:
            query["$or"] = [
                {"subject_id": {"$in": list(subject_ids)}},
                {"subject_id": None},
            ]
        # candidate_cap bounds a working batch, not which ages are searchable.
        # Scan the scoped collection in constant memory; native scoring runs off
        # the event loop. RecallService imposes a channel deadline.
        batch_size = max(1, min(candidate_cap, 512))
        cursor = self.col.find(query).batch_size(batch_size)
        best: list[VectorHit] = []
        batch: list[dict[str, Any]] = []
        try:
            async for doc in cursor:
                batch.append(doc)
                if len(batch) >= batch_size:
                    scored = await asyncio.to_thread(_score_batch, embedding, batch, limit)
                    best = heapq.nlargest(limit, (*best, *scored), key=lambda hit: hit.score)
                    batch = []
            if batch:
                scored = await asyncio.to_thread(_score_batch, embedding, batch, limit)
                best = heapq.nlargest(limit, (*best, *scored), key=lambda hit: hit.score)
        finally:
            await cursor.close()
        return tuple(best)

    async def count(self, guild_id: str) -> int:
        count = await self.col.count_documents({"guild_id": guild_id})
        return int(count)


def _score_batch(
    embedding: tuple[float, ...],
    docs: list[dict[str, Any]],
    limit: int,
) -> list[VectorHit]:
    aligned = [doc for doc in docs if len(doc["embedding"]) == len(embedding) * 4]
    if not aligned:
        return []
    query = np.asarray(embedding, dtype=np.float32)
    matrix = np.stack([np.frombuffer(doc["embedding"], dtype=np.float32) for doc in aligned])
    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    scores = np.divide(
        matrix @ query, denominator, out=np.zeros(len(aligned)), where=denominator > 0
    )
    indexes = heapq.nlargest(limit, range(len(aligned)), key=lambda i: float(scores[i]))
    return [
        VectorHit(id=aligned[i]["_id"], score=min(1.0, float(scores[i])))
        for i in indexes
        if math.isfinite(float(scores[i])) and scores[i] > 0
    ]


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    similarity = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    return max(0.0, min(1.0, similarity))
