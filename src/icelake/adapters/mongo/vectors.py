"""Mongo VectorIndex: guild-scoped brute-force cosine over stored embeddings."""

from __future__ import annotations

import math
import struct
from typing import Any

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

    async def upsert(self, items: tuple[VectorItem, ...]) -> None:
        for item in items:
            await self.col.update_one(
                {"_id": item.id},
                {
                    "$set": {
                        "guild_id": item.guild_id,
                        "subject_id": item.subject_id,
                        "dim": len(item.embedding),
                        "embedding": _pack(item.embedding),
                    }
                },
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
        cursor = self.col.find(query).limit(candidate_cap)
        scored: list[VectorHit] = []
        async for doc in cursor:
            score = _cosine(embedding, _unpack(doc["embedding"]))
            if score > 0.0:
                scored.append(VectorHit(id=doc["_id"], score=score))
        scored.sort(key=lambda hit: -hit.score)
        return tuple(scored[:limit])

    async def count(self, guild_id: str) -> int:
        count = await self.col.count_documents({"guild_id": guild_id})
        return int(count)


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
