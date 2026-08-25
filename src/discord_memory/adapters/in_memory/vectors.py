"""In-memory vector index: brute-force cosine over a scoped candidate cap."""

from __future__ import annotations

import asyncio
import math

from discord_memory.ports.vectors import VectorHit, VectorItem


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity clipped to [0, 1] (embeddings are non-negative here)."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class InMemoryVectorIndex:
    """Dict-backed brute-force index. O(N) per query within the candidate cap."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._items: dict[str, VectorItem] = {}

    async def setup(self) -> None:
        pass

    async def upsert(self, items: tuple[VectorItem, ...]) -> None:
        async with self._lock:
            for item in items:
                self._items[item.id] = item

    async def delete(self, ids: tuple[str, ...]) -> int:
        removed = 0
        async with self._lock:
            for item_id in ids:
                if self._items.pop(item_id, None) is not None:
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
        candidates = [item for item in self._items.values() if item.guild_id == guild_id]
        if server_only:
            candidates = [c for c in candidates if c.subject_id is None]
        elif subject_ids is not None:
            allowed = set(subject_ids)
            candidates = [c for c in candidates if c.subject_id is None or c.subject_id in allowed]
        candidates = candidates[:candidate_cap]
        scored = [
            VectorHit(id=candidate.id, score=cosine(embedding, candidate.embedding))
            for candidate in candidates
        ]
        scored.sort(key=lambda hit: -hit.score)
        return tuple(scored[:limit])

    async def count(self, guild_id: str) -> int:
        return sum(1 for item in self._items.values() if item.guild_id == guild_id)
