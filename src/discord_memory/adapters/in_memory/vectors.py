"""In-memory vector index: brute-force cosine over a scoped candidate cap."""

from __future__ import annotations

import asyncio

from discord_memory.ports.vectors import VectorHit, VectorItem, cosine


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
