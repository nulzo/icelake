"""Embedding cache: content-hash LRU over any inner Embedder.

Cuts recall-path latency and API cost when the same or similar texts are
embedded repeatedly (query-embedding reuse, reconcile collision checks,
consolidation sanity checks). Thread-safe; bounded by ``max_entries``.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Sequence

from discord_memory.ports.llm import Embedder


class CachedEmbedder:
    """LRU-cached wrapper implementing the Embedder protocol."""

    def __init__(self, inner: Embedder, *, max_entries: int = 50_000) -> None:
        self._inner = inner
        self._max_entries = max_entries
        self._cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        keys = [self._key(text) for text in texts]
        with self._lock:
            cached: dict[int, tuple[float, ...]] = {}
            missing_indices: list[int] = []
            for index, key in enumerate(keys):
                if key in self._cache:
                    cached[index] = self._cache[key]
                    self.hits += 1
                else:
                    missing_indices.append(index)
                    self.misses += 1
            # LRU touch for hits
            for key in (keys[i] for i in cached):
                self._cache.move_to_end(key)

        if missing_indices:
            uncached_texts = [texts[i] for i in missing_indices]
            new_vectors = await self._inner.embed(uncached_texts)
            for local_idx, vector in zip(missing_indices, new_vectors, strict=True):
                cached[local_idx] = vector
                with self._lock:
                    self._cache[keys[local_idx]] = vector
                    self._evict_if_needed()

        return tuple(cached[i] for i in range(len(texts)))

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()


__all__ = ["CachedEmbedder"]
