"""Optional sentence-transformers embedder (extra: ``local-embeddings``)."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from icelake.config import EmbeddingsConfig

if TYPE_CHECKING:
    pass


class _Model(Protocol):
    def encode(self, sentences: list[str], show_progress_bar: bool) -> list[list[float]]: ...


class LocalEmbedder:
    """Runs a local sentence-transformers model off-loop in a serialized worker calls."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        self._config = config
        self._model: _Model | None = None
        self._model_lock = threading.Lock()

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    def _load(self) -> _Model:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            name = (self._config.model or "").removeprefix("sentence-transformers/")
            model: Any = SentenceTransformer(name)
            self._model = model
        return self._model

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()

        def encode() -> tuple[tuple[float, ...], ...]:
            # The lock lives in the worker: cancelling an await must not allow
            # another thread to load or use the same model concurrently.
            with self._model_lock:
                model = self._load()
                vectors = model.encode(list(texts), False)
                return tuple(tuple(float(x) for x in vector) for vector in vectors)

        return await asyncio.to_thread(encode)


__all__ = ["LocalEmbedder"]
