"""Embedding adapters: deterministic hashing (default) and OpenAI-compatible API.

The hashing embedder is the zero-dependency default: signed feature-hashing over word
and char n-grams, L2-normalized. Deterministic, fast, free — good lexical-ish
semantics for small deployments and perfectly reproducible in tests. Swap to a real
model via config without touching any other code.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx

from discord_memory.config import EmbeddingsConfig, EmbeddingsProvider
from discord_memory.errors import ConfigError

if TYPE_CHECKING:
    from discord_memory.ports.llm import Embedder


class HashingEmbedder:
    """Signed feature-hashing embedder; stable across processes."""

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ConfigError("hashing embedder needs at least 32 dimensions")
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._embed_one(text) for text in texts)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vector = [0.0] * self._dimensions
        lowered = " ".join(text.lower().split())
        tokens = lowered.split()
        for token in tokens:
            self._add_feature(vector, token, 1.0)
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i + 1]}"
            self._add_feature(vector, bigram, 0.7)
        for i in range(max(0, len(lowered) - 3)):
            self._add_feature(vector, "c:" + lowered[i : i + 4], 0.25)
        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(round(v / norm, 8) for v in vector)

    def _add_feature(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.md5(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % self._dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight


class OpenAICompatEmbedder:
    """OpenAI ``/embeddings``-compatible adapter (OpenRouter/OpenAI/vLLM/Ollama)."""

    def __init__(self, config: EmbeddingsConfig) -> None:
        if not config.base_url or not config.model:
            raise ConfigError("openai embeddings require base_url and model")
        self._config = config
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def dimensions(self) -> int:
        return self._config.dimensions

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        out: list[tuple[float, ...]] = []
        batch_size = 64
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            response = await self._client.post(
                f"{self._config.base_url}/embeddings",
                headers=self._headers(),
                json={"model": self._config.model, "input": list(chunk)},
            )
            response.raise_for_status()
            payload = response.json()
            data = sorted(payload["data"], key=lambda item: item["index"])
            out.extend(tuple(float(x) for x in item["embedding"]) for item in data)
        return tuple(out)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers


def build_embedder(config: EmbeddingsConfig) -> Embedder:
    """Factory honoring the configured provider with a graceful local fallback."""
    if config.provider is EmbeddingsProvider.HASHING:
        return HashingEmbedder(config.dimensions)
    if config.provider is EmbeddingsProvider.OPENAI:
        return OpenAICompatEmbedder(config)
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is None:
        raise ConfigError(
            "local embeddings require the 'local-embeddings' extra "
            "(pip install discord-memory[local-embeddings])",
        )
    from discord_memory.adapters.embedders.local import LocalEmbedder

    return LocalEmbedder(config)


__all__ = ["HashingEmbedder", "OpenAICompatEmbedder", "build_embedder"]
