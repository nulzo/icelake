"""Reranker adapters: none (default), local cross-encoder, OpenRouter ``/rerank``.

L2 rerank is optional and degrades to first-stage hybrid scores when absent.
Hosted traffic uses OpenRouter's ``POST /api/v1/rerank`` (Cohere-shaped results).
Local runs a sentence-transformers CrossEncoder off-loop. Keep the hot path
zero-LLM: a reranker is a classifier, not a chat completion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

import httpx

from icelake.config import RerankerConfig, RerankerProvider
from icelake.errors import ConfigError
from icelake.ports.llm import Reranker


class _CrossEncoder(Protocol):
    def predict(self, pairs: list[list[str]], show_progress_bar: bool) -> list[float]: ...


class LocalReranker:
    """Local sentence-transformers CrossEncoder (Qwen3-Reranker / bge-reranker)."""

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._model: _CrossEncoder | None = None

    def _load(self) -> _CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            name = (self._config.model or "").removeprefix("sentence-transformers/")
            model: Any = CrossEncoder(name)
            self._model = model
        return self._model

    async def score(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        if not documents:
            return ()
        model = await asyncio.to_thread(self._load)
        pairs = [[query, document] for document in documents]
        scores = await asyncio.to_thread(model.predict, pairs, False)
        return tuple(float(score) for score in scores)


class OpenAICompatReranker:
    """OpenRouter ``/rerank`` adapter (also Jina/Cohere-shaped hosts)."""

    def __init__(self, config: RerankerConfig) -> None:
        if not config.base_url or not config.model:
            raise ConfigError("openai reranker requires base_url and model")
        self._config = config
        self._base_url = config.base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds)

    async def score(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        if not documents:
            return ()
        response = await self._client.post(
            f"{self._base_url}/rerank",
            headers=self._headers(),
            json={
                "model": self._config.model,
                "query": query,
                "documents": list(documents),
                "top_n": len(documents),
            },
        )
        response.raise_for_status()
        return _scores_from_results(response.json(), len(documents))

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    async def aclose(self) -> None:
        await self._client.aclose()


def _scores_from_results(payload: object, n: int) -> tuple[float, ...]:
    """Align OpenRouter ``results[].{index, relevance_score}`` back to input order."""
    scores = [0.0] * n
    if not isinstance(payload, dict):
        return tuple(scores)
    results = payload.get("results")
    if not isinstance(results, list):
        return tuple(scores)
    for item in results:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        if not isinstance(index, int) or not (0 <= index < n):
            continue
        scores[index] = float(item.get("relevance_score", 0.0))
    return tuple(scores)


def build_reranker(config: RerankerConfig) -> Reranker | None:
    """Factory honoring the configured provider; ``none`` disables L2."""
    if config.provider is RerankerProvider.NONE:
        return None
    if config.provider is RerankerProvider.OPENAI:
        return OpenAICompatReranker(config)
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is None:
        raise ConfigError(
            "local reranker requires the 'local-embeddings' extra "
            "(pip install icelake[local-embeddings])",
        )
    return LocalReranker(config)


__all__ = ["LocalReranker", "OpenAICompatReranker", "build_reranker"]
