"""CachedLLM: content-addressed response cache decorator (opt-in, see
``LlmConfig.cache_responses``). Identical requests replay for free — cache hits
report zero tokens so metering never double-counts spend. A dev/CI aid; leave
disabled in production unless prompts are highly repetitive.
"""

from __future__ import annotations

import hashlib
import json

from icelake.ports.llm import ChatLLM, ChatRequest, ChatResponse, LlmCache


def cache_key(model: str, request: ChatRequest) -> str:
    payload = json.dumps(
        {
            "model": model,
            "purpose": request.purpose,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [(m.role, m.content) for m in request.messages],
            "schema": request.response_schema,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedLLM:
    """ChatLLM decorator that replays identical completions from an LlmCache."""

    def __init__(self, inner: ChatLLM, cache: LlmCache) -> None:
        self._inner = inner
        self._cache = cache

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def complete(self, request: ChatRequest) -> ChatResponse:
        key = cache_key(self._inner.model_name, request)
        hit = await self._cache.get(key)
        if hit is not None:
            return hit
        response = await self._inner.complete(request)
        await self._cache.put(key, response)
        return response


__all__ = ["CachedLLM", "cache_key"]
