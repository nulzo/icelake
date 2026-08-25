"""OpenAI chat-completions-compatible LLM adapter (OpenRouter/OpenAI/Ollama/vLLM).

Async httpx client; retries transient failures with exponential backoff; reports token
usage so the Meter can enforce budgets.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from discord_memory.config import LlmConfig
from discord_memory.ports.llm import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatLLM:
    """Thin, provider-agnostic completion client."""

    def __init__(self, config: LlmConfig) -> None:
        if not config.base_url or not config.model:
            raise ValueError("OpenAICompatLLM requires base_url and model")

        self._config = config
        self._client = httpx.AsyncClient(timeout=config.timeout_seconds)

    @property
    def model_name(self) -> str:
        return self._config.model or ""

    async def complete(self, request: ChatRequest) -> ChatResponse:
        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._complete_once(request)
            except httpx.HTTPStatusError as exc:
                retries_exhausted = attempt > self._config.max_retries
                if exc.response.status_code not in _RETRY_STATUS or retries_exhausted:
                    raise
                await asyncio.sleep(min(2.0 * attempt, 5.0))
            except (httpx.TransportError, TimeoutError):
                if attempt > self._config.max_retries:
                    raise
                await asyncio.sleep(min(1.0 * attempt, 3.0))

    async def _complete_once(self, request: ChatRequest) -> ChatResponse:
        model = self._config.model or ""
        body: dict[str, object] = {
            "model": model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        response = await self._client.post(
            f"{self._config.base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=request.timeout_seconds or self._config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        text = ""
        choices = payload.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(part.get("text", "") for part in content)
        return ChatResponse(
            text=text,
            model=str(payload.get("model", model)),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["OpenAICompatLLM"]
