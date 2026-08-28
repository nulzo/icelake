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


def _strict_schema(schema: dict[str, object]) -> dict[str, object]:
    """Transform a JSON Schema into OpenAI-strict-compatible form.

    Strict mode requires: every property listed in ``required`` and
    ``additionalProperties: false`` on every object; optional fields become
    ``anyOf: [type, null]``. Non-object schemas pass through unchanged.
    """

    def transform(node: dict[str, object]) -> dict[str, object]:
        if node.get("type") != "object" and "properties" not in node:
            return node
        properties = node.get("properties")
        if not isinstance(properties, dict):
            return node
        raw_required = node.get("required", [])
        required: list[str] = (
            [str(item) for item in raw_required] if isinstance(raw_required, list) else []
        )
        new_props: dict[str, dict[str, object]] = {}
        for name, raw_prop in properties.items():
            prop = dict(raw_prop) if isinstance(raw_prop, dict) else {"type": "string"}
            if isinstance(prop.get("anyOf"), list):
                # already optional-shaped (e.g. anyOf [T, null])
                any_of = prop["anyOf"]
                has_null = any(isinstance(v, dict) and v.get("type") == "null" for v in any_of)
                if not has_null:
                    any_of = [*any_of, {"type": "null"}]
                prop["anyOf"] = any_of
                required.append(name)
            elif not prop.get("nullable"):
                # plain type: strict requires presence; keep as-is
                required.append(name)
            else:
                required.append(name)
            if isinstance(prop, dict) and "properties" in prop:
                prop = transform(prop)
            new_props[name] = prop
        out = {
            **node,
            "properties": {
                k: transform(v)
                if isinstance(v, dict) and (v.get("type") == "object" or "properties" in v)
                else v
                for k, v in new_props.items()
            },
            "required": sorted(set(required)),
            "additionalProperties": False,
        }
        return out

    return transform(schema)


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
        json_object_fallback = False
        while True:
            attempt += 1
            try:
                return await self._complete_once(request, json_object_fallback=json_object_fallback)
            except httpx.HTTPStatusError as exc:
                # Endpoint rejects native json_schema -> degrade once to plain
                # json_object rather than failing the batch. Support is
                # per-endpoint (OpenRouter structured-outputs docs, 2026).
                if (
                    exc.response.status_code == 400
                    and request.response_schema is not None
                    and not json_object_fallback
                ):
                    json_object_fallback = True
                    continue
                retries_exhausted = attempt > self._config.max_retries
                if exc.response.status_code not in _RETRY_STATUS or retries_exhausted:
                    raise
                await asyncio.sleep(min(2.0 * attempt, 5.0))
            except (httpx.TransportError, TimeoutError):
                if attempt > self._config.max_retries:
                    raise
                await asyncio.sleep(min(1.0 * attempt, 3.0))

    async def _complete_once(
        self, request: ChatRequest, *, json_object_fallback: bool = False
    ) -> ChatResponse:
        model = self._config.model or ""
        body: dict[str, object] = {
            "model": model,
            "messages": [message.model_dump() for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_schema is not None:
            if json_object_fallback:
                body["response_format"] = {"type": "json_object"}
            else:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{request.purpose}_output",
                        # Strict enforcement on capable providers; others treat
                        # it as a strong hint (OpenRouter docs, 2026).
                        "strict": True,
                        "schema": _strict_schema(request.response_schema),
                    },
                }
                if "openrouter.ai" in (self._config.base_url or ""):
                    # Route only to endpoints that actually enforce json_schema.
                    body["provider"] = {"require_parameters": True}
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
