"""OpenAI chat-completions-compatible LLM adapter (OpenRouter/OpenAI/Ollama/vLLM).

Async httpx client; retries transient failures with exponential backoff; reports token
usage so the Meter can enforce budgets. Capability mismatches (HTTP 400/404/422)
raise ``LlmCapabilityError`` immediately — the library never silently degrades
output quality; configuration declares what the model's endpoints support.
Provider-specific behavior lives in subclasses (e.g.
``llm_openrouter.OpenRouterLLM``) via the ``_apply_provider_extras`` hook.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from icelake.config import LlmConfig
from icelake.errors import LlmCapabilityError
from icelake.ports.llm import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
_CAPABILITY_STATUS = {400, 404, 422}


def _strict_schema(schema: dict[str, object]) -> dict[str, object]:
    """Transform a JSON Schema into OpenAI-strict-compatible form.

    Strict mode requires: every property listed in ``required`` and
    ``additionalProperties: false`` on every object; optional fields become
    ``anyOf: [type, null]``. Pydantic nests models under ``$defs`` + ``$ref``,
    so the transform must cover both the root and every definition — strict
    validators (OpenAI) reject the whole schema otherwise. Non-object schemas
    pass through unchanged.
    """

    defs = schema.get("$defs")
    definitions: dict[str, object] = defs if isinstance(defs, dict) else {}

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
            ref = prop.get("$ref")
            if isinstance(ref, str) and len(prop) > 1:
                # Strict validators reject $ref with sibling keywords (pydantic
                # emits {"$ref": ..., "default": ...} for enum fields with
                # defaults) and forbid allOf here — inline the definition and
                # drop the default (meaningless: the field is required).
                target = definitions.get(ref.removeprefix("#/$defs/"))
                description = prop.get("description")
                prop = (
                    {k: v for k, v in dict(target).items() if k != "default"}
                    if isinstance(target, dict)
                    else {"$ref": ref}
                )
                if description is not None:
                    prop["description"] = description
                required.append(name)
            elif isinstance(prop.get("anyOf"), list):
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

    root = transform(schema)
    if definitions:
        root["$defs"] = {
            key: transform(value) if isinstance(value, dict) else value
            for key, value in definitions.items()
        }
    return root


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
                status = exc.response.status_code
                if status in _CAPABILITY_STATUS:
                    raise _capability_error(self._config, exc, request) from exc
                retries_exhausted = attempt > self._config.max_retries
                if status not in _RETRY_STATUS or retries_exhausted:
                    raise
                await asyncio.sleep(min(2.0 * attempt, 5.0))
            except (httpx.TransportError, TimeoutError):
                if attempt > self._config.max_retries:
                    raise
                await asyncio.sleep(min(1.0 * attempt, 3.0))

    def _apply_provider_extras(self, body: dict[str, object], request: ChatRequest) -> None:
        """Subclass hook for provider-specific request fields (base: none)."""

    async def _complete_once(self, request: ChatRequest) -> ChatResponse:
        model = self._config.model or ""
        # User-declared passthrough first; library-managed keys win collisions.
        body: dict[str, object] = dict(self._config.params)
        body["model"] = model
        body["messages"] = [message.model_dump() for message in request.messages]
        temperature = (
            request.temperature if request.temperature is not None else self._config.temperature
        )
        if temperature is not None:
            body["temperature"] = temperature
        body.pop("max_tokens", None)
        body.pop("max_completion_tokens", None)
        body[self._config.max_tokens_key] = request.max_tokens
        if request.response_schema is not None:
            if self._config.structured_outputs == "strict":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"{request.purpose}_output",
                        "strict": True,
                        "schema": _strict_schema(request.response_schema),
                    },
                }
            else:
                body["response_format"] = {"type": "json_object"}
        self._apply_provider_extras(body, request)
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
            cost_usd=(float(cost) if (cost := usage.get("cost")) is not None else None),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["OpenAICompatLLM", "build_chat_llm"]


def _capability_error(
    config: LlmConfig, exc: httpx.HTTPStatusError, request: ChatRequest
) -> LlmCapabilityError:
    """Build an actionable capability-mismatch error from a 400/404/422."""
    detail = exc.response.text.strip().replace("\n", " ")[:300]
    guidance = (
        f"model {config.model!r} rejected the request "
        f"(HTTP {exc.response.status_code}): {detail}. "
        "Declare the endpoint's actual capabilities in LlmConfig instead of "
        "retrying: temperature=None if it rejects sampling params, "
        "structured_outputs='json_object' if it cannot enforce json_schema."
    )
    if request.response_schema is None:
        guidance = (
            f"model {config.model!r} rejected the request "
            f"(HTTP {exc.response.status_code}): {detail}. "
            "Check model id and LlmConfig.params against the provider's supported parameters."
        )
    return LlmCapabilityError(guidance)


def build_chat_llm(config: LlmConfig) -> OpenAICompatLLM:
    """Pick the adapter for an endpoint: provider quirks live in subclasses."""
    if "openrouter.ai" in (config.base_url or ""):
        from icelake.adapters.llm_openrouter import OpenRouterLLM

        return OpenRouterLLM(config)
    return OpenAICompatLLM(config)
