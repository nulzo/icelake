"""Native structured-output (strict json_schema) wire-format tests.

Mirrors the 2026 provider landscape: OpenAI/OpenRouter enforce json_schema
natively on capable endpoints. Capability mismatches (400/404/422) raise
``LlmCapabilityError`` loudly — the adapter never silently degrades.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel

from icelake.adapters.llm_openai_compat import (
    OpenAICompatLLM,
    _strict_schema,
)
from icelake.adapters.llm_openrouter import OpenRouterLLM
from icelake.config import LlmConfig
from icelake.errors import LlmCapabilityError
from icelake.ports.llm import ChatRequest, LlmMessage


def _client(transport, *, base_url: str = "https://llm.test/v1") -> OpenAICompatLLM:
    client = OpenAICompatLLM(LlmConfig(base_url=base_url, api_key="k", model="m-1", max_retries=1))
    client._client = httpx.AsyncClient(transport=transport)  # type: ignore[attr-defined]
    return client


class TestStrictSchemaTransform:
    def test_optionals_become_anyof_null_and_all_required(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "subject_token": {"type": "string"},
                "speaker_token": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
            },
        }
        out = _strict_schema(schema)
        assert set(out["required"]) == {"subject_token", "speaker_token"}
        assert out["additionalProperties"] is False
        speaker = out["properties"]["speaker_token"]
        assert {"type": "null"} in speaker["anyOf"]

    def test_nested_objects_get_additional_properties_false(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            },
        }
        out = _strict_schema(schema)
        inner = out["properties"]["operations"]
        assert inner["additionalProperties"] is False

    def test_defs_are_transformed_for_strict_validators(self) -> None:
        # Pydantic nests models under $defs + $ref; OpenAI's strict validator
        # rejects the whole schema if any $def keeps a partial required list.
        class Entity(BaseModel):
            name: str
            kind: str = "other"

        class Output(BaseModel):
            entities: list[Entity]

        out = _strict_schema(Output.model_json_schema())
        entity = out["$defs"]["Entity"]
        assert entity["additionalProperties"] is False
        assert set(entity["required"]) == {"name", "kind"}

    def test_ref_with_default_is_inlined(self) -> None:
        # Pydantic emits {"$ref": ..., "default": ...} for enum fields with
        # defaults; OpenAI's strict validator rejects $ref siblings and allOf
        # in property position, so the definition is inlined instead.
        from enum import StrEnum

        class Action(StrEnum):
            NONE = "none"
            QUERY = "query"

        class Output(BaseModel):
            action: Action = Action.NONE
            other: Action  # bare $ref stays a $ref

        out = _strict_schema(Output.model_json_schema())
        action = out["properties"]["action"]
        assert action["enum"] == ["none", "query"]
        assert action["type"] == "string"
        assert "default" not in action and "$ref" not in action
        assert out["properties"]["other"] == {"$ref": "#/$defs/Action"}
        assert set(out["required"]) == {"action", "other"}


class TestWireFormat:
    def test_strict_json_schema_sent_when_schema_present(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "{}"}}],
                    "usage": {},
                },
            )

        client = _client(httpx.MockTransport(handler))
        import asyncio

        asyncio.run(
            client.complete(
                ChatRequest(
                    messages=(LlmMessage(role="user", content="hi"),),
                    response_schema={"type": "object", "properties": {}},
                    purpose="extraction",
                )
            )
        )
        fmt = captured.get("response_format")
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True
        assert "provider" not in captured

    def test_openrouter_requests_require_parameters(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "{}"}}], "usage": {}},
            )

        client = OpenRouterLLM(
            LlmConfig(
                base_url="https://openrouter.ai/api/v1", api_key="k", model="m-1", max_retries=1
            )
        )
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[attr-defined]
        import asyncio

        asyncio.run(
            client.complete(
                ChatRequest(
                    messages=(LlmMessage(role="user", content="hi"),),
                    response_schema={"type": "object", "properties": {}},
                    purpose="extraction",
                )
            )
        )
        assert captured["provider"] == {"require_parameters": True}

    async def test_provider_400_on_schema_raises_capability_error(self) -> None:
        # No silent json_object degrade: capability mismatches fail loudly so
        # integrators fix the declared capabilities instead of losing quality.
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"message": "response_format json_schema unsupported"}},
            )

        client = _client(httpx.MockTransport(handler))
        with pytest.raises(LlmCapabilityError, match="structured_outputs='json_object'"):
            await client.complete(
                ChatRequest(
                    messages=(LlmMessage(role="user", content="hi"),),
                    response_schema={"type": "object", "properties": {}},
                    purpose="extraction",
                )
            )
        await client.aclose()
