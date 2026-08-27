"""Native structured-output (strict json_schema) + fallback ladder tests.

Mirrors the 2026 provider landscape: OpenAI/OpenRouter enforce json_schema
natively on capable models; others reject it with 400 or treat it as a hint.
The adapter must degrade gracefully without losing the batch.
"""

from __future__ import annotations

import json

import httpx

from discord_memory._json import coerce_extraction_payload, parse_json_object
from discord_memory.adapters.llm_openai_compat import (
    OpenAICompatLLM,
    _strict_schema,
)
from discord_memory.config import LlmConfig
from discord_memory.models.operations import ExtractionOutput
from discord_memory.ports.llm import ChatRequest, LlmMessage


def _client(transport) -> OpenAICompatLLM:
    client = OpenAICompatLLM(
        LlmConfig(base_url="https://llm.test/v1", api_key="k", model="m-1", max_retries=1)
    )
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

    def test_defs_objects_are_strict_and_defaults_stripped(self) -> None:
        from discord_memory.models.operations import ReconcileOutput

        out = _strict_schema(ReconcileOutput.model_json_schema())
        decision = out["$defs"]["ReconcileDecision"]
        assert decision["additionalProperties"] is False
        assert set(decision["required"]) == set(decision["properties"])
        assert "default" not in out["properties"]["decisions"]
        assert "default" not in decision["properties"]["confidence"]


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
                    json_mode=True,
                    response_schema={"type": "object", "properties": {}},
                    purpose="extraction",
                )
            )
        )
        fmt = captured.get("response_format")
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["strict"] is True

    async def test_provider_400_on_schema_degrades_to_json_object(self) -> None:
        state = {"calls": 0}
        formats_seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            state["calls"] += 1
            body = json.loads(request.content)
            fmt = body.get("response_format", {}).get("type", "none")
            formats_seen.append(fmt)
            if fmt == "json_schema":
                return httpx.Response(
                    400,
                    json={
                        "error": {"message": "response_format json_schema unsupported"},
                    },
                )
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"ok": 1}'}}],
                    "usage": {},
                },
            )

        client = _client(httpx.MockTransport(handler))
        response = await client.complete(
            ChatRequest(
                messages=(LlmMessage(role="user", content="hi"),),
                json_mode=True,
                response_schema={"type": "object", "properties": {}},
                purpose="extraction",
            )
        )
        assert response.text == '{"ok": 1}'
        assert formats_seen == ["json_schema", "json_object"]
        await client.aclose()


class TestCoercionSafetyNet:
    def test_graphiti_triple_shape_normalized(self) -> None:
        payload = {
            "facts": [
                {
                    "id": "0",
                    "subject": {"name": "Alice"},
                    "predicate": "likes",
                    "object": {"name": "tea"},
                    "text": "",
                }
            ],
        }
        coerced = coerce_extraction_payload(payload)
        operations = coerced["operations"]
        assert isinstance(operations, list)
        first = operations[0]
        assert isinstance(first, dict)
        assert "Alice" in str(first.get("text"))

    def test_canonical_payload_passthrough_unchanged(self) -> None:
        payload = {"operations": [{"subject_token": "p0", "text": "x"}]}
        assert coerce_extraction_payload(payload) == payload

    def test_parse_then_coerce_roundtrip(self) -> None:
        raw = 'noise {"memories": [{"subject_token": "p0", "text": "y"}]} tail'

        payload = coerce_extraction_payload(parse_json_object(raw))
        output = ExtractionOutput.model_validate(payload)
        assert output.operations[0].text == "y"
