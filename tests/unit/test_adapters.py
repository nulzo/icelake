"""Adapter unit tests: hashing embedder, meter budgets, LLM retry behavior."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from icelake.adapters.embedders import HashingEmbedder
from icelake.adapters.llm_openai_compat import OpenAICompatLLM
from icelake.adapters.llm_openrouter import OpenRouterLLM
from icelake.adapters.meter import InMemoryMeter
from icelake.config import BudgetsConfig, EmbeddingsConfig, EmbeddingsProvider, LlmConfig
from icelake.errors import ConfigError, LlmCapabilityError
from icelake.models.admin import BudgetStep
from icelake.ports.clock import FixedClock
from icelake.ports.llm import ChatRequest, ChatResponse, LlmMessage


class TestHashingEmbedder:
    async def test_deterministic_and_normalized(self) -> None:
        embedder = HashingEmbedder(256)
        (first,) = await embedder.embed(("hello world of memory",))
        (second,) = await embedder.embed(("hello world of memory",))
        assert first == second
        norm = sum(v * v for v in first) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-3)

    async def test_similar_text_scores_higher_than_unrelated(self) -> None:
        from icelake.adapters.in_memory.vectors import cosine

        embedder = HashingEmbedder(256)
        base, similar, unrelated = await embedder.embed(
            (
                "alice loves playing chess on weekends",
                "alice really loves playing chess every weekend",
                "the bus schedule changed for tuesday",
            )
        )
        assert cosine(base, similar) > cosine(base, unrelated)

    def test_minimum_dimensions_enforced(self) -> None:
        with pytest.raises(ConfigError):
            HashingEmbedder(8)

    async def test_batch_shapes(self) -> None:
        embedder = HashingEmbedder(64)
        vectors = await embedder.embed(())
        assert vectors == ()
        vectors = await embedder.embed(("one", "two", "three"))
        assert len(vectors) == 3 and len(vectors[0]) == 64


def test_local_embedder_extra_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
    from icelake.adapters.embedders import build_embedder

    with pytest.raises(ConfigError, match="local-embeddings"):
        build_embedder(EmbeddingsConfig(provider=EmbeddingsProvider.LOCAL))


class TestInMemoryMeter:
    NOW = datetime(2026, 8, 24, tzinfo=UTC)

    def test_budget_ladder(self) -> None:
        meter = InMemoryMeter(
            BudgetsConfig(guild_daily_prompt_tokens=100),
            FixedClock(self.NOW),
        )
        assert meter.check_budget("g") is BudgetStep.NONE
        meter.charge_guild("g", prompt_tokens=90)
        assert meter.check_budget("g") is BudgetStep.SKIP_RECONCILE
        meter.charge_guild("g", prompt_tokens=20)
        assert meter.check_budget("g") is BudgetStep.SKIP_EXTRACTION

    def test_snapshot_counts(self) -> None:
        meter = InMemoryMeter(BudgetsConfig(), FixedClock(self.NOW))
        meter.record_llm("extraction", prompt_tokens=10, completion_tokens=5, model="m")
        meter.increment("facts_added")
        snapshot = meter.snapshot()
        assert snapshot.calls["extraction"] == 1
        assert snapshot.prompt_tokens["extraction"] == 10

    def test_provider_reported_cost_wins_over_price_table(self) -> None:
        meter = InMemoryMeter(BudgetsConfig(), FixedClock(self.NOW))
        meter.record_llm(
            "extraction",
            prompt_tokens=1_000_000,
            completion_tokens=0,
            model="gemini-3.7-flash",
            cost_usd=0.5,
        )
        assert meter.snapshot().estimated_cost_usd["extraction"] == 0.5

    def test_actual_cost_recorded_for_unpriced_model(self) -> None:
        meter = InMemoryMeter(BudgetsConfig(), FixedClock(self.NOW))
        meter.record_llm(
            "extraction",
            prompt_tokens=10,
            completion_tokens=5,
            model="brand-new-model",
            cost_usd=0.000012,
        )
        assert meter.snapshot().estimated_cost_usd["extraction"] == 0.000012


class TestMeteredLLM:
    async def test_records_usage_per_purpose(self) -> None:
        from icelake.adapters.meter import MeteredLLM

        class _Fake:
            @property
            def model_name(self) -> str:
                return "google/gemini-3.7-flash"

            async def complete(self, request: ChatRequest) -> ChatResponse:
                return ChatResponse(text="{}", prompt_tokens=1000, completion_tokens=500)

        meter = InMemoryMeter(BudgetsConfig(), FixedClock(TestInMemoryMeter.NOW))
        llm = MeteredLLM(_Fake(), meter)
        await llm.complete(
            ChatRequest(
                messages=(LlmMessage(role="user", content="hi"),),
                purpose="extraction",
            )
        )
        snapshot = meter.snapshot()
        assert snapshot.calls["extraction"] == 1
        assert snapshot.prompt_tokens["extraction"] == 1000
        # 1000 in @ $0.375/M + 500 out @ $1.875/M, rounded to 6dp by the snapshot
        assert snapshot.estimated_cost_usd["extraction"] == pytest.approx(0.001312)


class TestCachedLLM:
    async def test_hit_replays_without_inner_call_and_zero_tokens(self) -> None:
        from icelake.adapters.llm_cache import CachedLLM
        from icelake.adapters.sqlite.connection import SqliteConnection
        from icelake.adapters.sqlite.llm_cache import SqliteLlmCache

        calls = {"n": 0}

        class _Fake:
            @property
            def model_name(self) -> str:
                return "m-1"

            async def complete(self, request: ChatRequest) -> ChatResponse:
                calls["n"] += 1
                return ChatResponse(text='{"ok": true}', prompt_tokens=10, completion_tokens=2)

        db = SqliteConnection("sqlite://:memory:")
        await db.connect()
        await db.ensure_schema()
        llm = CachedLLM(_Fake(), SqliteLlmCache(db))
        request = ChatRequest(messages=(LlmMessage(role="user", content="hi"),))

        first = await llm.complete(request)
        second = await llm.complete(request)

        assert first.text == '{"ok": true}' and second.text == '{"ok": true}'
        assert calls["n"] == 1
        assert second.prompt_tokens == 0 and second.completion_tokens == 0
        await db.close()

    async def test_distinct_requests_miss(self) -> None:
        from icelake.adapters.llm_cache import CachedLLM

        class _Fake:
            @property
            def model_name(self) -> str:
                return "m-1"

            async def complete(self, request: ChatRequest) -> ChatResponse:
                return ChatResponse(text=request.messages[-1].content)

        class _DictCache:
            def __init__(self) -> None:
                self.store: dict[str, ChatResponse] = {}

            async def get(self, key: str) -> ChatResponse | None:
                return self.store.get(key)

            async def put(self, key: str, response: ChatResponse) -> None:
                self.store[key] = response

        llm = CachedLLM(_Fake(), _DictCache())
        await llm.complete(ChatRequest(messages=(LlmMessage(role="user", content="a"),)))
        await llm.complete(ChatRequest(messages=(LlmMessage(role="user", content="b"),)))
        assert len(llm._cache.store) == 2


class TestOpenAICompatLLM:
    def _client(self, transport) -> OpenAICompatLLM:
        config = LlmConfig(base_url="https://llm.test/v1", api_key="k", model="m-1", max_retries=1)
        client = OpenAICompatLLM(config)
        client._client = httpx.AsyncClient(transport=transport)  # type: ignore[attr-defined]
        return client

    async def test_successful_completion_parses_usage(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["model"] == "m-1"
            assert "response_format" not in body
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                    "model": "m-1",
                },
            )

        client = self._client(httpx.MockTransport(handler))
        response = await client.complete(
            ChatRequest(
                messages=(LlmMessage(role="user", content="hi"),),
            )
        )
        assert response.text == '{"ok": true}'
        assert response.prompt_tokens == 12
        await client.aclose()

    async def test_max_completion_tokens_key_replaces_max_tokens(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}], "usage": {}},
            )

        config = LlmConfig(
            base_url="https://llm.test/v1",
            api_key="k",
            model="m-1",
            max_retries=0,
            max_tokens_key="max_completion_tokens",
            params={"max_tokens": 999},
        )
        client = OpenAICompatLLM(config)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await client.complete(
            ChatRequest(messages=(LlmMessage(role="user", content="hi"),), max_tokens=256)
        )
        assert captured.get("max_completion_tokens") == 256
        assert "max_tokens" not in captured
        await client.aclose()

    async def test_retries_then_succeeds_on_500(self) -> None:
        state = {"calls": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            state["calls"] += 1
            if state["calls"] < 2:
                return httpx.Response(500)
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {},
                },
            )

        client = self._client(httpx.MockTransport(handler))
        response = await client.complete(
            ChatRequest(messages=(LlmMessage(role="user", content="hi"),))
        )
        assert response.text == "ok"
        assert state["calls"] == 2
        await client.aclose()

    async def test_raises_after_exhausted_retries(self) -> None:
        calls = {"n": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(503)

        client = self._client(httpx.MockTransport(handler))
        with pytest.raises(httpx.HTTPStatusError):
            await client.complete(ChatRequest(messages=(LlmMessage(role="user", content="hi"),)))
        assert calls["n"] == 2  # initial attempt + 1 retry
        await client.aclose()


class TestOpenRouterLLM:
    def _client(self, transport, **overrides) -> OpenRouterLLM:
        config = LlmConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key="k",
            model="m-1",
            max_retries=0,
            **overrides,
        )
        client = OpenRouterLLM(config)
        client._client = httpx.AsyncClient(transport=transport)  # type: ignore[attr-defined]
        return client

    async def test_requests_and_parses_actual_cost(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["usage"] == {"include": True}
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.0000123},
                },
            )

        client = self._client(httpx.MockTransport(handler))
        response = await client.complete(
            ChatRequest(messages=(LlmMessage(role="user", content="hi"),))
        )
        assert response.cost_usd == 0.0000123
        await client.aclose()

    async def test_reasoning_effort_sent_only_when_configured(self) -> None:
        bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        client = self._client(httpx.MockTransport(handler), reasoning_effort="low")
        await client.complete(ChatRequest(messages=(LlmMessage(role="user", content="hi"),)))
        assert bodies[0]["reasoning"] == {"effort": "low"}
        await client.aclose()

        plain = self._client(httpx.MockTransport(handler))
        await plain.complete(ChatRequest(messages=(LlmMessage(role="user", content="hi"),)))
        assert "reasoning" not in bodies[1]
        await plain.aclose()

    async def test_schema_requests_require_parameters(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["provider"] == {"require_parameters": True}
            assert body["response_format"]["type"] == "json_schema"
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        client = self._client(httpx.MockTransport(handler))
        await client.complete(
            ChatRequest(
                messages=(LlmMessage(role="user", content="hi"),),
                response_schema={"type": "object", "properties": {}},
            )
        )
        await client.aclose()

    async def test_404_raises_capability_error_with_guidance(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404, json={"error": {"code": 404, "message": "No endpoints found"}}
            )

        client = self._client(httpx.MockTransport(handler))
        with pytest.raises(LlmCapabilityError, match="temperature=None"):
            await client.complete(
                ChatRequest(
                    messages=(LlmMessage(role="user", content="hi"),),
                    response_schema={"type": "object", "properties": {}},
                )
            )
        await client.aclose()

    async def test_404_without_schema_raises_capability_error(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": {"message": "No endpoints found"}})

        client = self._client(httpx.MockTransport(handler))
        with pytest.raises(LlmCapabilityError, match="Check model id"):
            await client.complete(ChatRequest(messages=(LlmMessage(role="user", content="hi"),)))
        await client.aclose()

    async def test_json_object_mode_sends_no_constraint(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["response_format"] == {"type": "json_object"}
            assert "provider" not in body
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        client = self._client(httpx.MockTransport(handler), structured_outputs="json_object")
        await client.complete(
            ChatRequest(
                messages=(LlmMessage(role="user", content="hi"),),
                response_schema={"type": "object", "properties": {}},
            )
        )
        await client.aclose()

    async def test_params_passthrough_merges_provider_prefs(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            # User provider prefs survive; require_parameters merges in.
            assert body["provider"] == {"order": ["openai"], "require_parameters": True}
            assert body["seed"] == 7
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        client = self._client(
            httpx.MockTransport(handler), params={"provider": {"order": ["openai"]}, "seed": 7}
        )
        await client.complete(
            ChatRequest(
                messages=(LlmMessage(role="user", content="hi"),),
                response_schema={"type": "object", "properties": {}},
            )
        )
        await client.aclose()

    async def test_temperature_omitted_when_configured_none(self) -> None:
        bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        client = self._client(httpx.MockTransport(handler), temperature=None)
        await client.complete(ChatRequest(messages=(LlmMessage(role="user", content="hi"),)))
        assert "temperature" not in bodies[0]
        await client.aclose()

        # Per-request override wins over the config default.
        client2 = self._client(httpx.MockTransport(handler))
        await client2.complete(
            ChatRequest(messages=(LlmMessage(role="user", content="hi"),), temperature=0.7)
        )
        assert bodies[1]["temperature"] == 0.7
        await client2.aclose()

    async def test_base_client_sends_no_provider_extras(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert "provider" not in body
            assert "usage" not in body
            assert "reasoning" not in body
            return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

        config = LlmConfig(
            base_url="https://llm.test/v1",
            api_key="k",
            model="m-1",
            max_retries=0,
            reasoning_effort="low",
        )
        client = OpenAICompatLLM(config)
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[attr-defined]
        await client.complete(ChatRequest(messages=(LlmMessage(role="user", content="hi"),)))
        await client.aclose()


import json  # noqa: E402
