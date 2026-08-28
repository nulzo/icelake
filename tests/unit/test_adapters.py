"""Adapter unit tests: hashing embedder, meter budgets, LLM retry behavior."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from discord_memory.adapters.embedders import HashingEmbedder
from discord_memory.adapters.llm_openai_compat import OpenAICompatLLM
from discord_memory.adapters.meter import InMemoryMeter
from discord_memory.config import BudgetsConfig, EmbeddingsConfig, EmbeddingsProvider, LlmConfig
from discord_memory.errors import ConfigError
from discord_memory.models.admin import BudgetStep
from discord_memory.ports.clock import FixedClock
from discord_memory.ports.llm import ChatRequest, ChatResponse, LlmMessage


class TestHashingEmbedder:
    async def test_deterministic_and_normalized(self) -> None:
        embedder = HashingEmbedder(256)
        (first,) = await embedder.embed(("hello world of memory",))
        (second,) = await embedder.embed(("hello world of memory",))
        assert first == second
        norm = sum(v * v for v in first) ** 0.5
        assert norm == pytest.approx(1.0, abs=1e-3)

    async def test_similar_text_scores_higher_than_unrelated(self) -> None:
        from discord_memory.adapters.in_memory.vectors import cosine

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
    from discord_memory.adapters.embedders import build_embedder

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


class TestMeteredLLM:
    async def test_records_usage_per_purpose(self) -> None:
        from discord_memory.adapters.meter import MeteredLLM

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
        from discord_memory.adapters.llm_cache import CachedLLM
        from discord_memory.adapters.sqlite.connection import SqliteConnection
        from discord_memory.adapters.sqlite.llm_cache import SqliteLlmCache

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
        from discord_memory.adapters.llm_cache import CachedLLM

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


import json  # noqa: E402
