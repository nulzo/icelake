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
from discord_memory.ports.llm import ChatRequest, LlmMessage


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
            assert body["response_format"] == {"type": "json_object"}
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
                json_mode=True,
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
