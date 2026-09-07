"""Reranker port + closed-ID citation contract tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from icelake.adapters.rerankers import OpenAICompatReranker, build_reranker
from icelake.config import RerankerConfig, RerankerProvider, RetrievalConfig
from icelake.errors import ConfigError
from icelake.models.facts import FactCategory, FactRecord, SourceRef, SourceRole
from icelake.models.retrieval import Citation, RecallQuery, ScoredFact
from icelake.retrieval.injection import InjectionBuilder, _primary_citation

GUILD = "500000000000000001"
ALICE = "100000000000000001"
NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)


def _fact(text: str, fact_id: str = "fct_1") -> FactRecord:
    return FactRecord(
        id=fact_id,
        guild_id=GUILD,
        subject_id=ALICE,
        text=text,
        category=FactCategory.INTERESTS,
        created_at=NOW,
        updated_at=NOW,
        observed_at=NOW,
        valid_from=NOW,
        citations=(
            SourceRef(
                message_id="m1",
                channel_id="c1",
                guild_id=GUILD,
                author_id=ALICE,
                author_name="alice",
                content_snippet=text[:80],
                message_url="",
                role=SourceRole.PRIMARY,
            ),
        ),
    )


class _StubReranker:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.calls: list[tuple[str, list[str]]] = []

    async def score(self, query: str, documents):
        self.calls.append((query, list(documents)))
        return tuple(self._scores)


class _ExplodingReranker:
    async def score(self, query: str, documents):
        raise RuntimeError("reranker exploded")


class _MismatchReranker:
    async def score(self, query: str, documents):
        return ()


class TestRerankerAdapters:
    def test_none_provider_returns_none(self) -> None:
        assert build_reranker(RerankerConfig(provider=RerankerProvider.NONE)) is None

    def test_openai_requires_base_url_and_model(self) -> None:
        with pytest.raises(ConfigError):
            RerankerConfig(provider=RerankerProvider.OPENAI, model="x")

    def test_local_requires_sentence_transformers(self, monkeypatch) -> None:
        import importlib.util

        monkeypatch.setattr(importlib.util, "find_spec", lambda _name: None)
        with pytest.raises(ConfigError, match="local-embeddings"):
            build_reranker(RerankerConfig(provider=RerankerProvider.LOCAL))

    def test_from_spec_none(self) -> None:
        assert RerankerConfig.from_spec("none").provider is RerankerProvider.NONE

    def test_from_spec_local(self) -> None:
        config = RerankerConfig.from_spec("local")
        assert config.provider is RerankerProvider.LOCAL
        assert config.model == "sentence-transformers/bge-reranker-v2-m3"

    def test_from_spec_openrouter(self) -> None:
        config = RerankerConfig.from_spec(
            "openai://key@openrouter.ai/api/v1?model=qwen/qwen3-reranker-8b"
        )
        assert config.provider is RerankerProvider.OPENAI
        assert config.base_url == "https://openrouter.ai/api/v1"
        assert config.api_key == "key"
        assert config.model == "qwen/qwen3-reranker-8b"

    def test_from_spec_openrouter_default_model(self) -> None:
        config = RerankerConfig.from_spec("openai://key@openrouter.ai/api/v1")
        assert config.model == "qwen/qwen3-reranker-8b"

    def test_from_spec_unknown_raises(self) -> None:
        with pytest.raises(ConfigError, match="unknown reranker spec"):
            RerankerConfig.from_spec("bogus")

    def test_retrieval_config_coerces_reranker_url(self) -> None:
        config = RetrievalConfig(reranker="openai://key@openrouter.ai/api/v1")
        assert config.reranker.provider is RerankerProvider.OPENAI
        assert config.reranker.model == "qwen/qwen3-reranker-8b"


class TestOpenRouterReranker:
    def _client(self, transport: httpx.MockTransport) -> OpenAICompatReranker:
        reranker = OpenAICompatReranker(
            RerankerConfig(
                provider=RerankerProvider.OPENAI,
                base_url="https://openrouter.ai/api/v1",
                api_key="key",
                model="qwen/qwen3-reranker-8b",
            )
        )
        reranker._client = httpx.AsyncClient(transport=transport)
        return reranker

    async def test_scores_aligned_to_input_order(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["auth"] = request.headers.get("authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"index": 1, "relevance_score": 0.1},
                        {"index": 0, "relevance_score": 0.9},
                    ]
                },
            )

        reranker = self._client(httpx.MockTransport(handler))
        scores = await reranker.score("table tennis", ["plays table tennis", "likes chess"])
        assert scores == (0.9, 0.1)
        assert captured["url"] == "https://openrouter.ai/api/v1/rerank"
        assert captured["auth"] == "Bearer key"
        assert captured["body"]["model"] == "qwen/qwen3-reranker-8b"
        assert captured["body"]["query"] == "table tennis"
        assert captured["body"]["documents"] == ["plays table tennis", "likes chess"]
        assert captured["body"]["top_n"] == 2
        await reranker.aclose()

    async def test_empty_documents_short_circuit(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            raise AssertionError("should not call the network")

        reranker = self._client(httpx.MockTransport(handler))
        assert await reranker.score("q", []) == ()

    async def test_http_error_raises_for_recall_to_degrade(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        reranker = self._client(httpx.MockTransport(handler))
        with pytest.raises(httpx.HTTPStatusError):
            await reranker.score("q", ["doc"])

    async def test_missing_results_returns_zeros(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        reranker = self._client(httpx.MockTransport(handler))
        assert await reranker.score("q", ["a", "b"]) == (0.0, 0.0)


class TestRerankerDegrade:
    async def test_recall_degrades_when_reranker_fails(self, make_client) -> None:
        client, _ = make_client(llm=False)
        client._reranker = _ExplodingReranker()
        await client.start()
        await client.facts.remember(guild_id=GUILD, subject_id=ALICE, text="plays table tennis")
        result = await client.recall(
            RecallQuery(guild_id=GUILD, text="table tennis", subject_ids=(ALICE,))
        )
        assert result.facts
        assert result.facts[0].rerank_score is None
        await client.close()

    async def test_recall_degrades_on_score_arity_mismatch(self, make_client) -> None:
        client, _ = make_client(llm=False)
        client._reranker = _MismatchReranker()
        await client.start()
        await client.facts.remember(guild_id=GUILD, subject_id=ALICE, text="plays table tennis")
        result = await client.recall(
            RecallQuery(guild_id=GUILD, text="table tennis", subject_ids=(ALICE,))
        )
        assert result.facts
        assert result.facts[0].rerank_score is None
        await client.close()

    async def test_recall_uses_reranker_scores(self, make_client) -> None:
        client, _ = make_client(llm=False)
        reranker = _StubReranker([0.9, 0.1])
        client._reranker = reranker
        await client.start()
        await client.facts.remember(guild_id=GUILD, subject_id=ALICE, text="plays table tennis")
        await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="likes playing chess on weekends"
        )
        result = await client.recall(
            RecallQuery(guild_id=GUILD, text="table tennis", subject_ids=(ALICE,))
        )
        assert result.facts
        assert result.facts[0].rerank_score == 0.9
        assert reranker.calls
        await client.close()

    async def test_recall_no_reranker_keeps_hybrid_order(self, make_client) -> None:
        client, _ = make_client(llm=False)
        client._reranker = None
        await client.start()
        await client.facts.remember(guild_id=GUILD, subject_id=ALICE, text="plays table tennis")
        result = await client.recall(
            RecallQuery(guild_id=GUILD, text="table tennis", subject_ids=(ALICE,))
        )
        assert result.facts
        assert result.facts[0].rerank_score is None
        await client.close()

    async def test_threshold_drops_low_rerank_scores(self, make_client) -> None:
        from tests.conftest import make_config

        config = make_config(retrieval={"reranker_threshold": 0.5, "reranker_pool_size": 8})
        client, _ = make_client(llm=False, config=config)
        client._reranker = _StubReranker([0.9, 0.1])
        await client.start()
        await client.facts.remember(guild_id=GUILD, subject_id=ALICE, text="plays table tennis")
        await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="likes playing chess on weekends"
        )
        result = await client.recall(
            RecallQuery(guild_id=GUILD, text="table tennis", subject_ids=(ALICE,))
        )
        assert result.facts
        assert all(
            f.rerank_score is None or f.rerank_score >= 0.5 for f in result.facts if f.rerank_score
        )
        await client.close()

    async def test_empty_query_skips_reranker(self, make_client) -> None:
        client, _ = make_client(llm=False)
        reranker = _StubReranker([0.9])
        client._reranker = reranker
        await client.start()
        await client.facts.remember(guild_id=GUILD, subject_id=ALICE, text="plays table tennis")
        result = await client.recall(RecallQuery(guild_id=GUILD, subject_ids=(ALICE,)))
        assert result.facts
        assert reranker.calls == []
        assert result.facts[0].rerank_score is None
        await client.close()


class TestCitationEnrichment:
    def test_citation_model_accepts_enriched_fields(self) -> None:
        citation = Citation(
            ref="mem:1",
            fact_id="fct_a",
            url="https://discord.com/channels/g/c/m1",
            message_id="m1",
            channel_id="c1",
            score=0.87,
            rerank_score=0.92,
        )
        assert citation.message_id == "m1"
        assert citation.channel_id == "c1"
        assert citation.score == 0.87
        assert citation.rerank_score == 0.92

    def test_primary_citation_enriched_from_scored_fact(self) -> None:
        record = _fact("plays table tennis")
        scored = ScoredFact(fact=record, score=0.9, rerank_score=0.85)
        citation = _primary_citation(scored, GUILD, 1)
        assert citation is not None
        assert citation.message_id == "m1"
        assert citation.channel_id == "c1"
        assert citation.score == 0.9
        assert citation.rerank_score == 0.85
        assert citation.subject_name == "alice"
        assert citation.url.endswith("/c1/m1")

    def test_primary_citation_prefers_display_name(self) -> None:
        record = _fact("plays table tennis")
        scored = ScoredFact(fact=record, score=0.9)
        citation = _primary_citation(scored, GUILD, 1, display_name="bob")
        assert citation is not None
        assert citation.subject_name == "bob"

    def test_primary_citation_none_when_no_source(self) -> None:
        record = _fact("plays table tennis").model_copy(update={"citations": ()})
        scored = ScoredFact(fact=record, score=0.9)
        assert _primary_citation(scored, GUILD, 1) is None

    def test_injection_emits_rich_citation_objects(self) -> None:
        builder = InjectionBuilder()
        scored = ScoredFact(fact=_fact("plays table tennis"), score=0.77, rerank_score=0.88)
        _block, citations, _trimmed = builder.build(
            asker_id=ALICE,
            facts_by_section={"asker": (scored,)},
            summaries={},
            token_budget=5_000,
            guild_id=GUILD,
            display_names={"asker": "alice"},
        )
        assert len(citations) == 1
        assert citations[0].ref == "mem:1"
        assert citations[0].message_id == "m1"
        assert citations[0].channel_id == "c1"
        assert citations[0].score == 0.77
        assert citations[0].rerank_score == 0.88
        assert citations[0].subject_name == "alice"
        assert "[mem:1]" not in _block  # plain facts; tags never reach the model
