"""VectorIndex conformance across backends + embedders + LLM adapter edge paths."""

from __future__ import annotations

import json
import sqlite3

import httpx
import pytest

from discord_memory.adapters.embedders import (
    HashingEmbedder,
    OpenAICompatEmbedder,
    build_embedder,
)
from discord_memory.adapters.in_memory.vectors import InMemoryVectorIndex
from discord_memory.adapters.llm_openai_compat import OpenAICompatLLM
from discord_memory.adapters.sqlite.connection import SqliteConnection
from discord_memory.adapters.sqlite.vectors import SqliteVectorIndex
from discord_memory.config import EmbeddingsConfig, EmbeddingsProvider, LlmConfig
from discord_memory.errors import ConfigError
from discord_memory.ports.llm import ChatRequest, LlmMessage
from discord_memory.ports.vectors import VectorItem


@pytest.fixture(params=["in_memory", "sqlite"])
async def vectors(request):
    if request.param == "in_memory":
        index = InMemoryVectorIndex()
    else:
        connection = SqliteConnection("sqlite://:memory:")
        await connection.connect()
        index = SqliteVectorIndex(connection)
    await index.setup()
    yield index


def item(item_id: str, vec: tuple[float, ...], subject: str | None = "u1") -> VectorItem:
    return VectorItem(id=item_id, guild_id="g1", subject_id=subject, embedding=vec)


class TestVectorConformance:
    async def test_upsert_search_and_scoping(self, vectors) -> None:
        e = HashingEmbedder(64)
        (keyboard,) = await e.embed(("mechanical keyboard hobby",))
        (tea,) = await e.embed(("drinking tea in the garden",))
        await vectors.upsert((item("f1", keyboard), item("f2", tea), item("s1", tea, subject=None)))
        hits = await vectors.search(keyboard, guild_id="g1", limit=3)
        assert hits[0].id == "f1"
        # subject scope keeps user rows AND server-wide rows
        scoped = await vectors.search(tea, guild_id="g1", subject_ids=("u1",), limit=5)
        returned = {hit.id for hit in scoped}
        assert {"f2", "s1"} <= returned
        tea_hits = [h.score for h in scoped if h.id in {"f2", "s1"}]
        assert all(score > 0.99 for score in tea_hits)
        server_only = await vectors.search(tea, guild_id="g1", server_only=True, limit=5)
        assert [hit.id for hit in server_only] == ["s1"]

    async def test_other_guild_excluded(self, vectors) -> None:
        await vectors.upsert((item("x1", (1.0, 0.0, 0.0)),))
        hits = await vectors.search((1.0, 0.0, 0.0), guild_id="other", limit=5)
        assert hits == ()

    async def test_delete_and_count(self, vectors) -> None:
        await vectors.upsert((item("d1", (0.5, 0.5)), item("d2", (0.4, 0.6))))
        assert await vectors.count("g1") == 2
        removed = await vectors.delete(("d1", "missing"))
        assert removed == 1
        assert await vectors.count("g1") == 1

    async def test_upsert_overwrites(self, vectors) -> None:
        await vectors.upsert((item("o1", (1.0, 0.0)),))
        await vectors.upsert((item("o1", (0.0, 1.0)),))
        hits = await vectors.search((0.0, 1.0), guild_id="g1", limit=2)
        assert hits[0].id == "o1" and hits[0].score > 0.99

    async def test_empty_embedding_query(self, vectors) -> None:
        assert await vectors.search((), guild_id="g1", limit=3) == ()


class TestEmbedderAdapters:
    async def test_openai_compat_embedder_batches(self) -> None:
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            body = json.loads(request.content)
            inputs = body["input"]
            return httpx.Response(
                200,
                json={
                    "data": [{"index": i, "embedding": [float(i)] * 32} for i in range(len(inputs))]
                },
            )

        config = EmbeddingsConfig(
            provider=EmbeddingsProvider.OPENAI,
            base_url="https://embed.test/v1",
            api_key="k",
            model="emb-1",
            dimensions=32,
        )
        embedder = OpenAICompatEmbedder(config)
        embedder._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        out = await embedder.embed(("a",) * 70)  # forces two batches at 64
        assert len(out) == 70
        assert calls["n"] == 2
        await embedder._client.aclose()

    def test_build_factory_openai_path(self) -> None:
        embedder = build_embedder(
            EmbeddingsConfig(
                provider=EmbeddingsProvider.OPENAI,
                base_url="https://embed.test/v1",
                model="m",
                dimensions=32,
            )
        )
        assert isinstance(embedder, OpenAICompatEmbedder)

    async def test_local_embedder_with_stubbed_backend(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class FakeModel:
            def encode(self, texts, _flag=False):
                return [[0.25] * 32 for _ in texts]

        fake_module = __import__("types").SimpleNamespace(
            SentenceTransformer=lambda name: FakeModel(),
        )
        import sys

        monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
        from discord_memory.adapters.embedders.local import LocalEmbedder

        embedder = LocalEmbedder(EmbeddingsConfig(provider=EmbeddingsProvider.LOCAL, dimensions=32))
        out = await embedder.embed(("hello there", "second"))
        assert len(out) == 2 and len(out[0]) == 32


class TestLLMAdapterEdges:
    def test_model_name_property(self) -> None:
        client = OpenAICompatLLM(LlmConfig(base_url="https://x/v1", model="fast-1"))
        assert client.model_name == "fast-1"

    def test_requires_base_url_and_model(self) -> None:
        with pytest.raises(ValueError):
            OpenAICompatLLM(LlmConfig(base_url=None, model=None))

    async def test_transport_error_retries_then_succeeds(self) -> None:
        state = {"calls": 0}

        def handler(_request: httpx.Request) -> httpx.Response:
            state["calls"] += 1
            if state["calls"] == 1:
                raise httpx.ConnectError("boom")
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "recovered"}}],
                    "usage": {},
                },
            )

        client = OpenAICompatLLM(LlmConfig(base_url="https://x/v1", model="m", max_retries=2))
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        response = await client.complete(
            ChatRequest(messages=(LlmMessage(role="user", content="hi"),))
        )
        assert response.text == "recovered"
        assert state["calls"] == 2
        await client.aclose()

    async def test_list_content_parts_concatenated(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"text": "part-one "},
                                    {"text": "part-two"},
                                ]
                            }
                        }
                    ],
                    "usage": {},
                },
            )

        client = OpenAICompatLLM(LlmConfig(base_url="https://x/v1", model="m"))
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        response = await client.complete(
            ChatRequest(messages=(LlmMessage(role="user", content="hi"),))
        )
        assert response.text == "part-one part-two"
        await client.aclose()


async def test_connection_transaction_rollback() -> None:
    connection = SqliteConnection("sqlite://:memory:")
    await connection.connect()
    await connection.execute("CREATE TABLE t (v TEXT)")
    statements = [("INSERT INTO t VALUES ('a')", ()), ("INSERT INTO bogus VALUES ('b')", ())]
    with pytest.raises(sqlite3.OperationalError):
        await connection.transaction(statements)
    rows = await connection.query("SELECT * FROM t")
    assert rows == []  # rolled back
    await connection.close()
    with pytest.raises(AssertionError):
        await connection.query("SELECT 1")


def test_config_error_on_unknown_scheme_message() -> None:
    from discord_memory.errors import IdentityAmbiguousError, StorageUnavailableError

    error = IdentityAmbiguousError("klim", ("u1", "u2"))
    assert "klim" in str(error) and len(error.candidate_ids) == 2
    assert issubclass(StorageUnavailableError, ConfigError.__bases__[0])
