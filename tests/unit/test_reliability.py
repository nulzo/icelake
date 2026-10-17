"""Failure-injection regressions for the memory reliability contract."""

from __future__ import annotations

import asyncio
import threading
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from icelake.adapters.citation_verifier import LlmCitationVerifier
from icelake.adapters.in_memory.store import InMemoryStore
from icelake.adapters.sqlite.connection import SqliteConnection
from icelake.config import LlmConfig
from icelake.ingest.context_builder import build_extraction_context
from icelake.models.facts import Attribution, FactRecord, SourceRef
from icelake.models.retrieval import ChannelName, RecallQuery
from icelake.ports.citations import CitationCandidate
from icelake.ports.clock import FixedClock
from icelake.retrieval.injection import InjectionBuilder, estimate_tokens
from icelake.retrieval.service import RecallService
from tests.conftest import ScriptedLLM

NOW = datetime(2026, 10, 17, tzinfo=UTC)


async def test_sqlite_cancellation_rolls_back_and_connection_remains_usable() -> None:
    db = SqliteConnection("sqlite://:memory:")
    await db.connect()
    await db.execute("CREATE TABLE example (value INTEGER)")
    entered = asyncio.Event()

    async def transaction() -> None:
        async with db.transaction_scope():
            await db.execute("INSERT INTO example VALUES (1)")
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(transaction())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    async with db.transaction_scope():
        await db.execute("INSERT INTO example VALUES (2)")
    assert [row["value"] for row in await db.query("SELECT * FROM example")] == [2]
    await db.close()


async def test_sqlite_cancelled_thread_retains_lock_until_statement_settles() -> None:
    db = SqliteConnection("sqlite://:memory:")
    await db.connect()
    entered, release = threading.Event(), threading.Event()

    def slow() -> int:
        entered.set()
        assert release.wait(5)
        return 1

    assert db._conn is not None
    db._conn.create_function("slow", 0, slow)
    task = asyncio.create_task(db.query("SELECT slow()"))
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    second = asyncio.create_task(db.query("SELECT 2"))
    await asyncio.sleep(0.01)
    assert not task.done() and not second.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await second)[0][0] == 2
    await db.close()


async def test_extraction_context_does_not_send_blocked_speaker_to_provider() -> None:
    store = InMemoryStore()
    await store.insert_fact(
        FactRecord(
            id="private",
            guild_id="g",
            subject_id="alice",
            text="private disclosure",
            attribution=Attribution(speaker_id="bob"),
        )
    )
    await store.set_opt_out("g", "bob", True)
    context = await build_extraction_context(
        store=store,
        vectors=None,
        guild_id="g",
        subject_id="alice",
        batch_text="new message",
        batch_embedding=None,
    )
    assert "private disclosure" not in context


async def test_export_never_resurrects_fenced_third_party_evidence(store) -> None:
    fact = FactRecord(
        id="f",
        guild_id="g",
        subject_id="alice",
        text="reported private information",
        citations=(SourceRef(message_id="m", channel_id="c", guild_id="g", author_id="bob"),),
    )
    await store.insert_fact(fact)
    await store.bump_governance_generation("g", "bob")
    facts, _, _ = await store.export_guild("g")
    assert facts == ()


async def test_purge_physically_removes_third_party_fact(store) -> None:
    await store.insert_fact(
        FactRecord(
            id="f",
            guild_id="g",
            subject_id="alice",
            text="private",
            attribution=Attribution(speaker_id="bob"),
        )
    )
    report = await store.purge_user_data("g", "bob", dry_run=False)
    assert report.facts_removed == 1
    # A new generation can admit new observations, but the old body is gone.
    assert await store.get_fact("g", "f") is None


async def test_failed_manual_embedding_retains_complete_fact_and_recovery_job(make_client) -> None:
    client, _ = make_client(llm=False)
    await client.start()
    original = client.facts._embedder

    class Unavailable:
        dimensions = 4

        async def embed(self, texts):
            raise RuntimeError("offline")

    client.facts._embedder = Unavailable()
    fact = await client.facts.remember(guild_id="g", subject_id="u", text="likes coffee")
    assert len(await client.facts.get_all("g", "u")) == 1
    assert len(await client._store.pending_projections(10, now=NOW + timedelta(days=3650))) == 1
    client.facts._embedder = original
    jobs = await client._store.pending_projections(10, now=NOW + timedelta(days=3650))
    await client._store.defer_projection(jobs[0], NOW)
    await client.ops.recover()
    assert not await client._store.pending_projections(10)
    assert fact.occurrences == 1
    await client.close(drain=False)


async def test_recall_rejects_expired_future_and_blocked_actor() -> None:
    from icelake.config import RetrievalConfig

    store = InMemoryStore()
    for record in (
        FactRecord(id="valid", guild_id="g", subject_id="u", text="valid"),
        FactRecord(id="expired", guild_id="g", text="expired", expires_at=NOW),
        FactRecord(id="future", guild_id="g", text="future", valid_from=NOW + timedelta(days=1)),
        FactRecord(id="actor", guild_id="g", text="blocked", attribution=Attribution(actor_id="b")),
    ):
        await store.insert_fact(record)
    await store.set_opt_out("g", "b", True)
    service = RecallService(
        store=store,
        vectors=None,
        embedder=None,
        config=RetrievalConfig(),
        clock=FixedClock(NOW),
        is_subject_blocked=store.get_opt_out,
    )
    result = await service.recall(
        RecallQuery(
            guild_id="g",
            subject_ids=("u",),
            channels=frozenset({ChannelName.BASELINE}),
        )
    )
    assert [sf.fact.id for sf in result.facts] == ["valid"]


@pytest.mark.parametrize("query", ["temperature=9", "max_tokens=-1"])
def test_url_config_cannot_bypass_validation(query: str) -> None:
    with pytest.raises(ValidationError):
        LlmConfig.from_url(f"openai://example.com/v1?model=test&{query}")


@pytest.mark.parametrize("budget", [0, 64, 128, 256, 512])
def test_injection_accounts_for_entire_output(budget: int) -> None:
    from icelake.models.retrieval import ScoredFact

    facts = tuple(
        ScoredFact(fact=FactRecord(id=str(i), guild_id="g", text="text " * 50), score=1)
        for i in range(8)
    )
    block, _, _ = InjectionBuilder().build(
        asker_id="u",
        guild_id="g",
        facts_by_section={"asker": facts},
        summaries={"asker": "profile " * 100},
        token_budget=budget,
    )
    assert estimate_tokens(block) <= budget


async def test_llm_citation_verifier_uses_structured_metered_boundary() -> None:
    llm = ScriptedLLM({"citation": '{"supported_ids":[0]}'})
    verifier = LlmCitationVerifier(llm, guild_id="g")
    result = await verifier.verify((CitationCandidate(id=0, claim="coffee", evidence="coffee"),))
    assert result.supported_ids == (0,)
    assert len(llm.calls) == 1
    assert llm.calls[0].response_schema and llm.calls[0].guild_id == "g"


async def test_pipeline_cancellation_returns_claim_to_queue(make_client, monkeypatch) -> None:
    from icelake.ports.queue import BatchKey, StoredMessage

    client, _ = make_client(llm=False)
    await client.start()
    await client._queue.put_message(
        StoredMessage(
            message_id="cancel",
            guild_id="g",
            author_id="u",
            subject_key="u",
            content="substantial pending observation",
            created_at=NOW,
        )
    )
    entered = asyncio.Event()

    async def blocked(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(client._pipeline, "_process_claimed", blocked)
    task = asyncio.create_task(
        client._pipeline.process_key(BatchKey(guild_id="g", subject_key="u"))
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await client._queue.pending_count("g") == 1
    assert await client._queue.dead_letter_count("g") == 0
    await client.close(drain=False)


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"index": 1, "embedding": [0.0] * 32}],
        [{"index": 0, "embedding": [0.0] * 31}],
        [{"index": 0, "embedding": [float("nan")] * 32}],
    ],
)
async def test_embedding_boundary_rejects_malformed_provider_output(data) -> None:
    import httpx

    from icelake.adapters.embedders import OpenAICompatEmbedder
    from icelake.config import EmbeddingsConfig

    embedder = OpenAICompatEmbedder(
        EmbeddingsConfig(
            provider="openai",
            base_url="https://example.com",
            model="test",
            dimensions=32,
        )
    )
    # String JSON permits the NaN fixture; the schema must reject nonfinite values.
    import json

    await embedder._client.aclose()
    embedder._client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=json.dumps({"data": data}))
        )
    )
    try:
        with pytest.raises(ValueError):
            await embedder.embed(("input",))
    finally:
        await embedder._client.aclose()
