"""Fault injection at durable boundaries, using the real adapter contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from icelake import DiscordMemory, MessageEvent, SchemaValidationError
from icelake.adapters.embedders import HashingEmbedder
from icelake.adapters.in_memory.vectors import InMemoryVectorIndex
from icelake.adapters.sqlite.store import SqliteStore
from icelake.ingest.pipeline import _receipt_key
from icelake.ingest.retries import read_retry
from icelake.models.facts import FactRecord
from icelake.models.recovery import ProjectionJob
from icelake.ports.clock import FixedClock
from icelake.ports.queue import BatchKey
from icelake.ports.vectors import VectorItem
from icelake.recovery import recover_projections
from tests.conftest import ExplodingLLM, ScriptedLLM, extraction_response, make_config

NOW = datetime(2026, 10, 1, tzinfo=UTC)
KEY = BatchKey(guild_id="g", subject_key="u")


def llm():
    return ScriptedLLM(
        {
            "extraction": extraction_response(
                [
                    {
                        "subject_token": "p0",
                        "text": "Alice restores antique watches as a weekend hobby",
                        "confidence": 0.9,
                        "category": "interests",
                        "source_message_indexes": [1],
                    }
                ]
            )
        }
    )


def message(identifier="m", channel="c"):
    return MessageEvent(
        message_id=identifier,
        guild_id="g",
        channel_id=channel,
        author_id="u",
        content="I enjoy collecting antique pocket watches.",
        created_at=NOW,
        author_display_name="Alice",
    )


async def test_transaction_rolls_back_fact_receipt_and_projection(store):
    job = ProjectionJob(guild_id="g", fact_id="f", version=1)
    with pytest.raises(RuntimeError, match="crash"):
        async with store.transaction():
            await store.insert_fact(FactRecord(id="f", guild_id="g", text="test"))
            await store.enqueue_projection(job)
            await store.set_cursor("g", "ingest:test", "done")
            raise RuntimeError("crash")
    assert await store.get_fact("g", "f") is None
    assert await store.pending_projections(10) == ()
    assert await store.get_cursor("g", "ingest:test") is None


async def test_outbox_ack_cannot_delete_newer_work(store):
    old = ProjectionJob(guild_id="g", fact_id="f", version=1, embedding=(1, 0))
    new = old.model_copy(update={"version": 2, "embedding": (0, 1)})
    await store.enqueue_projection(new)
    await store.enqueue_projection(old)
    await store.complete_projection(old)
    assert await store.pending_projections(10) == (new,)
    await store.complete_projection(new)
    assert not await store.pending_projections(10)


async def test_index_rejects_stale_projection(store):
    vectors = getattr(store, "vectors", None) or InMemoryVectorIndex()
    await vectors.setup()
    newer = VectorItem(id="f", guild_id="g", subject_id="u", embedding=(0, 1), version=2)
    await vectors.upsert((newer, newer.model_copy(update={"version": 1, "embedding": (1, 0)})))
    hits = await vectors.search((0, 1), guild_id="g")
    assert hits and hits[0].score == pytest.approx(1)


async def test_lost_ack_replay_does_not_reinforce_or_call_llm(tmp_path):
    url = f"sqlite:///{tmp_path / 'memory.sqlite'}"
    first_llm = llm()
    memory = DiscordMemory(make_config(storage=url), llm=first_llm)
    await memory.start()
    await memory.observe(message())
    original = memory._queue.complete_messages

    async def lost_ack(*args, **kwargs):
        raise RuntimeError("ack connection lost")

    memory._queue.complete_messages = lost_ack
    await memory.flush()
    facts = await memory.facts.get_all("g", "u")
    assert len(facts) == 1 and facts[0].occurrences == 1
    assert await memory._store.get_cursor("g", _receipt_key(KEY, "m"))
    memory._queue.complete_messages = original
    await memory.close(drain=False)
    replay_llm = llm()
    async with DiscordMemory(make_config(storage=url), llm=replay_llm) as restarted:
        await restarted.ops.retry_dead_letters("g")
        await restarted.flush()
        facts = await restarted.facts.get_all("g", "u")
        assert len(facts) == 1 and facts[0].occurrences == 1
        assert not replay_llm.calls


async def test_failed_graph_batch_has_no_receipt_and_can_retry(tmp_path):
    memory = DiscordMemory(make_config(), llm=llm())
    async with memory:
        await memory.observe(message())
        original = memory._store.add_links

        async def fail(rows):
            await original(rows)
            raise RuntimeError("graph unavailable")

        memory._store.add_links = fail
        await memory.flush()
        assert not await memory.facts.get_all("g", "u")
        assert not await memory._store.pending_projections(10)
        assert await memory._store.get_cursor("g", _receipt_key(KEY, "m")) is None
        memory._store.add_links = original
        await memory.ops.retry_dead_letters("g")
        await memory.flush()
        assert len(await memory.facts.get_all("g", "u")) == 1


async def test_retry_backoff_and_attempt_limit_survive_restart(tmp_path):
    clock = FixedClock(NOW)
    config = make_config(
        storage=f"sqlite:///{tmp_path / 'retry.sqlite'}",
        recovery={"max_attempts": 2, "retry_initial_seconds": 10},
    )
    first = DiscordMemory(config, clock=clock, llm=ExplodingLLM())
    await first.start()
    await first.observe(message())
    await first.flush()
    retry = await read_retry(first._store, KEY)
    assert retry.attempts == 1 and NOW < retry.next_attempt_at <= NOW + timedelta(seconds=10)
    assert (await first.ops.health()).pending_messages == 1
    await first.close(drain=False)
    async with DiscordMemory(config, clock=clock, llm=ExplodingLLM()) as second:
        assert (await second._pipeline.process_key(KEY)).skipped_reason == "retry_backoff"
        clock.advance(seconds=11)
        await second.flush()
        assert (await second.ops.health()).dead_letters == 1
        assert (await read_retry(second._store, KEY)).attempts == 2


async def test_external_index_failure_survives_reopening_store(tmp_path):
    store = SqliteStore(f"sqlite:///{tmp_path / 'index.sqlite'}")
    await store.setup()
    async with store.transaction():
        await store.insert_fact(FactRecord(id="f", guild_id="g", text="likes coffee"))
        await store.enqueue_projection(ProjectionJob(guild_id="g", fact_id="f", version=1))

    class Offline(InMemoryVectorIndex):
        async def upsert(self, items):
            raise OSError("offline")

    assert (
        await recover_projections(store, Offline(), HashingEmbedder(64), retry_delay_seconds=0)
    ).failed == 1
    await store.close()
    restored = SqliteStore(f"sqlite:///{tmp_path / 'index.sqlite'}")
    await restored.setup()
    index = InMemoryVectorIndex()
    report = await recover_projections(restored, index, HashingEmbedder(64))
    assert report.completed == 1 and await index.count("g") == 1
    assert not await restored.pending_projections(10)
    await restored.close()


async def test_manual_idempotency_is_atomic_and_rejects_conflicting_reuse(store):
    memory = DiscordMemory(make_config(), store=store)
    await memory.start()
    first = await memory.facts.remember(
        guild_id="g", subject_id="u", text="likes coffee", idempotency_key="discord-interaction-1"
    )
    again = await memory.facts.remember(
        guild_id="g", subject_id="u", text="likes coffee", idempotency_key="discord-interaction-1"
    )
    assert again.id == first.id and again.occurrences == 1
    with pytest.raises(SchemaValidationError, match="different inputs"):
        await memory.facts.remember(
            guild_id="g",
            subject_id="u",
            text="likes tea instead",
            idempotency_key="discord-interaction-1",
        )
    assert (await memory.facts.get("g", first.id)).occurrences == 1
    # The parametrized fixture owns and closes this store after assertions.


async def test_failed_projection_does_not_starve_other_facts(store):
    past = datetime.now(UTC) - timedelta(seconds=1)
    future = datetime.now(UTC) + timedelta(hours=1)
    first = ProjectionJob(guild_id="g", fact_id="a", version=1)
    second = ProjectionJob(guild_id="g", fact_id="b", version=1)
    await store.enqueue_projection(first)
    await store.enqueue_projection(second)
    await store.defer_projection(first, future)
    assert await store.pending_projections(1) == (second,)
    await store.defer_projection(first, past)
    assert {j.fact_id for j in await store.pending_projections(10)} == {"a", "b"}
