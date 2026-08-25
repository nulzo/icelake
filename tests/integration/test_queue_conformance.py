"""IngestQueue conformance: leases, reclaim, dead-letters across backends."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from discord_memory.adapters.in_memory.queue import InMemoryIngestQueue
from discord_memory.adapters.sqlite.connection import SqliteConnection
from discord_memory.adapters.sqlite.queue import SqliteIngestQueue
from discord_memory.ports.queue import BatchKey, StoredMessage


def make_queue(request) -> object:
    if request.param == "in_memory":
        return InMemoryIngestQueue()
    connection = SqliteConnection("sqlite://:memory:")
    return SqliteIngestQueue(connection), connection


NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)


def msg(message_id: str, *, author: str = "u1") -> StoredMessage:
    return StoredMessage(
        message_id=message_id,
        guild_id="g1",
        author_id=author,
        subject_key=author,
        content=f"message {message_id}",
        created_at=NOW,
    )


@pytest.fixture(params=["in_memory", "sqlite"])
async def queue(request):
    if request.param == "in_memory":
        yield InMemoryIngestQueue()
    else:
        connection = SqliteConnection("sqlite://:memory:")
        await connection.connect()
        await connection.ensure_schema()
        yield SqliteIngestQueue(connection)
        await connection.close()


class TestQueueConformance:
    async def test_put_and_duplicate(self, queue) -> None:
        assert await queue.put_message(msg("m1")) is True
        assert await queue.put_message(msg("m1")) is False

    async def test_due_keys_requires_batch_size_or_age(self, queue) -> None:
        for i in range(3):
            await queue.put_message(msg(f"m{i}"))
        keys = await queue.due_batch_keys(
            now=NOW,
            batch_size=3,
            max_age_seconds=300,
            limit=10,
        )
        assert len(keys) == 1 and keys[0].subject_key == "u1"

    async def test_stale_messages_become_due(self, queue) -> None:
        await queue.put_message(msg("m-old"))
        later = NOW + timedelta(seconds=301)
        keys = await queue.due_batch_keys(
            now=later,
            batch_size=100,
            max_age_seconds=300,
            limit=10,
        )
        assert len(keys) == 1

    async def test_claim_is_atomic_and_exclusive(self, queue) -> None:
        for i in range(3):
            await queue.put_message(msg(f"m{i}"))
        key = BatchKey(guild_id="g1", subject_key="u1")
        first = await queue.claim_batch(key, now=NOW, lease_seconds=60, owner="w1", limit=3)
        assert len(first.messages) == 3
        second = await queue.claim_batch(key, now=NOW, lease_seconds=60, owner="w2", limit=3)
        assert second.locked_by_other or not second.messages

    async def test_lease_expiry_reclaims(self, queue) -> None:
        await queue.put_message(msg("m-crash"))
        key = BatchKey(guild_id="g1", subject_key="u1")
        claimed = await queue.claim_batch(key, now=NOW, lease_seconds=30, owner="w1", limit=1)
        assert len(claimed.messages) == 1
        released = await queue.release_expired_leases(NOW + timedelta(seconds=31))
        assert released == 1
        reclaimed = await queue.claim_batch(
            key, now=NOW + timedelta(seconds=32), lease_seconds=30, owner="w2", limit=1
        )
        assert len(reclaimed.messages) == 1

    async def test_complete_and_counts(self, queue) -> None:
        await queue.put_message(msg("m-done"))
        key = BatchKey(guild_id="g1", subject_key="u1")
        claim = await queue.claim_batch(key, now=NOW, lease_seconds=60, owner="w1", limit=1)
        assert await queue.pending_count("g1") == 0
        completed = await queue.complete_messages(
            tuple(m.message_id for m in claim.messages),
            owner="w1",
        )
        assert completed == 1

    async def test_dead_letter_and_requeue(self, queue) -> None:
        await queue.put_message(msg("m-poison"))
        key = BatchKey(guild_id="g1", subject_key="u1")
        claim = await queue.claim_batch(key, now=NOW, lease_seconds=60, owner="w1", limit=1)
        dead = await queue.dead_letter_messages(
            tuple(m.message_id for m in claim.messages),
            owner="w1",
        )
        assert dead == 1
        assert await queue.dead_letter_count("g1") == 1
        requeued = await queue.requeue_dead_letters("g1")
        assert requeued == 1
        assert await queue.pending_count("g1") == 1

    async def test_server_scope_same_mechanism(self, queue) -> None:
        await queue.put_message(msg("ms1", author="__server__"))
        keys = await queue.due_batch_keys(
            now=NOW + timedelta(days=1), batch_size=1, max_age_seconds=0, limit=5
        )
        assert any(key.subject_key == "__server__" for key in keys)
