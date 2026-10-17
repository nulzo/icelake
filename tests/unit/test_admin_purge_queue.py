"""AdminApi purge clears queued messages on every backend composition."""

from __future__ import annotations

from datetime import UTC, datetime

from icelake.adapters.in_memory.queue import InMemoryIngestQueue
from icelake.adapters.in_memory.store import InMemoryStore
from icelake.api.groups import AdminApi
from icelake.models.facts import FactRecord
from icelake.ports.queue import StoredMessage


def _message(message_id: str, author_id: str) -> StoredMessage:
    return StoredMessage(
        message_id=message_id,
        guild_id="g",
        author_id=author_id,
        subject_key=author_id,
        content="hello",
        created_at=datetime.now(UTC),
    )


async def test_admin_purge_drops_queue_via_injected_queue() -> None:
    store = InMemoryStore()
    await store.setup()
    queue = InMemoryIngestQueue()
    await queue.put_message(_message("m1", "u"))
    await store.insert_fact(FactRecord(id="f", guild_id="g", subject_id="u", text="t"))
    await store.set_opt_out("g", "u", True)
    try:
        admin = AdminApi(store, queue=queue)
        report = await admin.purge_user("g", "u", dry_run=False)
        assert report.messages_removed == 1
        assert await queue.pending_count("g") == 0
        assert report.facts_removed == 1
        assert await store.get_opt_out("g", "u")

        dry = await admin.purge_user("g", "u", dry_run=True)
        assert dry.messages_removed == 0
    finally:
        await store.close()


async def test_admin_purge_without_queue_still_reports_store_side() -> None:
    store = InMemoryStore()
    await store.setup()
    queue = InMemoryIngestQueue()
    await queue.put_message(_message("m1", "u"))
    try:
        admin = AdminApi(store)
        report = await admin.purge_user("g", "u", dry_run=False)
        assert report.messages_removed == 0
        assert await queue.pending_count("g") == 1
    finally:
        await store.close()
