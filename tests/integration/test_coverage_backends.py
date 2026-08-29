"""Final gap coverage: maintenance across backends, mongo queue ops, store edges."""

from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta

import pytest


def _mongo_available() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 27017), timeout=0.3):
            return True
    except OSError:
        return False


@pytest.fixture(params=["in_memory", "sqlite", "mongo"])
async def store(request):
    if request.param == "in_memory":
        from icelake.adapters.in_memory.store import InMemoryStore

        backend = InMemoryStore()
    elif request.param == "sqlite":
        from icelake.adapters.sqlite.store import SqliteStore

        backend = SqliteStore("sqlite://:memory:")
    else:
        pytest.importorskip("pymongo")
        if not _mongo_available():
            pytest.skip("no MongoDB at localhost:27017")
        from icelake.adapters.mongo import MongoStore

        backend = MongoStore("mongodb://127.0.0.1:27017/icelake_maint_test")
    await backend.setup()
    if request.param == "mongo":
        for collection in (
            "dm_facts",
            "dm_aliases",
            "dm_links",
            "dm_relations",
            "dm_entities",
            "dm_entity_aliases",
            "dm_summaries",
            "dm_optouts",
            "dm_history",
            "dm_messages",
        ):
            await backend.db[collection].delete_many({})
    yield backend
    await backend.close()


def _seed_fact(
    store, fact_id: str, *, expires_in: timedelta | None, tier: str = "mid_term"
) -> None:
    from icelake.models.facts import (
        Attribution,
        AttributionType,
        FactCategory,
        FactRecord,
    )

    now = datetime.now(UTC)
    record = FactRecord(
        id=fact_id,
        guild_id="g1",
        subject_id="u1",
        text=f"maintenance seed {fact_id} with enough words here",
        category=FactCategory.INTERESTS,
        strength=1.0,
        last_reinforced_at=now,
        created_at=now - timedelta(days=30),
        updated_at=now,
        valid_from=now - timedelta(days=30),
        expires_at=(now + expires_in) if expires_in else None,
        attribution=Attribution(type=AttributionType.SELF),
        tier=tier,
    )
    import asyncio

    asyncio.get_event_loop().run_until_complete(store.insert_fact(record))


class TestMaintenanceAcrossBackends:
    async def test_sweep_expires_stale_facts(self, store, fixed_clock) -> None:
        from datetime import timedelta

        await store.insert_fact(_fact("exp", expires_at=fixed_clock.now() - timedelta(days=1)))
        await store.insert_fact(_fact("keep", expires_at=fixed_clock.now() + timedelta(days=400)))
        swept = await store.sweep_expired("g1", fixed_clock.now())
        assert swept == 1
        gone = await store.get_fact("g1", "exp")
        assert gone is not None and not gone.is_active
        alive = await store.get_fact("g1", "keep")
        assert alive is not None and alive.is_active

    async def test_forget_weak_non_core(self, store, fixed_clock) -> None:
        from datetime import timedelta

        weak = _fact("weakmem", expires_at=None)
        weak = weak.model_copy(
            update={
                "strength": 1.0,
                "created_at": fixed_clock.now() - timedelta(days=365),
                "last_reinforced_at": fixed_clock.now() - timedelta(days=365),
            }
        )
        await store.insert_fact(weak)
        forgotten = await store.apply_forgetting(
            "g1",
            now=fixed_clock.now(),
            retention_floor=0.99,
        )
        assert forgotten >= 1

    async def test_prune_enforces_caps(self, store, fixed_clock) -> None:
        from datetime import timedelta

        from icelake.models.facts import MemoryTier

        for index in range(4):
            record = _fact(f"cap{index}", expires_at=None)
            record = record.model_copy(
                update={
                    "tier": MemoryTier.SHORT_TERM,
                    "strength": float(index) + 1.0,
                    "last_reinforced_at": fixed_clock.now() - timedelta(days=index),
                }
            )
            await store.insert_fact(record)
        pruned = await store.prune_to_caps(
            "g1",
            max_per_user=2,
            max_server=10,
            now=fixed_clock.now(),
        )
        assert pruned == 2
        page = await store.list_facts("g1", subject_id="u1", active_only=True, limit=50)
        ids = {item.id for item in page.items}
        assert ids == {"cap2", "cap3"}

    async def test_update_fact_fields_roundtrip(self, store, fixed_clock) -> None:

        await store.insert_fact(_fact("upd"))
        updated = await store.update_fact_fields(
            "g1",
            "upd",
            text="updated text about chess strategy",
            text_normalized="updated text about chess strategy",
            confidence=0.7,
            updated_at=fixed_clock.now(),
        )
        assert updated is not None
        assert "chess" in updated.text
        assert updated.confidence == 0.7
        assert updated.version == 2


def _fact(fact_id: str, *, expires_at=None):
    """Build a SELF-attributed mid-tier seed fact."""
    from datetime import datetime as dt

    from icelake.models.facts import (
        Attribution,
        AttributionType,
        FactCategory,
        FactRecord,
        MemoryTier,
    )

    now = dt.now(UTC)
    return FactRecord(
        id=fact_id,
        guild_id="g1",
        subject_id="u1",
        text=f"seeded fact body {fact_id} with plenty of context words",
        category=FactCategory.INTERESTS,
        confidence=0.9,
        tier=MemoryTier.MID_TERM,
        attribution=Attribution(type=AttributionType.SELF),
        created_at=now,
        updated_at=now,
        observed_at=now,
        valid_from=now,
        expires_at=expires_at,
    )


@pytest.mark.skipif(not _mongo_available(), reason="no MongoDB")
class TestMongoQueueEdges:
    async def test_release_requeue_recent_counts(self) -> None:
        from pymongo import AsyncMongoClient

        from icelake.adapters.mongo.queue import MongoIngestQueue
        from icelake.ports.queue import BatchKey, StoredMessage

        client = AsyncMongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=3000)
        db = client["icelake_queue_edge"]
        await db["dm_messages"].delete_many({})
        await db["dm_batch_leases"].delete_many({})
        queue = MongoIngestQueue(db)
        await queue.setup()

        message = StoredMessage(
            message_id="qe1",
            guild_id="ge",
            author_id="ue",
            subject_key="ue",
            content="edge case message",
            created_at=datetime.now(UTC),
        )
        assert await queue.put_message(message)
        key = BatchKey(guild_id="ge", subject_key="ue")
        claim = await queue.claim_batch(
            key, now=datetime.now(UTC), lease_seconds=3600, owner="w9", limit=5
        )
        assert len(claim.messages) == 1
        # recent window returns processed messages regardless of status
        recent = await queue.recent_messages("ge", 10)
        assert len(recent) == 1
        # dead-letter then requeue
        dead = await queue.dead_letter_messages(("qe1",), owner="w9")
        assert dead == 1
        assert await queue.dead_letter_count("ge") == 1
        requeued = await queue.requeue_dead_letters("ge")
        assert requeued == 1
        assert await queue.pending_count("ge") == 1
        # a DIFFERENT worker cannot steal the live lease…
        stolen = await queue.claim_batch(
            key,
            now=datetime.now(UTC),
            lease_seconds=3600,
            owner="other",
            limit=5,
        )
        assert stolen.locked_by_other
        # …but expiry reclaim works after the lease lapses
        claim2 = await queue.claim_batch(
            key,
            now=datetime.now(UTC) + timedelta(seconds=3601),
            lease_seconds=3600,
            owner="w1",
            limit=5,
        )
        assert len(claim2.messages) == 1
        await db["dm_messages"].delete_many({})
        await client.close()


@pytest.mark.skipif(not _mongo_available(), reason="no MongoDB")
class TestMongoStoreEdges:
    async def test_list_cursor_search_branches(self) -> None:

        from icelake.adapters.mongo import MongoStore

        store = MongoStore("mongodb://127.0.0.1:27017/icelake_edges")
        await store.setup()
        for collection in ("dm_facts", "dm_aliases", "dm_links"):
            await store.db[collection].delete_many({})
        for index in range(3):
            await store.insert_fact(_fact(f"curs{index}", expires_at=None))
        page_one = await store.list_facts("g1", subject_id="u1", limit=2)
        assert len(page_one.items) == 2 and page_one.next_cursor
        page_two = await store.list_facts(
            "g1", subject_id="u1", limit=5, cursor=page_one.next_cursor
        )
        assert page_two.items
        hits = await store.search_facts_text("g1", "seeded fact body")
        assert hits
        assert await store.search_facts_text("g1", "   ") == ()
        await store.close()

    async def test_transition_missing_returns_none(self) -> None:

        from icelake.adapters.mongo import MongoStore
        from icelake.models.facts import MemoryTier

        store = MongoStore("mongodb://127.0.0.1:27017/icelake_edges")
        await store.setup()
        result = await store.transition_fact(
            "g",
            "fct_nope",
            superseded_by_id="x",
            updated_at=datetime.now(UTC),
        )
        assert result is None
        updated = await store.update_fact_fields(
            "g",
            "fct_nope",
            updated_at=datetime.now(UTC),
            text="n/a",
        )
        assert updated is None
        assert (
            await store.reinforce_fact(
                "g",
                "fct_nope",
                occurrences_delta=1,
                strength=2.0,
                last_reinforced_at=datetime.now(UTC),
                expires_at=None,
                tier=MemoryTier.CORE,
                confidence=0.9,
            )
            is None
        )
        await store.close()
