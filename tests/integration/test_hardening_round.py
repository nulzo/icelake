"""Regression tests for the hardening round: atomic leases, transactions,
purge completeness, scope pushdown, identity grounding."""

from __future__ import annotations

import asyncio
import socket
from datetime import UTC, datetime, timedelta

import pytest

from icelake import DiscordMemory
from tests.conftest import (
    ScriptedLLM,
    extraction_response,
    make_config,
)


def _mongo_available() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 27017), timeout=0.3):
            return True
    except OSError:
        return False


GUILD = "500000000000000001"
ALICE = "100000000000000001"
BOB = "200000000000000002"


class TestAtomicLeases:
    """P0-1/2/3: one worker per key, theft impossible, heartbeat holds."""

    async def test_concurrent_claim_single_winner(self) -> None:
        from icelake.adapters.sqlite.connection import SqliteConnection
        from icelake.adapters.sqlite.queue import SqliteIngestQueue

        conn = SqliteConnection("sqlite://:memory:")
        await conn.connect()
        await conn.ensure_schema()
        queue = SqliteIngestQueue(conn)
        now = datetime.now(UTC)
        for i in range(6):
            await queue.put_message(_msg(f"m{i}", now))
        key = __import__("icelake.ports.queue", fromlist=["BatchKey"]).BatchKey(
            guild_id="g1",
            subject_key="u1",
        )

        results = await asyncio.gather(
            *(
                queue.claim_batch(key, now=now, lease_seconds=60, owner=f"w{n}", limit=10)
                for n in range(4)
            )
        )
        winners = [r for r in results if r.messages]
        losers = [r for r in results if r.locked_by_other]
        assert len(winners) == 1
        assert len(winners[0].messages) == 6
        assert len(losers) == 3

    async def test_heartbeat_holds_past_original_expiry(self) -> None:
        from icelake.adapters.sqlite.connection import SqliteConnection
        from icelake.adapters.sqlite.queue import SqliteIngestQueue
        from icelake.ports.queue import BatchKey

        conn = SqliteConnection("sqlite://:memory:")
        await conn.connect()
        await conn.ensure_schema()
        queue = SqliteIngestQueue(conn)
        now = datetime.now(UTC)
        await queue.put_message(_msg("m", now))
        key = BatchKey(guild_id="g1", subject_key="u1")
        await queue.claim_batch(key, now=now, lease_seconds=60, owner="w1", limit=5)

        # thief at t+59s (original lease not yet expired): locked out
        thief = await queue.claim_batch(
            key, now=now + timedelta(seconds=59), lease_seconds=60, owner="thief", limit=5
        )
        assert thief.locked_by_other

        # heartbeat at t+59 extends to t+119
        assert await queue.renew_lease(
            key, owner="w1", now=now + timedelta(seconds=59), lease_seconds=60
        )
        # sweep at t+100 must NOT reclaim a renewed lease
        assert await queue.release_expired_leases(now + timedelta(seconds=100)) == 0
        # but at true expiry it does
        reclaimed = await queue.release_expired_leases(now + timedelta(seconds=130))
        assert reclaimed == 1

    async def test_same_owner_reclaim_does_not_double_process(self) -> None:
        """Regression: a concurrent same-owner claim must NOT re-read an
        in-flight batch (the old SELECT-by-owner returned it and two workers
        extracted/committed the same messages twice)."""
        from icelake.adapters.sqlite.connection import SqliteConnection
        from icelake.adapters.sqlite.queue import SqliteIngestQueue
        from icelake.ports.queue import BatchKey

        conn = SqliteConnection("sqlite://:memory:")
        await conn.connect()
        await conn.ensure_schema()
        queue = SqliteIngestQueue(conn)
        now = datetime.now(UTC)
        for i in range(3):
            await queue.put_message(_msg(f"m{i}", now))
        key = BatchKey(guild_id="g1", subject_key="u1")

        first = await queue.claim_batch(key, now=now, lease_seconds=60, owner="w1", limit=10)
        assert len(first.messages) == 3
        # Same owner (a second worker in the same process) re-claims mid-flight:
        # zero rows, not the in-flight batch.
        second = await queue.claim_batch(key, now=now, lease_seconds=60, owner="w1", limit=10)
        assert not second.locked_by_other
        assert second.messages == ()
        # After completion + release_key, the key is free and nothing re-claims.
        await queue.complete_messages(tuple(m.message_id for m in first.messages), owner="w1")
        await queue.release_key(key, owner="w1")
        third = await queue.claim_batch(key, now=now, lease_seconds=60, owner="w2", limit=10)
        assert third.messages == ()

    async def test_stolen_owner_cannot_complete(self) -> None:
        from icelake.adapters.sqlite.connection import SqliteConnection
        from icelake.adapters.sqlite.queue import SqliteIngestQueue
        from icelake.ports.queue import BatchKey

        conn = SqliteConnection("sqlite://:memory:")
        await conn.connect()
        await conn.ensure_schema()
        queue = SqliteIngestQueue(conn)
        now = datetime.now(UTC)
        await queue.put_message(_msg("m", now))
        key = BatchKey(guild_id="g1", subject_key="u1")
        await queue.claim_batch(key, now=now, lease_seconds=30, owner="w-old", limit=5)
        # lease lapses; w-new claims the same message
        await queue.release_expired_leases(now + timedelta(seconds=31))
        re_claim = await queue.claim_batch(
            key, now=now + timedelta(seconds=32), lease_seconds=30, owner="w-new", limit=5
        )
        assert len(re_claim.messages) == 1
        # original owner tries to ack its stale work: rejected
        assert await queue.complete_messages(("m",), owner="w-old") == 0
        assert await queue.dead_letter_messages(("m",), owner="w-old") == 0
        # real owner completes fine
        assert await queue.complete_messages(("m",), owner="w-new") == 1


def _msg(message_id: str, now: datetime):
    from icelake.ports.queue import StoredMessage

    return StoredMessage(
        message_id=message_id,
        guild_id="g1",
        author_id="u1",
        subject_key="u1",
        content=f"content {message_id}",
        created_at=now,
    )


@pytest.mark.skipif(not _mongo_available(), reason="no MongoDB")
class TestMongoLeaseParity:
    async def test_mongo_atomic_claim_contention(self) -> None:
        from datetime import datetime as dt

        from icelake.adapters.mongo import MongoStore

        store = MongoStore("mongodb://127.0.0.1:27017/icelake_lease_test")
        await store.setup()
        await store.db["dm_messages"].delete_many({})
        await store.db["dm_batch_leases"].delete_many({})
        for i in range(4):
            await store.queue.put_message(_msg(f"ml{i}", dt.now(UTC)))
        key = __import__("icelake.ports.queue", fromlist=["BatchKey"]).BatchKey(
            guild_id="g1",
            subject_key="u1",
        )
        results = await asyncio.gather(
            *(
                store.queue.claim_batch(
                    key, now=dt.now(UTC), lease_seconds=60, owner=f"w{n}", limit=10
                )
                for n in range(3)
            )
        )
        winners = [r for r in results if r.messages]
        assert len(winners) <= 1
        assert sum(len(r.messages) for r in results) <= 4
        await store.close()


class TestPurgeCompleteness:
    async def test_sqlite_purge_deletes_vectors(self, tmp_path) -> None:
        """P0-6 regression: purged users' embeddings must not persist."""
        from icelake.api.client import DiscordMemory
        from tests.conftest import make_config

        memory = DiscordMemory(make_config(), llm=None)
        await memory.start()
        await memory.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="secret fact that must vanish on purge",
            actor_id=ALICE,
        )
        vectors_before = await memory._store.vectors.count(GUILD)
        assert vectors_before >= 1
        report = await memory.admin.purge_user(GUILD, ALICE, dry_run=False)
        assert report.facts_removed >= 1
        vectors_after = await memory._store.vectors.count(GUILD)
        assert vectors_after == 0
        await memory.close()


class TestTopStrengthBeyondCap:
    async def test_subject_anchors_found_past_global_cap(self) -> None:
        """P0-7 regression: post-LIMIT filtering starved subjects."""
        from icelake.adapters.in_memory.store import InMemoryStore
        from icelake.models.facts import (
            Attribution,
            AttributionType,
            FactCategory,
            FactRecord,
        )

        store = InMemoryStore()
        now = datetime.now(UTC)

        # 600 high-strength facts for OTHER subjects first…
        for i in range(600):
            await store.insert_fact(
                FactRecord(
                    id=f"fct_noise_{i}",
                    guild_id=GUILD,
                    subject_id=f"noise-{i}",
                    text=f"noise fact {i} filler filler filler",
                    category=FactCategory.GENERAL,
                    strength=50.0,
                    attribution=Attribution(type=AttributionType.SELF),
                    created_at=now,
                    updated_at=now,
                    valid_from=now,
                )
            )
        # …then alice's anchors, which are beyond any global top-cap.
        for i in range(5):
            await store.insert_fact(
                FactRecord(
                    id=f"fct_alice_{i}",
                    guild_id=GUILD,
                    subject_id=ALICE,
                    text=f"alice anchor fact {i} about her real hobbies here",
                    category=FactCategory.INTERESTS,
                    strength=2.0,
                    attribution=Attribution(type=AttributionType.SELF),
                    created_at=now,
                    updated_at=now,
                    valid_from=now,
                )
            )
        anchors = await store.top_strength_facts(
            GUILD,
            subject_ids=(ALICE,),
            limit=5,
        )
        assert len(anchors) == 5
        assert all(a.subject_id == ALICE for a in anchors)


class TestObserveNeverRaises:
    async def test_storage_failure_returns_rejected_receipt(self) -> None:
        class ExplodingQueue(
            __import__(
                "icelake.adapters.in_memory.queue",
                fromlist=["InMemoryIngestQueue"],
            ).InMemoryIngestQueue
        ):
            async def put_message(self, message):
                raise RuntimeError("disk full")

        from icelake import MessageEvent
        from icelake.api.client import DiscordMemory
        from icelake.models.events import ObserveStatus, RejectReason
        from tests.conftest import make_config

        memory = DiscordMemory(make_config(), llm=None, queue=ExplodingQueue())
        await memory.start()
        receipt = await memory.observe(
            MessageEvent(
                message_id="x1",
                guild_id=GUILD,
                channel_id="c",
                author_id=ALICE,
                content="anything",
                created_at=datetime.now(UTC),
            )
        )
        assert receipt.status is ObserveStatus.REJECTED
        assert receipt.reason is RejectReason.STORAGE_UNAVAILABLE


class TestIdentityGrounding:
    async def test_backfill_via_fixture(self, fixed_clock) -> None:
        from icelake.api.client import DiscordMemory
        from icelake.models.identity import AliasSource

        memory = DiscordMemory(make_config(), clock=fixed_clock, llm=None)
        await memory.start()
        count = await memory.ops.backfill_aliases(
            GUILD,
            [("klim_id", "k.song", "Klim"), ("other_id", "other", "Other Guy")],
        )
        assert count >= 2
        resolution = await memory.identity.resolve(GUILD, "Klim")
        assert resolution.resolved is not None
        assert resolution.resolved.user_id == "klim_id"
        usernames = {
            r.alias_norm
            for r in await memory.identity.aliases_of(GUILD, "klim_id")
            if r.source is AliasSource.DISCORD_USERNAME
        }
        assert "k.song" in usernames
        await memory.close()

    async def test_name_in_text_fact_fallback(self, make_client, event_factory) -> None:
        """Ladder rung 3: 'my name is X' stored facts resolve later queries."""
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "klim said his name is Klim Song during introductions",
                            "category": "personal",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        },
                    ]
                )
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        KLIM = "444000000000000004"
        event = event_factory(
            content=("nice to meet everyone, my name is klim song as some of you guessed"),
            author_id=KLIM,
            display_name="klim_song",
        )
        await client.observe(event)
        await client.flush()

        # alias index has only the display name; "klim" needs the fact fallback
        resolution = await client.identity.resolve(GUILD, "klim")
        if resolution.resolved is None:
            # fallback lives on prompt_context path; verify via prompt_context
            ctx = await client.prompt_context(
                guild_id=GUILD,
                asker_id=BOB,
                text="what did klim say at introductions?",
                mentioned_ids=("klim",),
            )
            resolutions = {r.identifier: r for r in ctx.resolutions}
            klim_res = resolutions.get("klim")
            assert klim_res is not None and klim_res.resolved is not None
            assert klim_res.resolved.user_id == KLIM
        else:
            assert resolution.resolved.user_id == KLIM
        await client.close()


class TestExtractNow:
    async def test_extract_now_commits_immediately(self) -> None:
        from icelake import MessageEvent
        from icelake.api.client import DiscordMemory
        from tests.conftest import extraction_response

        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice finished reading project hail mary recently",
                            "category": "interests",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        },
                    ]
                )
            }
        )
        memory = DiscordMemory(make_config(), llm=llm)
        await memory.start()
        receipt = await memory.extract_now(
            MessageEvent(
                message_id="en1",
                guild_id=GUILD,
                channel_id="c1",
                author_id=ALICE,
                content="just finished project hail mary and loved every page of it",
                created_at=datetime.now(UTC),
                author_display_name="alice",
            )
        )
        assert receipt.status.value == "accepted"
        page = await memory.facts.list_for_subject(GUILD, ALICE)
        assert any("hail mary" in f.text for f in page.items)
        await memory.close()


def _fact_record_for_seed():
    pass


class TestTemporalRecall:
    async def test_as_of_surfaces_then_valid_facts(self, make_client, fixed_clock) -> None:
        """Time travel: facts superseded TODAY were still valid LAST MONTH."""
        from datetime import timedelta

        from icelake.models.retrieval import RecallQuery

        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice works as a barista at the campus cafe",
                            "category": "professional",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        },
                    ]
                )
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        builder = event_factory_stub(client)
        await client.observe(
            builder(
                author_id=ALICE,
                content=("pulling espresso shots at the campus cafe again this morning"),
            )
        )
        await client.flush()
        fact = (await client.facts.list_for_subject(GUILD, ALICE, include_server=False)).items[0]

        # Age the timeline deterministically:
        #   fact created at T0; expired at T0+35d; "today" is T0+40d.
        fixed_clock.advance(timedelta(days=40).total_seconds())
        expiry = fixed_clock.now() - timedelta(days=5)
        await client._store.transition_fact(
            GUILD,
            fact.id,
            valid_until=expiry,
            updated_at=fixed_clock.now(),
        )

        # as_of between creation and expiry -> surfaces even though invalid now.
        as_of_past = fixed_clock.now() - timedelta(days=20)
        past_result = await client.recall(
            RecallQuery(
                guild_id=GUILD,
                text="barista",
                subject_ids=(ALICE,),
                as_of=as_of_past,
            )
        )
        assert any("barista" in sf.fact.text for sf in past_result.facts)

        # as_of after expiry -> correctly absent.
        future_result = await client.recall(
            RecallQuery(
                guild_id=GUILD,
                text="barista",
                subject_ids=(ALICE,),
                as_of=fixed_clock.now(),
            )
        )
        assert not any("barista" in sf.fact.text for sf in future_result.facts)
        await client.close()


def event_factory_stub(client):
    """Minimal event builder bound to this module's constants."""
    from icelake.models.events import MessageEvent

    def _build(*, author_id: str, content: str) -> MessageEvent:
        return MessageEvent(
            message_id=f"stub-{abs(hash(content)) % 10**12}",
            guild_id=GUILD,
            channel_id="c1",
            author_id=author_id,
            content=content,
            created_at=datetime.now(UTC),
            author_display_name="alice",
        )

    return _build


class TestLazyStartAndClosedGuard:
    """Regression for the reported crash: observe() before start() must never
    raise, and a fresh client must self-initialize on first use."""

    async def test_observe_autostarts_storage(self) -> None:
        from icelake import MessageEvent
        from icelake.api.client import DiscordMemory
        from tests.conftest import make_config

        memory = DiscordMemory(make_config(workers={"enabled": False}), llm=None)
        assert not memory.started  # not started yet
        receipt = await memory.observe(
            MessageEvent(
                message_id="auto1",
                guild_id=GUILD,
                channel_id="c",
                author_id=ALICE,
                content="first message ever sent to this fresh client here",
                created_at=datetime.now(UTC),
                author_display_name="alice",
            )
        )
        assert receipt.status.value == "accepted"
        assert memory.started
        page = await memory.facts.list_for_subject(GUILD, ALICE)
        del page
        await memory.close()

    async def test_group_methods_autostart(self) -> None:

        from tests.unit.test_coverage_completion import _FastClock, _SeqGen

        memory = DiscordMemory(make_config(), clock=_FastClock(), id_gen=_SeqGen(), llm=None)
        fact = await memory.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="manual fact written before any explicit start call",
            actor_id="t",
        )
        assert fact.tier.value == "core"
        await memory.close()

    async def test_observe_after_close_returns_rejected_not_crash(self) -> None:
        """After close(), the listener path gets a REJECTED receipt — no raise."""
        from icelake import MessageEvent
        from icelake.api.client import DiscordMemory
        from icelake.models.events import ObserveStatus, RejectReason
        from tests.conftest import make_config

        memory = DiscordMemory(make_config(workers={"enabled": False}), llm=None)
        await memory.start()
        await memory.close(drain=False)
        receipt = await memory.observe(
            MessageEvent(
                message_id="late1",
                guild_id=GUILD,
                channel_id="c",
                author_id=ALICE,
                content="message after shutdown completes now",
                created_at=datetime.now(UTC),
            )
        )
        assert receipt.status is ObserveStatus.REJECTED
        assert receipt.reason is RejectReason.STORAGE_UNAVAILABLE


class TestReportedCrashRegression:
    """The exact user-reported failure: bot listener calls observe() before
    start(); consent query hit a closed connection and AssertionError escaped
    into discord.py's event loop."""

    async def test_observe_without_start_autostarts_cleanly(self) -> None:
        from datetime import UTC

        from icelake import MessageEvent
        from icelake.api.client import DiscordMemory
        from tests.conftest import make_config

        memory = DiscordMemory(
            make_config(workers={"enabled": False}),
            llm=None,
        )
        # NOTE: no start() — mirrors the failing example flow.
        receipt = await memory.observe(
            MessageEvent(
                message_id="1",
                guild_id="555",
                channel_id="777",
                author_id="100000000000000001",
                content="does the reported crash reproduce on an unstarted client?",
                created_at=datetime.now(UTC),
                author_display_name="genesis",
            )
        )
        assert receipt.status.value == "accepted"
        # storage was lazily initialized; facts pipeline can now run
        await memory.flush()
        stats = await memory.stats("555")
        assert stats.total_facts >= 0
        await memory.close()

    async def test_consent_query_on_unstarted_store_raises_typed_error(self) -> None:
        """Raw store access pre-connect raises our typed error, not AssertionError."""
        from icelake.adapters.sqlite.store import SqliteStore
        from icelake.errors import StorageUnavailableError

        store = SqliteStore("sqlite://:memory:")
        with pytest.raises(StorageUnavailableError):
            await store.get_opt_out("g", "u")
