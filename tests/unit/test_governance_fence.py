"""Governance fence: queued/in-flight opt-out and purge must not revive old output."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from icelake.adapters.in_memory.queue import InMemoryIngestQueue
from icelake.adapters.in_memory.store import InMemoryStore
from icelake.adapters.sqlite.store import SqliteStore
from icelake.identity.guards import BotGuard, ConsentPolicy, SubjectGate
from icelake.models.facts import Attribution, AttributionType, FactRecord
from icelake.ports.clock import FixedClock, UlidIdGen
from icelake.ports.queue import StoredMessage


def _message(message_id: str, author_id: str, gen: int = 0):
    return StoredMessage(
        message_id=message_id,
        guild_id="g",
        author_id=author_id,
        subject_key=author_id,
        content="hello there",
        created_at=datetime.now(UTC),
        governance={"u": gen},
    )


def test_mongo_message_round_trip_preserves_governance_stamp() -> None:
    from icelake.adapters.mongo.queue import doc_to_message, message_to_doc

    message = _message("m1", "u", gen=3)
    restored = doc_to_message(message_to_doc(message))
    assert restored.governance == {"u": 3}


def test_mongo_fact_mapping_round_trip_preserves_governance_stamp() -> None:
    from icelake.adapters.mongo.mapping import fact_from_doc, fact_to_doc

    record = FactRecord(id="f1", guild_id="g", subject_id="u", text="t", governance={"u": 4})
    assert fact_from_doc(fact_to_doc(record)).governance == {"u": 4}


@pytest.mark.parametrize("backend", ["in_memory", "sqlite"])
async def test_purge_drops_queued_and_bumps_generation(backend: str) -> None:
    store = InMemoryStore() if backend == "in_memory" else SqliteStore("sqlite://:memory:")
    await store.setup()
    queue = store.queue if backend == "sqlite" else InMemoryIngestQueue()
    try:
        assert await queue.put_message(_message("m1", "u"))
        assert await queue.pending_count("g") == 1

        await store.set_opt_out("g", "u", True)
        before = await store.consent_state("g", "u")
        assert before.opted_out and before.generation == 0

        generation = await store.bump_governance_generation("g", "u")
        assert generation == 1
        dropped = await queue.drop_governance("g", "u")
        assert dropped == 1
        assert await queue.pending_count("g") == 0

        report = await store.purge_user_data("g", "u", dry_run=False)
        assert report.facts_removed == 0
        assert await store.get_opt_out("g", "u")

        await store.set_opt_out("g", "u", False)
        after = await store.consent_state("g", "u")
        assert not after.opted_out
        assert after.generation == 2

        assert await queue.put_message(_message("m2", "u", gen=2))
        assert await queue.pending_count("g") == 1
    finally:
        await store.close()


@pytest.mark.parametrize("backend", ["in_memory", "sqlite"])
async def test_pre_purge_facts_fenced_after_optin(backend: str) -> None:
    store = InMemoryStore() if backend == "in_memory" else SqliteStore("sqlite://:memory:")
    await store.setup()
    try:
        await store.insert_fact(FactRecord(id="f1", guild_id="g", subject_id="u", text="old fact"))
        await store.set_opt_out("g", "u", True)
        await store.bump_governance_generation("g", "u")
        assert (await store.consent_state("g", "u")).generation == 1
        await store.purge_user_data("g", "u", dry_run=False)
        state = await store.consent_state("g", "u")
        assert state.generation == 2
        await store.set_opt_out("g", "u", False)

        assert await store.get_fact("g", "f1") is None

        stale = FactRecord(
            id="f2", guild_id="g", subject_id="u", text="in-flight fact", governance={"u": 1}
        )
        await store.insert_fact(stale)
        assert await store.get_fact("g", "f2") is None
        page = await store.list_facts("g", subject_id="u", include_server=True)
        assert page.items == ()

        fresh = FactRecord(
            id="f3", guild_id="g", subject_id="u", text="new fact", governance={"u": 2}
        )
        await store.insert_fact(fresh)
        assert await store.get_fact("g", "f3") is not None
    finally:
        await store.close()


@pytest.mark.parametrize("backend", ["in_memory", "sqlite"])
async def test_graph_and_search_reads_fence_pre_purge_rows(backend: str) -> None:
    from icelake.models.graph import EdgeKind, LinkRow, NodeType

    store = InMemoryStore() if backend == "in_memory" else SqliteStore("sqlite://:memory:")
    await store.setup()
    try:
        stale = FactRecord(
            id="f-stale", guild_id="g", subject_id="u", text="stale secret", governance={"u": 0}
        )
        await store.insert_fact(stale)
        await store.add_links(
            (
                LinkRow(
                    memory_id="f-stale",
                    guild_id="g",
                    node_type=NodeType.USER,
                    node_id="u",
                    kind=EdgeKind.SUBJECT_OF,
                ),
            )
        )
        await store.set_opt_out("g", "u", True)
        await store.purge_user_data("g", "u", dry_run=False)
        await store.set_opt_out("g", "u", False)

        assert await store.search_facts_text("g", "stale secret") == ()
        assert await store.links_for_node("g", NodeType.USER, "u") == ()
        assert await store.links_for_nodes("g", ((NodeType.USER, "u"),)) == ()
    finally:
        await store.close()


@pytest.mark.parametrize("backend", ["in_memory", "sqlite"])
async def test_cross_user_generations_independent_both_directions(backend: str) -> None:
    store = InMemoryStore() if backend == "in_memory" else SqliteStore("sqlite://:memory:")
    await store.setup()
    try:
        await store.set_opt_out("g", "alice", True)
        await store.purge_user_data("g", "alice", dry_run=False)
        await store.set_opt_out("g", "alice", False)
        alice_gen1 = (await store.consent_state("g", "alice")).generation
        assert alice_gen1 == 1
        bob_fact = FactRecord(
            id="f-bob",
            guild_id="g",
            subject_id="bob",
            text="bob fact",
            governance={"bob": 0},
        )
        await store.insert_fact(bob_fact)
        assert await store.get_fact("g", "f-bob") is not None

        for _ in range(4):
            await store.bump_governance_generation("g", "alice")
        alice_fact = FactRecord(
            id="f-alice-src",
            guild_id="g",
            subject_id="bob",
            text="about alice",
            attribution=Attribution(type=AttributionType.THIRD_PARTY, speaker_id="alice"),
            governance={"bob": 0, "alice": 5},
        )
        await store.insert_fact(alice_fact)
        assert await store.get_fact("g", "f-alice-src") is not None
        await store.bump_governance_generation("g", "bob")
        assert await store.get_fact("g", "f-alice-src") is None
        await store.purge_user_data("g", "bob", dry_run=False)
        bob_gen = (await store.consent_state("g", "bob")).generation
        assert await store.get_fact("g", "f-bob") is None
        fresh_after_bob = FactRecord(
            id="f-bob2",
            guild_id="g",
            subject_id="bob",
            text="post purge",
            governance={"bob": bob_gen},
        )
        await store.insert_fact(fresh_after_bob)
        assert await store.get_fact("g", "f-bob2") is not None
    finally:
        await store.close()


@pytest.mark.parametrize("backend", ["in_memory", "sqlite"])
async def test_manual_remember_after_purge_is_visible(backend: str) -> None:
    from icelake.api.facts_api import FactsApi

    store = InMemoryStore() if backend == "in_memory" else SqliteStore("sqlite://:memory:")
    await store.setup()
    try:
        await store.set_opt_out("g", "u", True)
        await store.purge_user_data("g", "u", dry_run=False)
        await store.set_opt_out("g", "u", False)
        generation = (await store.consent_state("g", "u")).generation
        from icelake.api.facts_api import FactsApi
        from icelake.config import MemoryConfig

        api = FactsApi(
            store=store,
            vectors=None,
            embedder=None,
            clock=FixedClock(datetime.now(UTC)),
            id_gen=UlidIdGen(),
            config=MemoryConfig(storage="sqlite://:memory:"),
            subject_gate=SubjectGate(BotGuard(), ConsentPolicy(store)),
        )
        record = await api.remember(guild_id="g", subject_id="u", text="brand new memory")
        assert record.governance.get("u") == generation
        assert await store.get_fact("g", record.id) is not None
    finally:
        await store.close()
