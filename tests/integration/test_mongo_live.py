from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("pymongo")


def _mongo_available() -> bool:
    import os

    if os.environ.get("MONGODB_URI"):
        return True
    try:
        with socket.create_connection(("127.0.0.1", 27017), timeout=0.3):
            return True
    except OSError:
        return False


if not _mongo_available():
    pytest.skip(
        "no MongoDB at localhost:27017 (set MONGODB_URI to enable)", allow_module_level=True
    )

from icelake.adapters.mongo import MongoStore  # noqa: E402
from icelake.models.graph import LinkRow, NodeType  # noqa: E402
from icelake.models.identity import AliasSource  # noqa: E402
from tests.integration.test_store_conformance import make_fact  # noqa: E402


@pytest.fixture()
async def store() -> AsyncIterator[MongoStore]:

    backend = MongoStore("mongodb://127.0.0.1:27017/icelake_test")
    await backend.setup()
    # clean test database between scenarios
    await backend.db["dm_facts"].delete_many({})
    await backend.db["dm_messages"].delete_many({})
    await backend.db["dm_aliases"].delete_many({})
    await backend.db["dm_links"].delete_many({})
    await backend.db["dm_relations"].delete_many({})
    await backend.db["dm_entities"].delete_many({})
    await backend.db["dm_entity_aliases"].delete_many({})
    await backend.db["dm_summaries"].delete_many({})
    await backend.db["dm_optouts"].delete_many({})
    yield backend
    await backend.close()


class TestMongoConformance:
    async def test_ping(self, store: MongoStore) -> None:
        assert await store.ping()

    async def test_aliases(self, store: MongoStore) -> None:
        await store.upsert_alias("g", "alice", "u1", AliasSource.DISCORD_USERNAME, 1.0)
        candidates = await store.resolve_alias_candidates("g", "alice")
        assert len(candidates) == 1 and candidates[0].user_id == "u1"
        prefix = await store.prefix_alias_candidates("g", "alic")
        assert any(r.user_id == "u1" for r in prefix)

    async def test_fact_crud_and_dedup(self, store: MongoStore) -> None:
        fact = make_fact(id="fct_mg1")
        await store.insert_fact(fact)
        loaded = await store.get_fact("g1", "fct_mg1")
        assert loaded is not None and loaded.text == "likes movies"
        dup = await store.find_duplicate("g1", "u1", "likes movies")
        assert dup is not None and dup.id == "fct_mg1"

    async def test_reinforce_transition_history(self, store: MongoStore) -> None:
        now = datetime.now(UTC)
        await store.insert_fact(make_fact(id="fct_mg2"))
        updated = await store.reinforce_fact(
            "g1",
            "fct_mg2",
            occurrences_delta=1,
            strength=3.0,
            last_reinforced_at=now + timedelta(hours=1),
            expires_at=None,
            tier="long_term",
            confidence=0.95,
        )
        assert updated is not None and updated.occurrences == 2
        superseded = await store.transition_fact(
            "g1",
            "fct_mg2",
            superseded_by_id="fct_newer",
            updated_at=now + timedelta(hours=2),
        )
        assert superseded is not None and not superseded.is_active
        history = await store.get_history("g1", "fct_mg2")
        del history

    async def test_links_relations_entities(self, store: MongoStore) -> None:
        from icelake.models.graph import Polarity, RelationEdge

        now = datetime.now(UTC)
        await store.insert_fact(make_fact(id="fct_lk"))
        await store.add_links(
            (
                LinkRow(
                    guild_id="g1",
                    memory_id="fct_lk",
                    node_type=NodeType.USER,
                    node_id="u1",
                    kind=__import__(
                        "icelake.models.graph", fromlist=["EdgeKind"]
                    ).EdgeKind.SUBJECT_OF,
                    created_at=now,
                ),
            )
        )
        forward = await store.links_for_node("g1", NodeType.USER, "u1")
        assert any(record.id == "fct_lk" for _, record in forward)

        edge = RelationEdge(
            guild_id="g1",
            src_type=NodeType.USER,
            src_id="u1",
            dst_type=NodeType.ENTITY,
            dst_id="movies",
            verb="likes",
            polarity=Polarity.POSITIVE,
            weight=1.0,
            valid_from=now,
            evidence_fact_ids=("fct_lk",),
        )
        await store.upsert_relation(edge)
        again = await store.upsert_relation(
            edge.model_copy(update={"evidence_fact_ids": ("fct_other",)})
        )
        assert again.occurrences == 2
        assert set(again.evidence_fact_ids) == {"fct_lk", "fct_other"}
        positives = await store.entity_stance_edges("g1", "movies", polarity=Polarity.POSITIVE)
        assert len(positives) == 1

    async def test_entities_merge(self, store: MongoStore) -> None:
        await store.upsert_entity("g", "films", "Films", "concept", ("film",))
        await store.bump_entity_facts("g", "films", delta=3)
        moved = await store.merge_entities("g", ("films",), to_slug="movies")
        assert moved == 1
        target = await store.get_entity("g", "movies")
        assert target is not None and target.fact_count >= 3

    async def test_summaries_consent_stats_export(self, store: MongoStore) -> None:
        from icelake.models.facts import ProfileSummary

        summary = ProfileSummary(guild_id="g", subject_id="u1", text="digest", source_fact_count=2)
        await store.put_summary(summary)
        loaded = await store.get_summary("g", "u1")
        assert loaded is not None and loaded.text == "digest"

        await store.set_opt_out("g", "uz", True)
        assert await store.get_opt_out("g", "uz")

        await store.insert_fact(make_fact(id="fct_st", guild_id="g"))
        stats = await store.guild_stats("g")
        assert stats.total_facts >= 1
        facts, entities, relations = await store.export_guild("g")
        assert any(f.id == "fct_st" for f in facts)
        del entities, relations

    async def test_queue_lease_cycle(self, store: MongoStore) -> None:
        from datetime import datetime as dt

        from icelake.ports.queue import BatchKey, StoredMessage

        message = StoredMessage(
            message_id="qm1",
            guild_id="gq",
            author_id="uq",
            subject_key="uq",
            content="queued content here",
            created_at=dt.now(UTC),
        )
        assert await store.queue.put_message(message) is True
        keys = await store.queue.due_batch_keys(
            now=dt.now(UTC) + timedelta(seconds=400),
            batch_size=100,
            max_age_seconds=300,
            limit=10,
        )
        assert any(k.subject_key == "uq" for k in keys)
        claim = await store.queue.claim_batch(
            BatchKey(guild_id="gq", subject_key="uq"),
            now=dt.now(UTC),
            lease_seconds=60,
            owner="w1",
            limit=5,
        )
        assert len(claim.messages) == 1
        completed = await store.queue.complete_messages(("qm1",), owner="w1")
        assert completed == 1

    async def test_purge_and_maintenance(self, store: MongoStore) -> None:
        now = datetime.now(UTC)
        record = make_fact(id="fct_pg", expires_at=now - timedelta(days=1))
        await store.insert_fact(record)
        swept = await store.sweep_expired("g1", now)
        assert swept >= 1
        forgotten = await store.apply_forgetting("g1", now=now, retention_floor=0.99)
        assert forgotten >= 0
        report = await store.purge_user_data("g1", "u1", dry_run=False)
        assert report.facts_removed >= 1
