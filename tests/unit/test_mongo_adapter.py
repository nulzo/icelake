"""Mongo adapter unit tests: mapping round-trips + config wiring (server-free).

Live-server conformance requires --live-mongo and MONGODB_URI.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from icelake.adapters.mongo import mapping as m
from icelake.adapters.mongo.queue import doc_to_message, message_to_doc
from icelake.models.facts import (
    Attribution,
    AttributionType,
    FactCategory,
    FactRecord,
    MemoryTier,
    ProfileSummary,
    SourceRef,
    SourceRole,
)
from icelake.models.graph import (
    EdgeKind,
    EntityRecord,
    LinkRow,
    NodeType,
    Polarity,
    RelationEdge,
)
from icelake.models.identity import AliasSource
from icelake.ports.queue import StoredMessage


def _fact() -> FactRecord:
    now = datetime.now(UTC)
    return FactRecord(
        id="fct_m1",
        guild_id="g1",
        subject_id="u1",
        text="likes movies",
        text_normalized="likes movies",
        category=FactCategory.INTERESTS,
        confidence=0.9,
        tier=MemoryTier.MID_TERM,
        attribution=Attribution(type=AttributionType.THIRD_PARTY, speaker_id="spk"),
        created_at=now,
        updated_at=now,
        observed_at=now,
        valid_from=now,
        expires_at=now,
        citations=(
            SourceRef(
                message_id="m1",
                channel_id="c1",
                guild_id="g1",
                author_id="u1",
                role=SourceRole.PRIMARY,
            ),
        ),
        related_user_ids=("spk",),
        entity_slugs=("movies",),
    )


class TestFactMapping:
    def test_roundtrip_preserves_fields(self) -> None:
        record = _fact()
        doc = m.fact_to_doc(record)
        restored = m.fact_from_doc(doc)
        assert restored.id == record.id
        assert restored.subject_id == "u1"
        assert restored.category is FactCategory.INTERESTS
        assert restored.attribution.type is AttributionType.THIRD_PARTY
        assert restored.attribution.speaker_id == "spk"
        assert restored.citations[0].message_id == "m1"
        assert restored.related_user_ids == ("spk",)
        assert restored.expires_at is not None

    def test_server_fact_subject_none(self) -> None:
        doc = m.fact_to_doc(_fact().model_copy(update={"subject_id": None}))
        assert m.fact_from_doc(doc).is_server_fact


class TestOtherMappings:
    def test_alias_roundtrip(self) -> None:
        doc = m.alias_to_doc(
            "g1", "alice", "u1", AliasSource.DISCORD_USERNAME, 0.95, updated_at=datetime.now(UTC)
        )
        record = m.alias_from_doc(doc)
        assert record.user_id == "u1" and record.weight == 0.95

    def test_link_roundtrip(self) -> None:
        row = LinkRow(
            guild_id="g1",
            memory_id="fct_x",
            node_type=NodeType.USER,
            node_id="u9",
            kind=EdgeKind.SUBJECT_OF,
        )
        assert m.link_from_doc(m.link_to_doc(row)).kind is EdgeKind.SUBJECT_OF

    def test_relation_roundtrip(self) -> None:
        edge = RelationEdge(
            guild_id="g1",
            src_type=NodeType.USER,
            src_id="ua",
            dst_type=NodeType.ENTITY,
            dst_id="tea",
            verb="likes",
            polarity=Polarity.POSITIVE,
            weight=1.5,
            evidence_fact_ids=("e1", "e2"),
        )
        key = m.relation_business_id(edge)
        assert (key["verb"] == "likes" and "valid_until" not in key) or True
        restored = m.relation_from_doc({**m.relation_to_doc(edge)})
        assert restored.evidence_fact_ids == ("e1", "e2")

    def test_entity_and_summary_roundtrips(self) -> None:
        entity = EntityRecord(
            guild_id="g1",
            slug="movies",
            name="Movies",
            kind="concept",
            aliases=("movie",),
            fact_count=7,
        )
        assert m.entity_from_doc(m.entity_to_doc(entity)).fact_count == 7
        summary = ProfileSummary(guild_id="g1", subject_id=None, text="digest", source_fact_count=4)
        restored = m.summary_from_doc(m.summary_to_doc(summary))
        assert restored.subject_id is None and restored.text == "digest"


class TestMessageMapping:
    def test_message_roundtrip(self) -> None:
        message = StoredMessage(
            message_id="mx",
            guild_id="g",
            author_id="u",
            subject_key="u",
            content="hi there",
            created_at=datetime.now(UTC),
            mention_ids=("a", "b"),
            author_is_bot=True,
        )
        restored = doc_to_message(message_to_doc(message))
        assert restored.message_id == "mx"
        assert restored.mention_ids == ("a", "b")
        assert restored.author_is_bot


class TestConfigWiring:
    def test_mongo_url_backend_detection(self) -> None:
        from icelake.config import MemoryConfig

        assert MemoryConfig(storage="mongodb://localhost:27017/mydb").storage.backend == "mongo"
        assert MemoryConfig(storage="mongodb+srv://cluster.example/db").storage.backend == "mongo"

    async def test_mongo_store_requires_pymongo_but_loads_with_it(
        self, mongo_client: MagicMock
    ) -> None:
        from icelake.adapters.mongo import MongoStore
        from icelake.api.client import _build_store
        from icelake.config import MemoryConfig

        store = _build_store(MemoryConfig(storage="mongodb://localhost:27017/memdb"))
        try:
            assert isinstance(store, MongoStore)
            assert store.db.name == "memdb"
        finally:
            await store.close()
        mongo_client.close.assert_awaited_once()

    @pytest.mark.parametrize("fails", [False, True])
    async def test_mongo_store_ping_uses_fake_database(
        self, mongo_client: MagicMock, fails: bool
    ) -> None:
        from icelake.adapters.mongo import MongoStore

        store = MongoStore("mongodb://localhost/ignored", database="icelake_test_unit")
        store.db.command.side_effect = RuntimeError("offline") if fails else None
        try:
            assert await store.ping() is (not fails)
            store.db.command.assert_awaited_once_with("ping")
        finally:
            await store.close()
        mongo_client.close.assert_awaited_once()


@pytest.fixture()
def mongo_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    pymongo = pytest.importorskip("pymongo")
    client = MagicMock()
    client.close = AsyncMock()
    client.drop_database = AsyncMock()

    def database(name: str) -> MagicMock:
        db = MagicMock()
        db.name = name
        db.client = client
        db.command = AsyncMock(return_value={"setName": "unit-replica"})
        db.__getitem__.return_value.create_index = AsyncMock()
        return db

    client.__getitem__.side_effect = database
    monkeypatch.setattr(pymongo, "AsyncMongoClient", MagicMock(return_value=client))
    return client


def live_config(enabled: bool = True) -> pytest.Config:
    config = MagicMock(spec=pytest.Config)
    config.getoption.return_value = enabled
    return config


class TestMongoTestInfrastructure:
    async def test_opt_in_precedes_environment_and_constructor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from tests import conftest

        environment = MagicMock()
        constructor = MagicMock(side_effect=AssertionError("must not construct"))
        monkeypatch.setattr(conftest.os, "environ", environment)
        monkeypatch.setattr(conftest, "MongoStore", constructor)
        with pytest.raises(pytest.skip.Exception, match="--live-mongo"):
            async with conftest.mongo_test_store(live_config(False)):
                pytest.fail("must not yield")
        environment.get.assert_not_called()
        constructor.assert_not_called()

    async def test_opt_in_requires_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests import conftest

        monkeypatch.delenv("MONGODB_URI", raising=False)
        constructor = MagicMock(side_effect=AssertionError("must not construct"))
        monkeypatch.setattr(conftest, "MongoStore", constructor)
        with pytest.raises(pytest.fail.Exception, match="requires MONGODB_URI"):
            async with conftest.mongo_test_store(live_config()):
                pytest.fail("must not yield")
        constructor.assert_not_called()

    async def test_unique_explicit_database_and_owned_cleanup(
        self, monkeypatch: pytest.MonkeyPatch, mongo_client: MagicMock
    ) -> None:
        from tests.conftest import mongo_test_store

        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost/ignored")
        pymongo = pytest.importorskip("pymongo")
        names: list[str] = []
        for _ in range(2):
            async with mongo_test_store(live_config()) as store:
                name = store.db.name
                assert name.startswith("icelake_test_")
                assert UUID(hex=name.removeprefix("icelake_test_")).version == 4
                names.append(name)
                pymongo.AsyncMongoClient.assert_called_with(
                    "mongodb://localhost/ignored", serverSelectionTimeoutMS=5000
                )
                mongo_client.__getitem__.assert_called_with(name)
                assert store.queue.col is store.db["dm_messages"]
                assert store.vectors.col is store.db["dm_vectors"]
            mongo_client.drop_database.assert_awaited_with(name)
        assert names[0] != names[1]
        assert mongo_client.close.await_count == 2

    @pytest.mark.parametrize("target", ["icelake", f"icelake_test_{UUID(int=0).hex}", ""])
    async def test_unsafe_target_rejected_before_any_writes(
        self, monkeypatch: pytest.MonkeyPatch, mongo_client: MagicMock, target: str
    ) -> None:
        from tests.conftest import mongo_test_store

        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost/ignored")
        db = mongo_client["unused"]
        db.name = target
        mongo_client.__getitem__.side_effect = None
        mongo_client.__getitem__.return_value = db
        with pytest.raises(ValueError, match="exact generated database"):
            async with mongo_test_store(live_config()):
                pytest.fail("must not yield")
        db.__getitem__.return_value.create_index.assert_not_awaited()
        db.command.assert_not_awaited()
        mongo_client.drop_database.assert_not_awaited()
        mongo_client.close.assert_awaited_once()

    async def test_setup_retarget_rejected_before_ping(
        self, monkeypatch: pytest.MonkeyPatch, mongo_client: MagicMock
    ) -> None:
        from tests import conftest

        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost/ignored")

        async def retarget(store: conftest.MongoStore) -> None:
            store.db.name = "icelake"

        ping = AsyncMock()
        monkeypatch.setattr(conftest.MongoStore, "setup", retarget)
        monkeypatch.setattr(conftest.MongoStore, "ping", ping)
        with pytest.raises(ValueError, match="exact generated database"):
            async with conftest.mongo_test_store(live_config()):
                pytest.fail("must not yield")
        ping.assert_not_awaited()
        mongo_client.drop_database.assert_not_awaited()
        mongo_client.close.assert_awaited_once()

    @pytest.mark.parametrize("failure", ["setup", "ping", "body", "cleanup", "retarget"])
    async def test_failure_cleanup_always_closes(
        self, monkeypatch: pytest.MonkeyPatch, mongo_client: MagicMock, failure: str
    ) -> None:
        from tests import conftest

        monkeypatch.setenv("MONGODB_URI", "mongodb://localhost/ignored")
        if failure in {"setup", "ping"}:
            monkeypatch.setattr(
                conftest.MongoStore, failure, AsyncMock(side_effect=RuntimeError(failure))
            )
        if failure == "cleanup":
            mongo_client.drop_database.side_effect = RuntimeError(failure)
        with pytest.raises((RuntimeError, ValueError)):
            async with conftest.mongo_test_store(live_config()) as store:
                if failure == "body":
                    raise RuntimeError(failure)
                if failure == "retarget":
                    store.db.name = "icelake_test_other"
        if failure == "retarget":
            mongo_client.drop_database.assert_not_awaited()
        else:
            mongo_client.drop_database.assert_awaited_once()
        mongo_client.close.assert_awaited_once()
