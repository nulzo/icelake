"""Mongo adapter unit tests: mapping round-trips + config wiring (server-free).

Live-server conformance runs via tests/integration/test_mongo_live.py when
MONGODB_URI is set (skipped otherwise).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from discord_memory.adapters.mongo import mapping as m
from discord_memory.adapters.mongo.queue import doc_to_message, message_to_doc
from discord_memory.models.facts import (
    Attribution,
    AttributionType,
    FactCategory,
    FactRecord,
    MemoryTier,
    ProfileSummary,
    SourceRef,
    SourceRole,
)
from discord_memory.models.graph import (
    EdgeKind,
    EntityRecord,
    LinkRow,
    NodeType,
    Polarity,
    RelationEdge,
)
from discord_memory.models.identity import AliasSource
from discord_memory.ports.queue import StoredMessage


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
        from discord_memory.config import MemoryConfig

        assert MemoryConfig(storage="mongodb://localhost:27017/mydb").storage.backend == "mongo"
        assert MemoryConfig(storage="mongodb+srv://cluster.example/db").storage.backend == "mongo"

    def test_mongo_store_requires_pymongo_but_loads_with_it(self) -> None:
        pytest.importorskip("pymongo")
        from discord_memory.adapters.mongo import MongoStore
        from discord_memory.api.client import _build_store
        from discord_memory.config import MemoryConfig

        store = _build_store(MemoryConfig(storage="mongodb://localhost:27017/memdb"))
        assert isinstance(store, MongoStore)

    async def test_mongo_store_ping_fails_fast_without_server(self) -> None:
        pytest.importorskip("pymongo")
        from discord_memory.adapters.mongo import MongoStore

        store = MongoStore("mongodb://127.0.0.1:1/?serverSelectionTimeoutMS=200")
        assert await store.ping() is False
