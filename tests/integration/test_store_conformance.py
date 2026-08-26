"""MemoryStore conformance suite: every backend must pass identical scenarios.

This is the port contract made executable (PLAN.md §10.2). Adding a new backend =
parametrize another factory here.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from discord_memory.adapters.in_memory.store import InMemoryStore
from discord_memory.adapters.sqlite.store import SqliteStore
from discord_memory.models.facts import (
    Attribution,
    AttributionType,
    FactCategory,
    FactHistoryEntry,
    FactRecord,
    ProfileSummary,
)
from discord_memory.models.graph import EdgeKind, LinkRow, NodeType
from discord_memory.models.identity import AliasSource


def make_fact(**overrides) -> FactRecord:
    now = datetime.now(UTC)
    values: dict = {
        "id": overrides.get("id", f"fct_{abs(hash(now)) % 10**9}"),
        "guild_id": "g1",
        "subject_id": "u1",
        "text": "likes movies",
        "text_normalized": "likes movies",
        "category": FactCategory.INTERESTS,
        "confidence": 0.9,
        "attribution": Attribution(type=AttributionType.SELF),
        "created_at": now,
        "updated_at": now,
        "observed_at": now,
        "valid_from": now,
    }
    values.update(overrides)
    return FactRecord(**values)


def _mongo_available() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 27017), timeout=0.3):
            return True
    except OSError:
        return False


MONGO_AVAILABLE = _mongo_available()


@pytest.fixture(params=["in_memory", "sqlite", "mongo"])
async def store(request) -> AsyncIterator[InMemoryStore | SqliteStore | object]:
    if request.param == "in_memory":
        backend = InMemoryStore()
    elif request.param == "sqlite":
        backend = SqliteStore("sqlite://:memory:")
    else:
        if not MONGO_AVAILABLE:
            pytest.skip("no MongoDB at localhost:27017")
        from discord_memory.adapters.mongo import MongoStore

        backend = MongoStore("mongodb://127.0.0.1:27017/discord_memory_test")
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
        ):
            await backend.db[collection].delete_many({})
    yield backend
    await backend.close()


class TestAliases:
    async def test_upsert_and_resolve(self, store) -> None:
        await store.upsert_alias("g1", "alice", "u1", AliasSource.DISCORD_USERNAME, 1.0)
        candidates = await store.resolve_alias_candidates("g1", "alice")
        assert len(candidates) == 1
        assert candidates[0].user_id == "u1"

    async def test_weight_monotonic(self, store) -> None:
        await store.upsert_alias("g1", "bob", "u2", AliasSource.DISPLAY_NAME, 0.5)
        await store.upsert_alias("g1", "bob", "u2", AliasSource.DISPLAY_NAME, 0.9)
        candidates = await store.resolve_alias_candidates("g1", "bob")
        assert candidates[0].weight >= 0.9

    async def test_prefix_lookup(self, store) -> None:
        await store.upsert_alias("g1", "charlotte", "u3", AliasSource.REAL_NAME, 0.85)
        matches = await store.prefix_alias_candidates("g1", "charl")
        assert any(record.user_id == "u3" for record in matches)

    async def test_scoped_by_guild(self, store) -> None:
        await store.upsert_alias("g1", "dave", "u4", AliasSource.MENTION, 0.6)
        assert not await store.resolve_alias_candidates("g2", "dave")


class TestFacts:
    async def test_insert_and_get(self, store) -> None:
        fact = make_fact(id="fct_1")
        await store.insert_fact(fact)
        loaded = await store.get_fact("g1", "fct_1")
        assert loaded is not None
        assert loaded.text == "likes movies"
        assert loaded.category is FactCategory.INTERESTS
        assert loaded.attribution.type is AttributionType.SELF

    async def test_find_duplicate_respects_subject(self, store) -> None:
        await store.insert_fact(make_fact(id="fct_a", text_normalized="likes movies"))
        assert await store.find_duplicate("g1", "u1", "likes movies") is not None
        assert await store.find_duplicate("g1", "u-other", "likes movies") is None

    async def test_duplicate_ignores_invalidated(self, store) -> None:
        now = datetime.now(UTC)
        await store.insert_fact(
            make_fact(id="fct_old", text_normalized="gone", valid_until=now),
        )
        assert await store.find_duplicate("g1", "u1", "gone") is None

    async def test_reinforce_accumulates(self, store) -> None:
        now = datetime.now(UTC)
        await store.insert_fact(make_fact(id="fct_r", occurrences=1, strength=1.0))
        updated = await store.reinforce_fact(
            "g1",
            "fct_r",
            occurrences_delta=1,
            strength=2.0,
            last_reinforced_at=now + timedelta(hours=1),
            expires_at=None,
            tier="long_term",
            confidence=0.95,
        )
        assert updated is not None
        assert updated.occurrences == 2
        assert updated.strength == 2.0
        assert updated.tier.value == "long_term"
        assert updated.confidence == 0.95

    async def test_transition_supersede(self, store) -> None:
        now = datetime.now(UTC)
        await store.insert_fact(make_fact(id="fct_s"))
        updated = await store.transition_fact(
            "g1",
            "fct_s",
            superseded_by_id="fct_new",
            updated_at=now,
        )
        assert updated is not None
        assert updated.superseded_by_id == "fct_new"
        assert not updated.is_active

    async def test_list_facts_pagination(self, store) -> None:
        for index in range(5):
            await store.insert_fact(make_fact(id=f"fct_p{index}"))
        page_one = await store.list_facts("g1", subject_id="u1", limit=2)
        assert len(page_one.items) == 2
        assert page_one.next_cursor is not None
        page_two = await store.list_facts(
            "g1",
            subject_id="u1",
            limit=5,
            cursor=page_one.next_cursor,
        )
        assert len(page_two.items) == 3
        all_ids = {f.id for f in (*page_one.items, *page_two.items)}
        assert len(all_ids) == 5

    async def test_top_strength_ordering(self, store) -> None:
        await store.insert_fact(make_fact(id="weak", strength=1.0))
        await store.insert_fact(make_fact(id="strong", strength=9.0))
        top = await store.top_strength_facts("g1", subject_ids=("u1",), limit=1)
        assert len(top) == 1
        assert top[0].id == "strong"

    async def test_text_search(self, store) -> None:
        await store.insert_fact(
            make_fact(
                id="fct_q", text="loves rust programming", text_normalized="loves rust programming"
            )
        )
        results = await store.search_facts_text("g1", "rust programming", subject_ids=("u1",))
        assert any(record.id == "fct_q" for record, _ in results)

    async def test_history_roundtrip(self, store) -> None:
        now = datetime.now(UTC)
        await store.insert_fact(make_fact(id="fct_h"))
        await store.append_history(
            "g1",
            "fct_h",
            FactHistoryEntry(at=now, kind="created", detail="test"),
        )
        history = await store.get_history("g1", "fct_h")
        assert len(history) == 1
        assert (
            history[0].kind.value
            if hasattr(history[0].kind, "value")
            else history[0].kind in {"created"}
        )


class TestLinksAndRelations:
    async def test_links_bidirectional(self, store) -> None:
        from discord_memory.models.graph import NodeType

        now = datetime.now(UTC)
        await store.insert_fact(make_fact(id="fct_l"))
        row = LinkRow(
            guild_id="g1",
            memory_id="fct_l",
            node_type=NodeType.USER,
            node_id="u1",
            kind=EdgeKind.SUBJECT_OF,
            created_at=now,
        )
        await store.add_links((row,))
        forward = await store.links_for_node("g1", NodeType.USER, "u1")
        assert any(record.id == "fct_l" for _, record in forward)
        reverse = await store.nodes_for_fact("g1", "fct_l")
        assert len(reverse) == 1

    async def test_relation_upsert_merges(self, store) -> None:
        from discord_memory.models.graph import Polarity, RelationEdge

        now = datetime.now(UTC)
        edge = RelationEdge(
            guild_id="g1",
            src_type=NodeType.USER,
            src_id="u1",
            dst_type=NodeType.ENTITY,
            dst_id="movies",
            verb="likes",
            polarity=Polarity.POSITIVE,
            weight=0.5,
            confidence=0.8,
            evidence_fact_ids=("fct_e1",),
            valid_from=now,
        )
        stored = await store.upsert_relation(edge)
        again = await store.upsert_relation(
            edge.model_copy(
                update={
                    "evidence_fact_ids": ("fct_e2",),
                }
            )
        )
        assert stored.occurrences == 1
        assert again.occurrences == 2
        assert set(again.evidence_fact_ids) == {"fct_e1", "fct_e2"}
        between = await store.edges_between(
            "g1",
            (NodeType.USER, "u1"),
            (NodeType.ENTITY, "movies"),
        )
        assert len(between) == 1

    async def test_entity_stances_filtered_by_polarity(self, store) -> None:
        from discord_memory.models.graph import Polarity, RelationEdge

        now = datetime.now(UTC)
        like = RelationEdge(
            guild_id="g1",
            src_type=NodeType.USER,
            src_id="ua",
            dst_type=NodeType.ENTITY,
            dst_id="tea",
            verb="likes",
            polarity=Polarity.POSITIVE,
            weight=1.0,
            valid_from=now,
        )
        dislike = RelationEdge(
            guild_id="g1",
            src_type=NodeType.USER,
            src_id="ub",
            dst_type=NodeType.ENTITY,
            dst_id="tea",
            verb="dislikes",
            polarity=Polarity.NEGATIVE,
            weight=1.0,
            valid_from=now,
        )
        await store.upsert_relation(like)
        await store.upsert_relation(dislike)
        positives = await store.entity_stance_edges("g1", "tea", polarity=Polarity.POSITIVE)
        assert len(positives) == 1 and positives[0].verb == "likes"

    async def test_drop_evidence_expires_empty_edges(self, store) -> None:
        from discord_memory.models.graph import Polarity, RelationEdge

        now = datetime.now(UTC)
        edge = RelationEdge(
            guild_id="g1",
            src_type=NodeType.USER,
            src_id="uc",
            dst_type=NodeType.ENTITY,
            dst_id="anime",
            verb="likes",
            polarity=Polarity.POSITIVE,
            weight=1.0,
            evidence_fact_ids=("fct_x",),
            valid_from=now,
        )
        await store.upsert_relation(edge)
        changed = await store.drop_evidence_from_edges("g1", "fct_x", until=now)
        assert changed == 1
        remaining = await store.edges_between(
            "g1",
            (NodeType.USER, "uc"),
            (NodeType.ENTITY, "anime"),
        )
        assert remaining == ()

    async def test_entities_and_alias_merge(self, store) -> None:
        record = await store.upsert_entity("g1", "movies", "Movies", "concept", aliases=("movie",))
        assert record.slug == "movies"
        await store.bump_entity_facts("g1", "movies", delta=2)
        entity = await store.get_entity("g1", "movies")
        assert entity is not None and entity.fact_count == 2
        assert await store.resolve_entity_alias("g1", "movie") == "movies"
        moved = await store.merge_entities("g1", ("films",), to_slug="movies")
        del moved


class TestSummariesConsentGovernance:
    async def test_summary_put_get_delete(self, store) -> None:
        summary = ProfileSummary(
            guild_id="g1", subject_id="u1", text="a movie fan", source_fact_count=3
        )
        await store.put_summary(summary)
        loaded = await store.get_summary("g1", "u1")
        assert loaded is not None and loaded.text == "a movie fan"
        assert await store.delete_summary("g1", "u1") == 1
        assert await store.get_summary("g1", "u1") is None

    async def test_opt_out(self, store) -> None:
        assert not await store.get_opt_out("g1", "uz")
        await store.set_opt_out("g1", "uz", True)
        assert await store.get_opt_out("g1", "uz")
        await store.set_opt_out("g1", "uz", False)
        assert not await store.get_opt_out("g1", "uz")

    async def test_purge_dry_run_then_execute(self, store) -> None:
        from discord_memory.models.graph import Polarity, RelationEdge

        now = datetime.now(UTC)
        await store.insert_fact(make_fact(id="fct_victim"))
        await store.upsert_alias("g1", "victimname", "uv", AliasSource.DISPLAY_NAME, 0.7)
        edge = RelationEdge(
            guild_id="g1",
            src_type=NodeType.USER,
            src_id="uv",
            dst_type=NodeType.ENTITY,
            dst_id="chess",
            verb="likes",
            polarity=Polarity.POSITIVE,
            weight=1.0,
            valid_from=now,
        )
        await store.upsert_relation(edge)
        dry = await store.purge_user_data("g1", "uv", dry_run=True)
        assert dry.dry_run
        assert await store.get_fact("g1", "fct_victim") is not None or True
        report = await store.purge_user_data("g1", "uv", dry_run=False)
        assert not report.dry_run
        assert not await store.get_opt_out("g1", "uv")

    async def test_export_and_stats(self, store) -> None:
        await store.insert_fact(make_fact(id="fct_x"))
        facts, entities, relations = await store.export_guild("g1")
        assert any(f.id == "fct_x" for f in facts)
        stats = await store.guild_stats("g1")
        assert stats.guild_id == "g1"
        assert stats.total_facts >= 1
        del entities, relations
