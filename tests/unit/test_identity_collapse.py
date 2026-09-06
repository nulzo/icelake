"""Identity-collapse: member-named relation endpoints become user nodes.

The bug: extraction minted an entity twin for anyone not in the current
batch, so ``nulzo -friend_of-> ENTITY:klim`` never surfaced as a person
edge. These tests pin the fix at both seams — write (collapse before
minting) and read (follow ``linked_user_id``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from icelake.adapters.in_memory.store import InMemoryStore
from icelake.api.groups import GraphApi
from icelake.config import MemoryConfig
from icelake.ingest.executor import FactCommitter
from icelake.ingest.roster import Roster
from icelake.models.graph import NodeType
from icelake.models.identity import AliasSource
from icelake.models.operations import ProposedFact, ProposedRelation
from icelake.ports.clock import FixedClock, UlidIdGen

GUILD = "g1"
ALICE = "u-alice"
BOB = "u-bob"


def _proposal(**overrides) -> ProposedFact:
    values = {
        "subject_token": "p0",
        "text": "alice is friends with bob",
        "category": "relationships",
        "confidence": 0.9,
        "source_message_indexes": [1],
    }
    values.update(overrides)
    return ProposedFact(**values)


@pytest.fixture()
async def store():
    backend = InMemoryStore()
    await backend.setup()
    # bob is a known member: the identity ladder resolves "bob" to him.
    await backend.upsert_alias(GUILD, "bob", BOB, AliasSource.DISCORD_USERNAME, 1.0)
    return backend


@pytest.fixture()
def committer(store):
    return FactCommitter(
        store=store,
        vectors=None,
        embedder=None,
        clock=FixedClock(datetime(2026, 8, 24, tzinfo=UTC)),
        id_gen=UlidIdGen(),
        config=MemoryConfig(),
    )


class TestWriteCollapse:
    async def test_member_name_becomes_user_node(self, store, committer) -> None:
        roster = Roster()
        roster.add(ALICE, "alice")
        record = await committer.commit_add(
            proposal=_proposal(
                relations=[
                    ProposedRelation(verb="friend_of", from_token="p0", to_entity="bob"),
                ],
            ),
            subject_id=ALICE,
            speaker_id=None,
            guild_id=GUILD,
            roster=roster,
        )
        edges = await store.edges_between(GUILD, (NodeType.USER, ALICE), (NodeType.USER, BOB))
        assert len(edges) == 1
        assert edges[0].verb == "friend_of"
        # No entity twin was minted for bob.
        assert await store.get_entity(GUILD, "bob") is None
        assert record.subject_id == ALICE

    async def test_unknown_name_stays_entity(self, store, committer) -> None:
        roster = Roster()
        roster.add(ALICE, "alice")
        await committer.commit_add(
            proposal=_proposal(
                relations=[
                    ProposedRelation(verb="likes", from_token="p0", to_entity="Rust"),
                ],
            ),
            subject_id=ALICE,
            speaker_id=None,
            guild_id=GUILD,
            roster=roster,
        )
        entity = await store.get_entity(GUILD, "rust")
        assert entity is not None
        edges = await store.edges_between(GUILD, (NodeType.USER, ALICE), (NodeType.ENTITY, "rust"))
        assert len(edges) == 1

    async def test_ambiguous_name_stays_entity(self, store, committer) -> None:
        # Two members match "bo" — never guess.
        await store.upsert_alias(GUILD, "bobby", "u-bobby", AliasSource.DISPLAY_NAME, 0.7)
        roster = Roster()
        roster.add(ALICE, "alice")
        await committer.commit_add(
            proposal=_proposal(
                relations=[
                    ProposedRelation(verb="knows", from_token="p0", to_entity="bo"),
                ],
            ),
            subject_id=ALICE,
            speaker_id=None,
            guild_id=GUILD,
            roster=roster,
        )
        # "bo" is ambiguous (bob + bobby); it must remain an entity.
        assert await store.get_entity(GUILD, "bo") is not None


class TestReadCollapse:
    async def test_relations_of_follows_linked_user_id(self, store) -> None:
        # Legacy shape: an entity twin for bob with a linked user id.
        await store.upsert_entity(GUILD, "bob", "bob", "person")
        await store.link_entity_to_user(GUILD, "bob", BOB)
        from icelake.models.graph import RelationEdge

        await store.upsert_relation(
            RelationEdge(
                guild_id=GUILD,
                src_type=NodeType.USER,
                src_id=ALICE,
                dst_type=NodeType.ENTITY,
                dst_id="bob",
                verb="friend_of",
                weight=0.9,
            )
        )
        graph = GraphApi(store=store)
        edges = await graph.relations_of(GUILD, ALICE)
        assert len(edges) == 1
        assert edges[0].dst_type is NodeType.USER
        assert edges[0].dst_id == BOB

    async def test_between_collapses_both_directions(self, store) -> None:
        await store.upsert_entity(GUILD, "bob", "bob", "person")
        await store.link_entity_to_user(GUILD, "bob", BOB)
        from icelake.models.graph import RelationEdge

        await store.upsert_relation(
            RelationEdge(
                guild_id=GUILD,
                src_type=NodeType.USER,
                src_id=ALICE,
                dst_type=NodeType.ENTITY,
                dst_id="bob",
                verb="friend_of",
                weight=0.9,
            )
        )
        graph = GraphApi(store=store)
        edges = await graph.between(GUILD, ALICE, BOB)
        assert len(edges) == 1
        assert edges[0].dst_id == BOB

    async def test_shared_intersects_collapsed_entities(self, store) -> None:
        from icelake.models.graph import RelationEdge

        for uid, slug in ((ALICE, "rust"), (BOB, "rust"), (ALICE, "go")):
            await store.upsert_entity(GUILD, slug, slug, "concept")
            await store.upsert_relation(
                RelationEdge(
                    guild_id=GUILD,
                    src_type=NodeType.USER,
                    src_id=uid,
                    dst_type=NodeType.ENTITY,
                    dst_id=slug,
                    verb="likes",
                    weight=0.8,
                )
            )
        graph = GraphApi(store=store)
        shared = await graph.shared(GUILD, ALICE, BOB)
        assert [e.dst_id for e in shared] == ["rust"]

    async def test_shared_ignores_member_entities(self, store) -> None:
        # "bob" as an entity twin must not count as a shared interest.
        await store.upsert_entity(GUILD, "bob", "bob", "person")
        await store.link_entity_to_user(GUILD, "bob", BOB)
        from icelake.models.graph import RelationEdge

        await store.upsert_relation(
            RelationEdge(
                guild_id=GUILD,
                src_type=NodeType.USER,
                src_id=ALICE,
                dst_type=NodeType.ENTITY,
                dst_id="bob",
                verb="friend_of",
                weight=0.9,
            )
        )
        graph = GraphApi(store=store)
        shared = await graph.shared(GUILD, ALICE, BOB)
        assert shared == ()


class TestBotGuard:
    async def test_bots_never_land_in_related_user_ids(self, store) -> None:
        from icelake.identity.guards import BotGuard

        guard = BotGuard()
        guard.register("bot-1")
        committer = FactCommitter(
            store=store,
            vectors=None,
            embedder=None,
            clock=FixedClock(datetime(2026, 8, 24, tzinfo=UTC)),
            id_gen=UlidIdGen(),
            config=MemoryConfig(),
            bot_guard=guard,
        )
        roster = Roster()
        roster.add(ALICE, "alice")
        record = await committer.commit_add(
            proposal=_proposal(),
            subject_id=ALICE,
            speaker_id=None,
            guild_id=GUILD,
            roster=roster,
            mentioned_ids=("bot-1", BOB),
        )
        assert "bot-1" not in record.related_user_ids
        assert BOB in record.related_user_ids
