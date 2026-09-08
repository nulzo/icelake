"""Entity-centric reads: users_for_entity, shared_n, entity-seeded hop, twin collapse."""

from __future__ import annotations

import pytest

from icelake.models.graph import EntityKind
from icelake.models.identity import AliasSource
from icelake.models.operations import ProposedEntity, ProposedRelation
from icelake.models.retrieval import ChannelName, RecallQuery, Scope, channels

GUILD = "500000000000000001"
ALICE = "100000000000000001"
BOB = "200000000000000002"
CAROL = "300000000000000003"
DAVE = "400000000000000004"


@pytest.fixture()
async def golf_world(make_client):
    """alice likes golf; bob plays golf; carol only mentions it; dave golfs solo."""
    client, _ = make_client(llm=False)
    await client.start()
    await client.facts.remember(
        guild_id=GUILD,
        subject_id=ALICE,
        text="alice loves golf and plays every weekend",
        entities=(ProposedEntity(name="Golf", kind=EntityKind.CONCEPT),),
        relations=(ProposedRelation(verb="likes", from_token=ALICE, to_entity="Golf"),),
    )
    await client.facts.remember(
        guild_id=GUILD,
        subject_id=BOB,
        text="bob plays golf on sundays",
        entities=(ProposedEntity(name="Golf", kind=EntityKind.CONCEPT),),
        relations=(ProposedRelation(verb="plays", from_token=BOB, to_entity="Golf"),),
    )
    await client.facts.remember(
        guild_id=GUILD,
        subject_id=CAROL,
        text="carol walked past the golf course on tuesday",
        entities=(ProposedEntity(name="Golf", kind=EntityKind.CONCEPT),),
    )
    yield client
    await client.close()


class TestUsersForEntity:
    async def test_stance_buckets_and_mentions(self, golf_world) -> None:
        users = await golf_world.graph.users_for_entity(GUILD, "golf")
        assert users.entity_slug == "golf"
        assert users.positive == (ALICE,)
        assert BOB in users.other
        assert CAROL in users.mentioned
        assert CAROL not in users.positive + users.negative + users.other
        assert users.total_evidence >= 2

    async def test_unknown_entity_returns_empty(self, golf_world) -> None:
        users = await golf_world.graph.users_for_entity(GUILD, "curling")
        assert users.positive == ()
        assert users.mentioned == ()

    async def test_alias_surface_name_resolves(self, golf_world) -> None:
        users = await golf_world.graph.users_for_entity(GUILD, "Golf")
        assert users.positive == (ALICE,)


class TestSharedN:
    async def test_three_way_intersection(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        for user_id, name in ((ALICE, "alice"), (BOB, "bob"), (CAROL, "carol")):
            await client.facts.remember(
                guild_id=GUILD,
                subject_id=user_id,
                text=f"{name} plays chess weekly",
                entities=(ProposedEntity(name="Chess", kind=EntityKind.CONCEPT),),
                relations=(ProposedRelation(verb="plays", from_token=user_id, to_entity="Chess"),),
            )
        for user_id, name in ((ALICE, "alice"), (BOB, "bob")):
            await client.facts.remember(
                guild_id=GUILD,
                subject_id=user_id,
                text=f"{name} enjoys hiking",
                entities=(ProposedEntity(name="Hiking", kind=EntityKind.CONCEPT),),
                relations=(
                    ProposedRelation(verb="enjoys", from_token=user_id, to_entity="Hiking"),
                ),
            )
        edges_by_user = await client.graph.shared_n(GUILD, (ALICE, BOB, CAROL))
        assert len(edges_by_user) == 3
        for edges in edges_by_user:
            assert {edge.dst_id for edge in edges} == {"chess"}
        pair = await client.graph.shared_n(GUILD, (ALICE, BOB))
        assert {edge.dst_id for edge in pair[0]} == {"chess", "hiking"}
        await client.close()

    async def test_shared_attributions_matches_shared_n(self, golf_world) -> None:
        left, right = await golf_world.graph.shared_attributions(GUILD, ALICE, BOB)
        left_n, right_n = await golf_world.graph.shared_n(GUILD, (ALICE, BOB))
        assert {e.dst_id for e in left} == {e.dst_id for e in left_n} == {"golf"}
        assert {e.dst_id for e in right} == {e.dst_id for e in right_n}

    async def test_fewer_than_two_users_returns_empty(self, golf_world) -> None:
        assert await golf_world.graph.shared_n(GUILD, (ALICE,)) == ((),)


class TestEntitySeededRecall:
    async def test_entity_hint_hops_to_members(self, golf_world) -> None:
        """'golf' with no subjects: hop from the entity reaches members' facts."""
        result = await golf_world.recall(
            RecallQuery(
                guild_id=GUILD,
                entity_hint="golf",
                scope=Scope.GUILD,
                top_k=10,
                max_per_subject=10,
            )
        )
        subjects = {sf.fact.subject_id for sf in result.facts}
        assert {ALICE, BOB} <= subjects

    async def test_hop_path_records_provenance(self, golf_world) -> None:
        result = await golf_world.recall(
            RecallQuery(
                guild_id=GUILD,
                entity_hint="golf",
                scope=Scope.GUILD,
                channels=channels(ChannelName.GRAPH_HOP),
                top_k=10,
                max_per_subject=10,
            )
        )
        paths = [sf.hop_path for sf in result.facts if sf.hop_path]
        assert paths, "graph-hop facts must record the node path that surfaced them"
        assert all("entity:golf" in path for path in paths)

    async def test_unknown_entity_hint_matches_nothing(self, golf_world) -> None:
        """An unresolved hint contributes no entity-channel matches (baseline
        anchors still return — the hint is additive, never a filter)."""
        result = await golf_world.recall(
            RecallQuery(guild_id=GUILD, entity_hint="curling", scope=Scope.GUILD)
        )
        assert all(
            ChannelName.ENTITY not in sf.matched_channels
            and ChannelName.GRAPH_HOP not in sf.matched_channels
            for sf in result.facts
        )


class TestEntityNgramChannel:
    async def test_multiword_alias_matches_as_one_entity(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="alice hosts board games night",
            entities=(ProposedEntity(name="Board Games", kind=EntityKind.CONCEPT),),
        )
        result = await client.recall(
            RecallQuery(
                guild_id=GUILD,
                text="who hosts board games?",
                scope=Scope.GUILD,
                channels=channels(ChannelName.ENTITY),
            )
        )
        assert any("board games" in sf.fact.text for sf in result.facts)
        await client.close()


class TestTwinCollapseRecall:
    async def test_pair_intersection_includes_twin_facts(self, make_client) -> None:
        """A fact linked only to bob's entity twin still counts as bob's."""
        client, _ = make_client(llm=False)
        await client.start()
        record = await client.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="alice and bob won the trivia night together",
            entities=(ProposedEntity(name="Bob", kind=EntityKind.PERSON),),
        )
        await client._store.link_entity_to_user(GUILD, "bob", BOB)
        result = await client.recall(
            RecallQuery(
                guild_id=GUILD,
                subject_ids=(ALICE, BOB),
                scope=Scope.SUBJECTS,
                pair_ids=((ALICE, BOB),),
                channels=channels(ChannelName.LINKS),
            )
        )
        assert record.id in {sf.fact.id for sf in result.facts}
        await client.close()

    async def test_links_channel_reaches_twin_facts(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        record = await client.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="alice says dave is a brilliant chess coach",
            entities=(ProposedEntity(name="Dave", kind=EntityKind.PERSON),),
        )
        await client._store.link_entity_to_user(GUILD, "dave", DAVE)
        result = await client.recall(
            RecallQuery(
                guild_id=GUILD,
                subject_ids=(DAVE,),
                scope=Scope.SUBJECTS,
                channels=channels(ChannelName.LINKS),
            )
        )
        assert record.id in {sf.fact.id for sf in result.facts}
        await client.close()


class TestPersonEntityBridging:
    async def test_register_alias_bridges_person_entity(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="alice says dave collects vintage stamps",
            entities=(ProposedEntity(name="Dave", kind=EntityKind.PERSON),),
        )
        entity = await client._store.get_entity(GUILD, "dave")
        assert entity is not None and entity.linked_user_id is None
        await client.identity.register_alias(GUILD, DAVE, "dave", source=AliasSource.REAL_NAME)
        entity = await client._store.get_entity(GUILD, "dave")
        assert entity is not None and entity.linked_user_id == DAVE
        await client.close()

    async def test_concept_entity_is_never_bridged(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="alice loves golf",
            entities=(ProposedEntity(name="Golf", kind=EntityKind.CONCEPT),),
        )
        await client.identity.register_alias(GUILD, DAVE, "golf", source=AliasSource.REAL_NAME)
        entity = await client._store.get_entity(GUILD, "golf")
        assert entity is not None and entity.linked_user_id is None
        await client.close()

    async def test_ambiguous_alias_does_not_bridge(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD,
            subject_id=ALICE,
            text="alice says dave is tall",
            entities=(ProposedEntity(name="Dave", kind=EntityKind.PERSON),),
        )
        await client.identity.register_alias(GUILD, BOB, "dave", source=AliasSource.REAL_NAME)
        await client.identity.register_alias(GUILD, DAVE, "dave", source=AliasSource.REAL_NAME)
        entity = await client._store.get_entity(GUILD, "dave")
        assert entity is not None and entity.linked_user_id is None
        await client.close()
