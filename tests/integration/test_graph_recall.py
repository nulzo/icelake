"""Graph-hop recall channel + similar_users discovery tests."""

from __future__ import annotations

import pytest

from discord_memory.models.retrieval import (
    CHANNELS_DISCOVERY,
    RecallQuery,
    channels,
)
from tests.conftest import ScriptedLLM, extraction_response

GUILD = "500000000000000001"
ALICE = "100000000000000001"
BOB = "200000000000000002"
CAROL = "300000000000000003"


@pytest.fixture()
async def linked_world(make_client, event_factory):
    """alice—likes→movies←dislikes—bob; bob—friend_of→carol."""
    llm = ScriptedLLM(
        {
            "extraction": extraction_response(
                [
                    {
                        "subject_token": "p0",
                        "text": "alice absolutely loves watching movies",
                        "category": "interests",
                        "confidence": 0.9,
                        "source_message_indexes": [1],
                        "entities": [{"name": "Movies", "kind": "concept"}],
                        "relations": [{"verb": "likes", "from_token": "p0", "to_entity": "Movies"}],
                    },
                    {
                        "subject_token": "server",
                        "speaker_token": None,
                        "text": "bob dislikes movies with a passion shared by nobody else",
                        "category": "interests",
                        "confidence": 0.9,
                        "source_message_indexes": [1],
                        "entities": [{"name": "Movies", "kind": "concept"}],
                        "relations": [
                            {"verb": "dislikes", "from_token": "server", "to_entity": "Movies"}
                        ],
                    },
                ]
            )
        }
    )
    client, _ = make_client(llm=llm)
    await client.start()
    await client.observe(
        event_factory(
            content="movie night was incredible i love watching films every weekend",
            author_id=ALICE,
            display_name="alice",
        )
    )
    # second message from bob (mentions alice) so both land in the batch window
    await client.observe(
        event_factory(
            content="honestly i dislike most films they are just too long for me",
            author_id=BOB,
            display_name="bob",
        )
    )
    await client.flush()

    await client.facts.remember(
        guild_id=GUILD,
        subject_id=BOB,
        text="bob considers carol one of his best friends forever",
    )
    yield client
    await client.close()


class TestGraphHopChannel:
    async def test_hop_channel_surfaces_neighbor_facts(self, linked_world) -> None:
        result = await linked_world.recall(
            RecallQuery(
                guild_id=GUILD,
                text="movies",
                subject_ids=(ALICE,),
                scope=__import__(
                    "discord_memory.models.retrieval", fromlist=["Scope"]
                ).Scope.SUBJECTS,
                channels=frozenset(
                    {
                        __import__(
                            "discord_memory.models.retrieval", fromlist=["ChannelName"]
                        ).ChannelName.GRAPH_HOP
                    }
                ),
            )
        )
        texts = {sf.fact.text for sf in result.facts}
        assert any("dislikes" in text or "movies" in text for text in texts)


class TestSimilarUsers:
    async def test_shared_entity_trait_ranks(self, linked_world) -> None:
        similar = await linked_world.graph.similar_users(GUILD, ALICE)
        ids = [user_id for user_id, _score in similar]
        assert BOB in ids
        top_score = similar[0][1]
        assert 0 < top_score <= 1.0

    async def test_isolated_user_has_no_similar(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        assert await client.graph.similar_users(GUILD, CAROL) == ()
        await client.close()


class TestDiscoveryChannelSet:
    async def test_discovery_includes_hop(self) -> None:
        from discord_memory.models.retrieval import ChannelName

        assert ChannelName.GRAPH_HOP in CHANNELS_DISCOVERY
        custom = channels(ChannelName.GRAPH_HOP)
        assert ChannelName.GRAPH_HOP in custom
