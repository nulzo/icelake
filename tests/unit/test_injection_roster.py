"""Injection builder, roster, and recall-channel unit tests."""

from __future__ import annotations

from icelake.ingest.roster import Roster
from icelake.models.facts import FactCategory, FactRecord
from icelake.models.graph import EdgeKind, LinkRow, NodeType
from icelake.models.retrieval import ScoredFact
from icelake.retrieval.injection import (
    InjectionBuilder,
    estimate_tokens,
    message_url,
)
from tests.conftest import ScriptedLLM


def _fact(text: str, *, subject_id: str | None = "u1", with_source: bool = True) -> FactRecord:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return FactRecord(
        id=f"fct_{abs(hash(text)) % 10**8}",
        guild_id="g1",
        subject_id=subject_id,
        text=text,
        category=FactCategory.INTERESTS,
        created_at=now,
        updated_at=now,
        observed_at=now,
        valid_from=now,
        citations=(
            type(
                "R",
                (),
                {
                    "role": type("Role", (), {"value": "primary"})(),
                    "message_id": "m1",
                    "channel_id": "c1",
                    "message_url": "",
                    "content_snippet": "snippet text",
                },
            )(),  # duck-typed SourceRef substitute is NOT allowed by pydantic;
        )
        if False
        else (),
    )


class TestInjectionBuilder:
    def _scored(self, fact: FactRecord) -> ScoredFact:
        return ScoredFact(fact=fact, score=0.9)

    def test_sections_render_with_labels(self) -> None:
        builder = InjectionBuilder()
        asker_fact = _fact("likes building keyboards", subject_id="u-asker")
        other_fact = _fact("called the asker a hacker once", subject_id="u-other")
        block, citations, _trimmed = builder.build(
            asker_id="u-asker",
            facts_by_section={
                "asker": (self._scored(asker_fact),),
                "user:u-other": (self._scored(other_fact),),
            },
            summaries={"asker": None, "user:u-other": "a rival gamer"},
            token_budget=10_000,
            guild_id="g1",
        )
        assert "CURRENT ASKER" in block.upper()
        assert "REFERENCED USER" in block
        assert "a rival gamer" in block
        assert "- likes building keyboards" in block
        assert len(citations) == 0  # no real source refs → plain bullets, no citations

    def test_display_name_overrides_speaker_name(self) -> None:
        """A third-party fact (alice speaking about bob) must render under bob's
        name, never under alice's. The caller resolves the subject's name from
        the alias ladder and passes it in."""
        builder = InjectionBuilder()
        fact = _fact("was called a hacker by alice", subject_id="u-bob")
        # Simulate the fact carrying the SPEAKER's name (alice) — the bug.
        fact = fact.model_copy(
            update={"attribution": fact.attribution.model_copy(update={"speaker_name": "alice"})}
        )
        block, _citations, _trimmed = builder.build(
            asker_id="u-asker",
            facts_by_section={"user:u-bob": (self._scored(fact),)},
            summaries={},
            token_budget=10_000,
            guild_id="g1",
            display_names={"user:u-bob": "bob"},
        )
        assert "REFERENCED USER: bob" in block
        assert "REFERENCED USER: alice" not in block

    def test_budget_forces_trim(self) -> None:
        builder = InjectionBuilder()
        facts = tuple(
            self._scored(_fact(f"fact number {i} about something fairly long here"))
            for i in range(20)
        )
        block, _citations, trimmed = builder.build(
            asker_id="u1",
            facts_by_section={"asker": facts},
            summaries={"asker": None},
            token_budget=40,
            guild_id="g1",
        )
        assert trimmed or len(block) < 400


class TestRoster:
    def test_token_minting_is_stable_per_user(self) -> None:
        roster = Roster()
        first = roster.add("u1", "alice")
        again = roster.add("u1", "alice")
        assert first == again == "p0"

    def test_knows_and_lookup(self) -> None:
        roster = Roster()
        roster.add("u2", "bob")
        assert roster.knows("p0")
        assert roster.knows("server")
        assert not roster.knows("p9")
        assert roster.user_id_for("p0") == "u2"
        assert roster.name_for("p0") == "bob"

    def test_bind_names_rewrites_minted_tokens_only(self) -> None:
        roster = Roster()
        roster.add("u1", "alice")
        assert roster.bind_names("p0 loves Go") == "alice loves Go"
        assert roster.bind_names("p0's wife") == "alice's wife"
        assert roster.bind_names("p9 left town") == "p9 left town"
        assert roster.bind_names("the server loves Go") == "the server loves Go"
        assert roster.display_name("u1") == "alice"

    def test_render_lists_participants_and_server(self) -> None:
        roster = Roster()
        roster.add("u3", "carol")
        rendered = roster.render()
        assert "p0 = carol" in rendered
        assert "server" in rendered


class TestChannels:
    async def test_links_channel_cross_profile_reach(
        self,
        make_client,
        event_factory,
    ):
        client, _llm = make_client(
            llm=ScriptedLLM(
                {
                    "extraction": json_dumps(
                        {
                            "operations": [
                                {
                                    "subject_token": "p1",
                                    "speaker_token": "p0",
                                    "text": "bob was teased by alice during the game session",
                                    "category": "relationships",
                                    "confidence": 0.9,
                                    "source_message_indexes": [1],
                                }
                            ],
                        }
                    )
                }
            )
        )
        await client.start()
        event = event_factory(
            content="bob got completely destroyed in that ranked match yesterday",
            author_id="100000000000000001",
            mentions=("200000000000000002",),
            display_name="alice",
        )
        await client.observe(event)
        await client.flush()

        from icelake.retrieval.channels import links_channel

        output = await links_channel(
            store=client._store,
            guild_id=event.guild_id,
            subject_ids=("200000000000000002",),
            limit=50,
        )
        assert output.ranked_ids  # bob's incidence reaches alice's stored fact
        await client.close()

    async def test_vector_channel_empty_without_embedder(self):
        from icelake.retrieval.channels import vector_channel

        output = await vector_channel(
            vectors=None,
            embedder=None,
            guild_id="g",
            query_text="x",
            subject_ids=None,
            server_only=False,
            limit=5,
            candidate_cap=100,
        )
        assert output.ranked_ids == ()

    async def test_entity_channel_resolves_slugs(self, make_client):
        client, _ = make_client(llm=False)
        await client.start()
        await client._store.upsert_entity("g1", "chess", "Chess", "concept", aliases=("chess",))
        record = await client.facts.remember(
            guild_id="g1", subject_id="u1", text="plays chess every single weekend"
        )
        await client._store.add_links(
            (
                LinkRow(
                    guild_id="g1",
                    memory_id=record.id,
                    node_type=NodeType.ENTITY,
                    node_id="chess",
                    kind=EdgeKind.ABOUT_ENTITY,
                ),
            )
        )
        from icelake.retrieval.channels import entity_channel

        output = await entity_channel(
            store=client._store, guild_id="g1", query_text="who plays chess", limit=10
        )
        assert record.id in output.ranked_ids
        await client.close()


class TestMiscUtils:
    def test_message_url_sentinel_channel(self) -> None:
        url = message_url("g", "", "m")
        assert "/g/0/m" in url

    def test_estimate_tokens_positive(self) -> None:
        assert estimate_tokens("") >= 1
        assert estimate_tokens("x" * 400) == 100


import json  # noqa: E402


def json_dumps(payload) -> str:
    return json.dumps(payload)
