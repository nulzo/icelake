from __future__ import annotations

from datetime import UTC, datetime

import pytest

from icelake.adapters.in_memory.store import InMemoryStore
from icelake.config import MemoryConfig
from icelake.ingest.executor import FactCommitter
from icelake.models.graph import EdgeKind, NodeType
from icelake.models.operations import ProposedEntity, ProposedFact, ProposedRelation
from icelake.ports.clock import FixedClock, UlidIdGen


def _proposal(**overrides) -> ProposedFact:
    values = {
        "subject_token": "p0",
        "text": "alice plays the violin in a local orchestra",
        "category": "interests",
        "confidence": 0.9,
        "source_message_indexes": [1],
    }
    values.update(overrides)
    return ProposedFact(**values)


@pytest.fixture()
async def committer():
    store = InMemoryStore()
    config = MemoryConfig()
    return (
        FactCommitter(
            store=store,
            vectors=None,
            embedder=None,
            clock=FixedClock(datetime(2026, 8, 24, tzinfo=UTC)),
            id_gen=UlidIdGen(),
            config=config,
        ),
        store,
    )


class TestCommitAdd:
    async def test_personal_fact_with_entity_and_relation(
        self,
        committer,
    ) -> None:
        from icelake.ingest.roster import Roster

        roster = Roster()
        roster.add("u-alice", "alice")
        roster.add("u-bob", "bob")
        commit, store = committer
        proposal = _proposal(
            entities=[
                ProposedEntity(name="Violin", kind="concept"),
            ],
            relations=[
                ProposedRelation(verb="plays", from_token="p0", to_entity="Violin"),
            ],
        )
        record = await commit.commit_add(
            proposal=proposal,
            subject_id="u-alice",
            speaker_id=None,
            guild_id="g1",
            roster=roster,
        )
        assert record.subject_id == "u-alice"
        assert record.entity_slugs == ()  # slugs tracked via links not the record field
        links = await store.nodes_for_fact("g1", record.id)
        kinds = {link.kind for link in links}
        assert EdgeKind.SUBJECT_OF in kinds
        assert EdgeKind.ABOUT_ENTITY in kinds
        assert record.attribution.speaker_name == "alice"
        entity = await store.get_entity("g1", "violin")
        assert entity is not None
        edges = await store.edges_between(
            "g1",
            (NodeType.USER, "u-alice"),
            (NodeType.ENTITY, "violin"),
        )
        assert any(edge.verb == "plays" for edge in edges)

    async def test_third_party_attribution_and_speaker_link(
        self,
        committer,
    ) -> None:
        from icelake.ingest.roster import Roster

        roster = Roster()
        roster.add("u-speaker", "speaker")
        roster.add("u-target", "target")
        commit, store = committer
        record = await commit.commit_add(
            proposal=_proposal(subject_token="p1", text="target was called out publicly today"),
            subject_id="u-target",
            speaker_id="u-speaker",
            guild_id="g1",
            roster=roster,
        )
        assert record.attribution.type.value == "third_party"
        assert record.attribution.speaker_id == "u-speaker"
        assert record.attribution.speaker_name == "speaker"
        links = await store.nodes_for_fact("g1", record.id)
        kinds = {(link.node_type, link.kind) for link in links}
        assert (NodeType.USER, EdgeKind.SPEAKER_OF) in kinds

    async def test_server_fact_scope(self, committer) -> None:
        from icelake.ingest.roster import Roster

        commit, store = committer
        record = await commit.commit_add(
            proposal=_proposal(
                subject_token="server", text="the community loves strategy tournaments"
            ),
            subject_id=None,
            speaker_id=None,
            guild_id="g1",
            roster=Roster(),
        )
        assert record.is_server_fact
        del store

    async def test_duplicate_relations_merge_into_single_edge(self, committer):
        from icelake.ingest.roster import Roster

        roster = Roster()
        roster.add("ua", "a")
        roster.add("ub", "b")
        commit, store = committer
        for _ in range(2):
            await commit.commit_add(
                proposal=_proposal(
                    subject_token="p0",
                    text=f"teammate bond between them number {_}",
                    relations=[ProposedRelation(verb="friend_of", from_token="p0", to_token="p1")],
                ),
                subject_id="ua",
                speaker_id=None,
                guild_id="g1",
                roster=roster,
            )
        edges = await store.edges_between("g1", (NodeType.USER, "ua"), (NodeType.USER, "ub"))
        friend_edges = [e for e in edges if e.verb == "friend_of"]
        assert len(friend_edges) == 1
        assert friend_edges[0].occurrences == 2

    async def test_relation_with_unknown_tokens_skipped(self, committer):
        from icelake.ingest.roster import Roster

        commit, store = committer
        await commit.commit_add(
            proposal=_proposal(
                text="mystery relation nobody can verify here",
                relations=[ProposedRelation(verb="knows", from_token="pX", to_token="pY")],
            ),
            subject_id="ua",
            speaker_id=None,
            guild_id="g1",
            roster=Roster(),
        )
        stats = await store.guild_stats("g1")
        assert stats.relation_count == 0

    async def test_supersede_drops_old_fact_edge_evidence(self, committer):
        """A superseded fact must stop holding its relation edges alive."""
        from icelake.ingest.roster import Roster

        roster = Roster()
        roster.add("ua", "a")
        roster.add("ub", "b")
        commit, store = committer
        old = await commit.commit_add(
            proposal=_proposal(
                text="ua mentors ub on weekends",
                relations=[ProposedRelation(verb="mentors", from_token="p0", to_token="p1")],
            ),
            subject_id="ua",
            speaker_id=None,
            guild_id="g1",
            roster=roster,
        )
        edges = await store.edges_between("g1", (NodeType.USER, "ua"), (NodeType.USER, "ub"))
        assert any(e.verb == "mentors" for e in edges)

        await commit.commit_supersede(
            old_record=old,
            proposal=_proposal(text="ua no longer mentors ub"),
            subject_id="ua",
            speaker_id=None,
            reason="contradicted",
            guild_id="g1",
            roster=roster,
        )
        edges_after = await store.edges_between(
            "g1", (NodeType.USER, "ua"), (NodeType.USER, "ub")
        )
        assert not [e for e in edges_after if e.verb == "mentors"]

    async def test_commit_add_binds_roster_tokens_in_text(self, committer) -> None:
        from icelake.ingest.roster import Roster

        roster = Roster()
        roster.add("u-alice", "alice")
        commit, _store = committer
        record = await commit.commit_add(
            proposal=_proposal(text="p0 loves Red Bull and Go"),
            subject_id="u-alice",
            speaker_id=None,
            guild_id="g1",
            roster=roster,
        )
        assert record.text == "alice loves Red Bull and Go"
        assert "p0" not in record.text_normalized


class TestReinforceAndTransitions:
    async def test_commit_reinforce_bumps_strength(self, committer):
        from icelake.ingest.roster import Roster

        commit, _store = committer
        record = await commit.commit_add(
            proposal=_proposal(),
            subject_id="u-alice",
            speaker_id=None,
            guild_id="g1",
            roster=Roster(),
        )
        updated = await commit.commit_reinforce(record, _proposal())
        assert updated.strength > record.strength
        assert updated.occurrences == record.occurrences + 1

    async def test_commit_supersede_links_history(self, committer):
        from icelake.ingest.roster import Roster

        roster = Roster()
        roster.add("u-alice", "alice")
        commit, store = committer
        old = await commit.commit_add(
            proposal=_proposal(),
            subject_id="u-alice",
            speaker_id=None,
            guild_id="g1",
            roster=roster,
        )
        new, _old = await commit.commit_supersede(
            old_record=old,
            proposal=_proposal(text="alice performs first violin in the city orchestra"),
            subject_id="u-alice",
            speaker_id=None,
            reason="refined",
            guild_id="g1",
            roster=roster,
        )
        assert new.supersedes_id == old.id
        reloaded_old = await store.get_fact("g1", old.id)
        assert reloaded_old.superseded_by_id == new.id

    async def test_commit_supersede_keeps_citations(self, committer) -> None:
        from icelake.ingest.roster import Roster
        from icelake.models.facts import SourceRef

        roster = Roster()
        roster.add("u-alice", "alice")
        commit, _store = committer
        old = await commit.commit_add(
            proposal=_proposal(),
            subject_id="u-alice",
            speaker_id=None,
            guild_id="g1",
            roster=roster,
        )
        refs = (
            SourceRef(
                message_id="m1",
                channel_id="c1",
                guild_id="g1",
                author_id="u-alice",
                content_snippet="alice plays first violin now",
            ),
        )
        new, _old = await commit.commit_supersede(
            old_record=old,
            proposal=_proposal(text="alice performs first violin in the city orchestra"),
            subject_id="u-alice",
            speaker_id=None,
            reason="refined",
            guild_id="g1",
            roster=roster,
            source_refs=refs,
            mentioned_ids=("u-bob",),
        )
        assert new.citations == refs
        assert "u-bob" in new.related_user_ids

    async def test_commit_invalidate_detaches_evidence(self, committer):
        from icelake.ingest.roster import Roster

        roster = Roster()
        roster.add("u-alice", "alice")
        commit, _store = committer
        record = await commit.commit_add(
            proposal=_proposal(
                entities=[ProposedEntity(name="Tea")],
                relations=[ProposedRelation(verb="likes", from_token="p0", to_entity="Tea")],
            ),
            subject_id="u-alice",
            speaker_id=None,
            guild_id="g1",
            roster=roster,
        )
        invalidated = await commit.commit_invalidate(old_record=record, reason="retracted")
        assert invalidated.valid_until is not None
