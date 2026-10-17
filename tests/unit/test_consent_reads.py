from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from icelake.adapters.in_memory.store import InMemoryStore
from icelake.api.facts_api import FactsApi
from icelake.api.groups import AdminApi, GraphApi
from icelake.config import MemoryConfig
from icelake.errors import FactNotFoundError
from icelake.identity.guards import BotGuard, ConsentPolicy, SubjectGate
from icelake.models.facts import Attribution, FactRecord, SourceRef
from icelake.models.graph import EntityKind
from icelake.models.operations import ProposedEntity, ProposedRelation
from icelake.ports.clock import SystemClock, UlidIdGen


@pytest.fixture
async def store() -> AsyncIterator[InMemoryStore]:
    backend = InMemoryStore()
    await backend.setup()
    try:
        yield backend
    finally:
        await backend.close()


def facts_api(store: InMemoryStore) -> FactsApi:
    return FactsApi(
        store=store,
        vectors=None,
        embedder=None,
        clock=SystemClock(),
        id_gen=UlidIdGen(),
        config=MemoryConfig(),
        subject_gate=SubjectGate(BotGuard(), ConsentPolicy(store)),
    )


@pytest.mark.parametrize("participant", ["subject", "speaker", "actor", "related", "citation"])
async def test_fact_reads_hide_any_opted_out_participant(
    store: InMemoryStore, participant: str
) -> None:
    record = FactRecord(
        id="fact",
        guild_id="g",
        subject_id="subject",
        text="enjoys golf",
        attribution=Attribution(speaker_id="speaker", actor_id="actor"),
        related_user_ids=("related",),
        citations=(SourceRef(message_id="m", channel_id="c", guild_id="g", author_id="citation"),),
    )
    await store.insert_fact(record)
    facts = facts_api(store)
    admin = AdminApi(store)
    await admin.set_opt_out("g", participant, True)
    with pytest.raises(FactNotFoundError):
        await facts.get("g", record.id)
    with pytest.raises(FactNotFoundError):
        await facts.history(record.id, guild_id="g")
    assert not (await facts.list_for_subject("g", "subject")).items
    assert not await facts.search("g", "golf")
    await admin.set_opt_out("g", participant, False)
    assert await facts.get("g", record.id) == record
    assert (await facts.list_for_subject("g", "subject")).items == (record,)
    assert (await facts.search("g", "golf"))[0][0] == record


@pytest.mark.parametrize("participant", ["subject", "speaker"])
async def test_graph_reads_hide_opted_out_participants_and_restore(
    store: InMemoryStore, participant: str
) -> None:
    facts = facts_api(store)
    await facts.remember(
        guild_id="g",
        subject_id="subject",
        speaker_id="speaker",
        text="enjoys golf every weekend",
        entities=(ProposedEntity(name="Golf", kind=EntityKind.CONCEPT),),
        relations=(ProposedRelation(verb="likes", from_token="subject", to_entity="Golf"),),
    )
    graph = GraphApi(store=store)
    admin = AdminApi(store)
    assert await graph.relations_of("g", "subject")
    await admin.set_opt_out("g", participant, True)
    assert not await graph.relations_of("g", "subject")
    assert not (await graph.entity_stances("g", "golf")).positive
    users = await graph.users_for_entity("g", "golf")
    assert not users.positive
    assert not users.mentioned
    await admin.set_opt_out("g", participant, False)
    assert await graph.relations_of("g", "subject")
    users = await graph.users_for_entity("g", "golf")
    assert users.positive == ("subject",)
    assert set(users.mentioned) == {"subject", "speaker"}
