from unittest.mock import AsyncMock

import pytest

from icelake import DiscordMemory
from icelake.config import MemoryConfig
from icelake.models.common import Page
from icelake.models.facts import FactCategory, SourceRef
from icelake.models.identity import AliasSource
from icelake.models.retrieval import RecallResult, RecallWarning
from icelake.permissions import AccessStatus, Audience, MemoryAccessError
from icelake.retrieval.injection import estimate_tokens
from tests.conftest import ScriptedLLM, extraction_response, make_config

GUILD = "500000000000000001"
USER = "100000000000000001"
OTHER = "100000000000000002"
CHANNEL = "700000000000000001"
AUDIENCE = Audience(channel_id=CHANNEL, requester_id=USER)


async def remember_name(client, now, *, uid=USER, channel=CHANNEL, mid="m1", name="Nolan"):
    return await client.facts.remember(
        guild_id=GUILD,
        subject_id=uid,
        actor_id=uid,
        text=f"member prefers to be called {name}",
        category=FactCategory.PERSONAL,
        source_refs=(
            SourceRef(
                guild_id=GUILD,
                channel_id=channel,
                message_id=mid,
                author_id=uid,
                author_name="discord_handle",
                content_snippet=f"call me {name}",
                created_at=now,
            ),
        ),
    )


async def test_names_survive_unrelated_topic_without_alias_or_quote_changes(
    make_client, fixed_clock, monkeypatch
):
    client, llm = make_client()
    await remember_name(client, fixed_clock.now())
    await remember_name(client, fixed_clock.now(), uid=OTHER, mid="m2", name="Klim")
    await client.identity.register_alias(
        GUILD, USER, "discord_handle", AliasSource.DISCORD_USERNAME
    )
    await client.identity.register_alias(GUILD, USER, "WrongName", AliasSource.REAL_NAME)
    monkeypatch.setattr(client, "recall", AsyncMock(return_value=RecallResult()))
    before = len(llm.calls)
    context = await client.prompt_context(
        guild_id=GUILD, asker_id=USER, mentioned_ids=(OTHER,), text="hello", audience=AUDIENCE
    )
    assert {n.user_id: n.name for n in context.names} == {USER: "Nolan", OTHER: "Klim"}
    assert "WrongName" not in context.injection_block
    assert "configured tone" in context.injection_block
    assert all(c.author_name == "discord_handle" for c in context.citations)
    assert {c.snippet for c in context.citations} == {"call me Nolan", "call me Klim"}
    assert await client.identity.display_name(GUILD, USER) == "discord_handle"
    assert len(llm.calls) == before
    assert context.usage.prompt_tokens == estimate_tokens(context.injection_block)


async def test_private_name_and_guessed_alias_never_leak(make_client, fixed_clock):
    client, _ = make_client()
    await remember_name(client, fixed_clock.now(), channel="private")
    await client.identity.register_alias(GUILD, USER, "Secret", AliasSource.REAL_NAME)
    result = await client.name_context(guild_id=GUILD, user_ids=(USER,), audience=AUDIENCE)
    assert result.injection_block == "" and not result.names and not result.citations
    assert result.warnings == (RecallWarning.ACCESS_RESTRICTED,)


@pytest.mark.parametrize("operation", ["forget", "delete", "optout", "correct"])
async def test_names_follow_fact_lifecycle(make_client, fixed_clock, operation):
    client, _ = make_client()
    fact = await remember_name(client, fixed_clock.now())
    assert (await client.name_context(guild_id=GUILD, user_ids=(USER,), audience=AUDIENCE)).names
    if operation == "forget":
        await client.facts.forget(fact.id, guild_id=GUILD)
    elif operation == "delete":
        await client.delete_source(GUILD, "m1")
    elif operation == "optout":
        await client.admin.set_opt_out(GUILD, USER, True)
    else:
        source = fact.citations[0].model_copy(
            update={"message_id": "m2", "content_snippet": "call me Nolo"}
        )
        await client.facts.update(
            fact.id, guild_id=GUILD, text="member prefers to be called Nolo", source_refs=(source,)
        )
    names = (await client.name_context(guild_id=GUILD, user_ids=(USER,), audience=AUDIENCE)).names
    assert [n.name for n in names] == (["Nolo"] if operation == "correct" else [])


async def test_required_audience_and_tiny_budget(make_client, fixed_clock):
    client, _ = make_client(config=make_config(access={"require_audience": True}))
    await remember_name(client, fixed_clock.now())
    missing = await client.name_context(guild_id=GUILD, user_ids=(USER,))
    assert missing.warnings == (RecallWarning.AUDIENCE_REQUIRED,)
    tiny = await client.prompt_context(
        guild_id=GUILD, asker_id=USER, text="hello", audience=AUDIENCE, token_budget_tokens=10
    )
    assert not tiny.names and estimate_tokens(tiny.injection_block) <= 10
    assert RecallWarning.BUDGET_TRIMMED in tiny.warnings


@pytest.mark.parametrize("category", ["personal", "general", "identity", None])
@pytest.mark.parametrize(
    ("statement", "wording"),
    [
        ("I go by Jo", "alice prefers to be called Jo starting today"),
        ("I go by Jo", "alice uses the handle Jo"),
        (
            "yeah, my name is Jo and I work with databases",
            "alice is named Jo and works with databases",
        ),
    ],
)
async def test_actual_ingestion_learns_name_without_unscoped_alias(
    make_client, event_factory, category, statement, wording
):
    llm = ScriptedLLM(
        {
            "extraction": extraction_response(
                [
                    {
                        "subject_token": "p0",
                        "text": wording,
                        "category": category,
                        "confidence": 0.95,
                        "source_message_indexes": [1],
                    }
                ]
            )
        }
    )
    config: MemoryConfig = make_config(access={"require_audience": True})
    client, _ = make_client(llm=llm, config=config)
    await client.observe(event_factory(content=statement))
    await client.flush()
    await client.observe(event_factory(content=statement))
    await client.flush()
    result = await client.name_context(guild_id=GUILD, user_ids=(USER,), audience=AUDIENCE)
    assert [n.name for n in result.names] == ["Jo"]
    assert result.facts[0].fact.expires_at is None
    assert result.facts[0].fact.occurrences == 2
    assert all(
        a.source is not AliasSource.REAL_NAME for a in await client.identity.aliases_of(GUILD, USER)
    )


async def test_name_projection_uses_every_storage_backend(store, fixed_clock, monkeypatch):
    client = DiscordMemory(make_config(), store=store, clock=fixed_clock, llm=ScriptedLLM())
    try:
        await remember_name(client, fixed_clock.now())
        result = await client.name_context(guild_id=GUILD, user_ids=(USER,), audience=AUDIENCE)
        assert [n.name for n in result.names] == ["Nolan"]
    finally:
        # The fixture must drop its disposable Mongo DB before closing its client.
        with monkeypatch.context() as owned_store:
            owned_store.setattr(store, "close", AsyncMock())
            await client.close(drain=False)


async def test_explicit_discord_name_withdrawal_reaches_prompt(make_client, fixed_clock):
    client, _ = make_client()
    fact = await remember_name(client, fixed_clock.now())
    source = fact.citations[0].model_copy(
        update={"message_id": "m2", "content_snippet": "use my Discord name"}
    )
    await client.facts.remember(
        guild_id=GUILD,
        subject_id=USER,
        text="member prefers their Discord name",
        source_refs=(source,),
    )
    result = await client.name_context(guild_id=GUILD, user_ids=(USER,), audience=AUDIENCE)
    assert len(result.names) == 1 and result.names[0].name is None
    assert '"name": null' in result.injection_block


async def test_access_failure_has_no_name_and_is_not_an_absence_claim(
    make_client, fixed_clock, monkeypatch
):
    client, _ = make_client()
    await remember_name(client, fixed_clock.now())
    monkeypatch.setattr(
        client, "visible_facts", AsyncMock(side_effect=MemoryAccessError(AccessStatus.UNAVAILABLE))
    )
    result = await client.name_context(guild_id=GUILD, user_ids=(USER,), audience=AUDIENCE)
    assert not result.names and not result.injection_block
    assert result.warnings == (RecallWarning.ACCESS_UNAVAILABLE,)


async def test_incomplete_profile_abstains_instead_of_using_old_name(
    make_client, fixed_clock, monkeypatch
):
    client, _ = make_client()
    fact = await remember_name(client, fixed_clock.now())
    read = AsyncMock(return_value=Page(items=(fact,) * 501, next_cursor=fact.id))
    monkeypatch.setattr(client.facts, "list_for_subject", read)
    result = await client.name_context(guild_id=GUILD, user_ids=(USER,) * 3, audience=AUDIENCE)
    assert not result.names and result.warnings == (RecallWarning.BUDGET_TRIMMED,)
    read.assert_awaited_once()
    assert read.call_args.kwargs["include_server"] is False


async def test_ingestion_keeps_full_name_separate_from_preference(event_factory, fixed_clock):
    from icelake.adapters.in_memory.vectors import InMemoryVectorIndex

    class CollidingEmbedder:
        dimensions = 2

        async def embed(self, texts):
            return tuple((1.0, 0.0) for _ in texts)

    responses = iter(["alice prefers Atlas", "alice's full name is Avery Example"])

    def extract(_request):
        return extraction_response(
            [
                {
                    "subject_token": "p0",
                    "text": next(responses),
                    "category": "general",
                    "confidence": 0.95,
                    "source_message_indexes": [1],
                }
            ]
        )

    # Deliberately wrong model decision: it must never get the opportunity to
    # replace a preferred form of address with a full-name statement.
    llm = ScriptedLLM(
        {
            "extraction": extract,
            "reconcile": '{"decisions":[{"candidate_index":0,"kind":"update","target_id":"0"}]}',
        }
    )
    async with DiscordMemory(
        make_config(),
        llm=llm,
        clock=fixed_clock,
        embedder=CollidingEmbedder(),
        vectors=InMemoryVectorIndex(),
    ) as client:
        for content in ("call me Atlas", "my full name is Avery Example"):
            fixed_clock.advance(1)
            await client.observe(event_factory(content=content))
            assert await client.flush() == 1
            await client.ops.recover()
        result = await client.name_context(guild_id=GUILD, user_ids=(USER,), audience=AUDIENCE)
        assert [n.name for n in result.names] == ["Atlas"]
        facts = await client.facts.list_for_subject(GUILD, USER, include_server=False)
        assert len(facts.items) == 2
        assert all(f.expires_at is None for f in facts.items)
        assert not any(call.purpose == "reconcile" for call in llm.calls)


async def test_migrated_profile_uses_confirmed_name_without_discord(make_client, fixed_clock):
    """Real incident shape: competing aliases, wrong old prose, no original name citation."""
    client, _ = make_client(config=make_config(access={"enabled": False}))
    for alias, source in (
        ("nulzo", AliasSource.DISCORD_USERNAME),
        ("Joshy Washy", AliasSource.REAL_NAME),
        ("Nolan", AliasSource.REAL_NAME),
    ):
        await client.identity.register_alias(GUILD, USER, alias, source)
    await client.facts.remember(guild_id=GUILD, subject_id=USER, text="Joshy Washy likes camping")
    fact = await client.facts.remember(
        guild_id=GUILD,
        subject_id=USER,
        actor_id=USER,
        text="call me Nolan",
        category=FactCategory.PERSONAL,
        idempotency_key="confirmed-name",
    )
    assert fact.expires_at is None
    assert not fact.citations
    for audience in (None, AUDIENCE, Audience(channel_id="different", requester_id=OTHER)):
        result = await client.prompt_context(
            guild_id=GUILD,
            asker_id=USER,
            text="hello",
            audience=audience,
        )
        assert [(n.user_id, n.name) for n in result.names] == [(USER, "Nolan")]
        assert not result.warnings
        assert '"name": "Nolan"' in result.injection_block
    other = await client.name_context(guild_id=GUILD, user_ids=(OTHER,))
    assert not other.names
    await client.admin.set_opt_out(GUILD, USER, True)
    assert not (await client.name_context(guild_id=GUILD, user_ids=(USER,))).names


async def test_conversational_correction_survives_ingestion(make_client, event_factory):
    llm = ScriptedLLM(
        {
            "extraction": extraction_response(
                [
                    {
                        "subject_token": "p0",
                        "text": "alice does not want to be called Joshy Washy",
                        "category": "personal",
                        "confidence": 0.95,
                        "source_message_indexes": [1],
                    }
                ]
            )
        }
    )
    client, _ = make_client(llm=llm, config=make_config(access={"enabled": False}))
    await client.observe(event_factory(content="I am not Joshy Washy, stop getting it wrong"))
    await client.flush()
    result = await client.name_context(guild_id=GUILD, user_ids=(USER,), audience=AUDIENCE)
    assert len(result.names) == 1 and result.names[0].name is None
    assert result.facts[0].fact.expires_at is None


async def test_inferred_alias_never_labels_new_facts(fixed_clock, event_factory):
    llm = ScriptedLLM(
        {
            "extraction": extraction_response(
                [
                    {
                        "subject_token": "p0",
                        "text": "p0 likes camping",
                        "confidence": 0.95,
                        "source_message_indexes": [1],
                    }
                ]
            )
        }
    )
    from icelake.adapters.in_memory.store import InMemoryStore

    store = InMemoryStore()
    async with DiscordMemory(
        make_config(access={"enabled": False}), store=store, llm=llm, clock=fixed_clock
    ) as client:
        await store.upsert_alias(GUILD, "joshy washy", USER, AliasSource.REAL_NAME, 3.0)
        await client.observe(event_factory(content="I like camping in national parks"))
        await client.flush()
        records = await client.facts.get_all(GUILD, USER)
        assert records and all("joshy" not in f.text.lower() for f in records)
        assert all("joshy" not in str(c).lower() for c in llm.calls)
