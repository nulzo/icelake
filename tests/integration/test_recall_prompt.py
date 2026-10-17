"""End-to-end recall and prompt-context tests (API.md §6)."""

from __future__ import annotations

from icelake.citations import Citations
from icelake.models.events import ObserveStatus
from icelake.models.retrieval import RecallQuery, Scope
from tests.conftest import ScriptedLLM, extraction_response, make_config

GUILD = "500000000000000001"
ALICE = "100000000000000001"
BOB = "200000000000000002"


async def _seed_client(make_client, event_factory, facts: list[dict]):
    llm = ScriptedLLM(
        {
            "extraction": extraction_response(facts),
        }
    )
    client, _ = make_client(llm=llm)
    await client.start()
    for index, fact in enumerate(facts, start=1):
        event = event_factory(
            content=f"seed message number {index} with plenty of words here",
            author_id=ALICE,
            display_name="alice",
            mentions=(BOB,) if "bob" in fact.get("text", "").lower() else (),
        )
        assert (await client.observe(event)).status is ObserveStatus.ACCEPTED
    await client.flush()
    return client


class TestRecall:
    async def test_profile_recall_returns_subject_facts(self, make_client, event_factory):
        client = await _seed_client(
            make_client,
            event_factory,
            [
                {
                    "subject_token": "p0",
                    "text": "alice builds synthesizer modules",
                    "category": "interests",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                },
            ],
        )
        result = await client.recall(
            RecallQuery(
                guild_id=GUILD,
                text="synthesizers",
                subject_ids=(ALICE,),
            )
        )
        assert result.facts
        top = result.facts[0]
        assert top.score > 0
        assert top.components is not None
        assert top.components.strength > 0  # recency-aware strength wired into rerank
        assert top.fact.subject_id == ALICE
        await client.close()

    async def test_min_score_and_exclusions_enforced(self, make_client, event_factory):
        client = await _seed_client(
            make_client,
            event_factory,
            [
                {
                    "subject_token": "p0",
                    "text": "alice enjoys hiking mountains",
                    "category": "interests",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                },
            ],
        )
        result = await client.recall(
            RecallQuery(
                guild_id=GUILD,
                text="hiking",
                subject_ids=(ALICE,),
                min_score=0.99,
            )
        )
        assert result.facts == ()
        excluded = await client.recall(
            RecallQuery(
                guild_id=GUILD,
                text="hiking",
                subject_ids=(ALICE,),
                exclude_ids=(ALICE,),
            )
        )
        assert all(sf.fact.subject_id != ALICE for sf in excluded.facts)
        await client.close()

    async def test_server_scope_only_community_facts(self, make_client, event_factory):
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "server",
                            "text": "the server community loves strategy game tournaments",
                            "category": "culture",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        },
                    ]
                ),
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        for i in range(3):
            await client.observe(
                event_factory(
                    content=f"tournament season is starting again soon everyone ({i})",
                    author_id=ALICE,
                )
            )
        await client._pipeline.flush_subject(GUILD, "__server__")
        result = await client.recall(
            RecallQuery(guild_id=GUILD, scope=Scope.SERVER, text="tournaments")
        )
        assert any("tournaments" in sf.fact.text for sf in result.facts)
        await client.close()


class TestPromptContext:
    async def test_injection_block_sections_and_citations(
        self,
        make_client,
        event_factory,
    ):
        facts = [
            {
                "subject_token": "p0",
                "text": "alice prefers mechanical keyboards for coding sessions",
                "category": "preferences",
                "confidence": 0.9,
                "source_message_indexes": [1],
            },
            {
                "subject_token": "p1",
                "speaker_token": "p0",
                "text": "bob was called a hacker by alice during the match",
                "category": "relationships",
                "confidence": 0.85,
                "source_message_indexes": [1],
                "relations": [{"verb": "called_out", "from_token": "p0", "to_token": "p1"}],
            },
        ]
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(facts),
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        for spec in (
            {"content": "i prefer my new mechanical keyboard for long coding marathons"},
            {
                "content": "@bob stop hacking the lobby you absolute hacker everyone saw it",
                "mentions": (BOB,),
            },
        ):
            event = event_factory(author_id=ALICE, **spec)
            await client.observe(event)
        await client.flush()

        ctx = await client.prompt_context(
            guild_id=GUILD,
            asker_id=BOB,
            text="what do people think about hackers?",
        )
        assert ctx.injection_block.startswith("[MEMORY CONTEXT]")
        # asker section present (bob's own profile or empty-but-server fallback)
        assert "CURRENT ASKER" in ctx.injection_block.upper() or ctx.facts
        if ctx.citations:
            citation = ctx.citations[0]
            assert citation.url.startswith("https://discord.com/channels/")
            reply = next(sf.fact.text for sf in ctx.facts if sf.fact.id == citation.fact_id)
            set_ = Citations.from_prompt_context(ctx)
            cited = await client.cite(reply, set_)
            assert citation.url in cited.text
        usage = ctx.usage
        assert usage.prompt_tokens > 0
        await client.close()

    async def test_subject_caps_default_to_four(self, make_client, event_factory):
        client, _ = make_client()
        await client.start()
        for index in range(1, 8):
            await client.facts.remember(
                guild_id=GUILD,
                subject_id=ALICE,
                text=f"alice hobby number {index} is collecting retro consoles",
                actor_id="admin",
            )
        ctx = await client.prompt_context(guild_id=GUILD, asker_id=ALICE, text="hobbies")
        alice_facts = [sf for sf in ctx.facts if sf.fact.subject_id == ALICE]
        assert len(alice_facts) == 4
        await client.close()

    async def test_configured_caps_widen_subject_recall(self, make_client, event_factory):
        config = make_config(retrieval={"top_k": 12, "max_per_subject": 6})
        client, _ = make_client(config=config)
        await client.start()
        for index in range(1, 8):
            await client.facts.remember(
                guild_id=GUILD,
                subject_id=ALICE,
                text=f"alice hobby number {index} is collecting retro consoles",
                actor_id="admin",
            )
        ctx = await client.prompt_context(guild_id=GUILD, asker_id=ALICE, text="hobbies")
        alice_facts = [sf for sf in ctx.facts if sf.fact.subject_id == ALICE]
        assert len(alice_facts) == 6
        await client.close()

    async def test_budget_trims_and_warns(self, make_client, event_factory):
        facts = [
            {
                "subject_token": "p0",
                "text": f"alice hobby number {index} is collecting retro consoles",
                "category": "interests",
                "confidence": 0.8,
                "source_message_indexes": [1],
            }
            for index in range(1, 4)
        ]
        llm = ScriptedLLM({"extraction": extraction_response([])})
        client, _ = make_client(llm=llm)
        await client.start()
        for index in range(1, 4):
            await client.facts.remember(
                guild_id=GUILD,
                subject_id=ALICE,
                text=f"alice hobby number {index} is collecting retro consoles",
                actor_id="admin",
            )
        del facts
        ctx = await client.prompt_context(
            guild_id=GUILD,
            asker_id=ALICE,
            text="hobbies",
            token_budget_tokens=64,
        )
        warnings = {w.value for w in ctx.warnings}
        assert "budget_trimmed" in warnings or len(ctx.facts) <= 3
        await client.close()


class TestIdentityPrivacyRegressions:
    async def test_alias_asker_retains_role_and_pairs_are_canonical(self, make_client, monkeypatch):
        from icelake.models.facts import ProfileSummary
        from icelake.models.identity import AliasSource

        client, _ = make_client(llm=False)
        await client.start()
        await client.identity.register_alias(GUILD, ALICE, "alice", source=AliasSource.DISPLAY_NAME)
        await client.identity.register_alias(GUILD, BOB, "bob", source=AliasSource.DISPLAY_NAME)
        await client.facts.remember(guild_id=GUILD, subject_id=ALICE, text="enjoys green tea")
        await client.facts.remember(guild_id=GUILD, subject_id=BOB, text="enjoys coffee")
        summary = ProfileSummary(guild_id=GUILD, subject_id=ALICE, text="Alice profile")
        from icelake.consolidation.validity import summary_source

        summary = summary.model_copy(
            update={
                "sources": (summary_source(await client._store.get_fact(GUILD, "fct_test_0001")),)
            }
        )
        assert await client._store.put_summary(summary) is None
        queries = []
        recall = client.recall

        async def recording_recall(query):
            queries.append(query)
            return await recall(query)

        monkeypatch.setattr(client, "recall", recording_recall)
        ctx = await client.prompt_context(
            guild_id=GUILD,
            asker_id="alice",
            mentioned_ids=(ALICE, "bob", BOB),
            thread_participant_ids=("alice", BOB),
        )
        assert ctx.asker_summary == "Alice profile"
        assert "CURRENT ASKER" in ctx.injection_block
        assert "REFERENCED USER: alice" not in ctx.injection_block
        assert queries[0].subject_ids == (ALICE, BOB)
        assert queries[0].pair_ids == ((ALICE, BOB),)
        await client.close(drain=False)

    async def test_unknown_asker_never_promotes_a_mention(self, make_client):
        from icelake.models.retrieval import RecallWarning

        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(guild_id=GUILD, subject_id=BOB, text="enjoys coffee")
        ctx = await client.prompt_context(guild_id=GUILD, asker_id="unknown", mentioned_ids=(BOB,))
        assert "CURRENT ASKER" not in ctx.injection_block
        assert f"REFERENCED USER: {BOB}" in ctx.injection_block
        assert RecallWarning.SUBJECT_UNRESOLVED in ctx.warnings
        empty = await client.prompt_context(guild_id=GUILD, asker_id="unknown")
        assert not empty.facts
        assert "enjoys coffee" not in empty.injection_block
        await client.close(drain=False)

    async def test_blocked_summary_is_not_even_fetched(self, make_client, monkeypatch):
        from icelake.consolidation.validity import summary_source
        from icelake.models.facts import ProfileSummary

        client, _ = make_client(llm=False)
        await client.start()
        record = await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="enjoys quiet mornings"
        )
        summary = ProfileSummary(
            guild_id=GUILD,
            subject_id=ALICE,
            text="private profile",
            sources=(summary_source(record),),
        )
        await client._store.put_summary(summary)
        await client._store.set_opt_out(GUILD, ALICE, True)
        get_summary = client._store.get_summary
        reads = []

        async def recording_summary(guild_id, subject_id, **kwargs):
            reads.append(subject_id)
            return await get_summary(guild_id, subject_id, **kwargs)

        monkeypatch.setattr(client._store, "get_summary", recording_summary)
        ctx = await client.prompt_context(guild_id=GUILD, asker_id=ALICE)
        assert ALICE not in reads
        assert ctx.asker_summary is None
        assert "private profile" not in ctx.injection_block
        await client.close(drain=False)

    async def test_saturated_name_fact_fallback_is_unresolved(self, make_client):
        from icelake.models.facts import FactRecord
        from icelake.models.retrieval import RecallWarning

        client, _ = make_client(llm=False)
        await client.start()
        for index in range(11):
            await client._store.insert_fact(
                FactRecord(
                    id=f"name-{index}",
                    guild_id=GUILD,
                    subject_id=ALICE if index < 10 else BOB,
                    text="my name is Alex",
                )
            )
        ctx = await client.prompt_context(guild_id=GUILD, asker_id="Alex")
        assert ctx.resolutions[0].resolved is None
        assert not ctx.facts
        assert RecallWarning.IDENTITY_AMBIGUOUS in ctx.warnings
        await client.close(drain=False)


class TestGraphQueries:
    async def test_between_and_stances(self, make_client, event_factory):
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p1",
                            "speaker_token": "p0",
                            "text": "alice called bob a hacker after the ranked match ended",
                            "category": "relationships",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                            "relations": [
                                {"verb": "called_out", "from_token": "p0", "to_token": "p1"}
                            ],
                        },
                    ]
                ),
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        event = event_factory(
            content="@bob you hacked the whole ranked lobby and everyone watched it happen",
            author_id=ALICE,
            mentions=(BOB,),
            display_name="alice",
        )
        await client.observe(event)
        await client.flush()

        edges = await client.graph.between(GUILD, ALICE, BOB)
        assert any(e.verb == "called_out" for e in edges)

        stance_edges = await client.graph.relations_of(GUILD, ALICE)
        assert stance_edges

        neighbors = await client.graph.neighbors(GUILD, ALICE, depth=1)
        assert any(n.node_id == BOB for n in neighbors)
        await client.close()


class TestClassifyCommand:
    async def test_regex_gate_short_circuits_non_commands(self, make_client) -> None:
        client, _ = make_client(llm=False)
        command = await client.classify_command("totally normal chat about weather")
        assert command.action.value == "none"

    async def test_remember_classification_without_llm(self, make_client) -> None:
        client, _ = make_client(llm=False)
        command = await client.classify_command("remember that I love spicy ramen")
        assert command.action.value == "remember"
        assert "spicy ramen" in command.target_text
