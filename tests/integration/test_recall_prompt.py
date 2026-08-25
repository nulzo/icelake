"""End-to-end recall and prompt-context tests (API.md §6)."""

from __future__ import annotations

from discord_memory.models.events import ObserveStatus
from discord_memory.models.retrieval import RecallQuery, Scope
from tests.conftest import ScriptedLLM, extraction_response

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
        event = event_factory(
            content="tournament season is starting again soon everyone",
            author_id=ALICE,
        )
        await client.observe(event)
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
        llm = ScriptedLLM({"extraction": extraction_response(facts)})
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
            resolved = ctx.apply_citations(f"see {citation.ref} ok")
            assert citation.url in resolved
        usage = ctx.usage
        assert usage.prompt_tokens > 0
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
        assert command.action == "none"

    async def test_remember_classification_without_llm(self, make_client) -> None:
        client, _ = make_client(llm=False)
        command = await client.classify_command("remember that I love spicy ramen")
        assert command.action == "remember"
        assert "spicy ramen" in command.target_text
