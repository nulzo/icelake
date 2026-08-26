"""Coverage for this review round: JSON repair, maintenance, alias mining,
mention links, summary refresh, pair/entity-hint recall, cite instructions."""

from __future__ import annotations

import pytest

from discord_memory._json import parse_json_object
from discord_memory.identity.aliases import (
    extract_self_name_aliases,
    is_third_party_name_reference,
)
from discord_memory.lifecycle.maintenance import MaintenanceService
from tests.conftest import ScriptedLLM, extraction_response

GUILD = "500000000000000001"
ALICE = "100000000000000001"


class TestJsonRepairLadder:
    def test_clean_json(self) -> None:
        assert parse_json_object('{"a": 1}') == {"a": 1}

    def test_fenced_json(self) -> None:
        assert parse_json_object('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}

    def test_prose_surrounding(self) -> None:
        assert parse_json_object('Sure! Here you go: {"ok": true} — done.') == {
            "ok": True,
        }

    def test_nested_braces_in_strings(self) -> None:
        payload = '{"text": "curly } brace { inside", "n": 1}'
        assert parse_json_object(payload)["n"] == 1

    def test_truncated_string_repaired(self) -> None:
        truncated = '{"operations": [{"text": "cut off mid str'
        parsed = parse_json_object(truncated)
        assert "operations" in parsed

    def test_truncated_brackets_repaired(self) -> None:
        truncated = '{"operations": [1, 2'
        parsed = parse_json_object(truncated)
        assert isinstance(parsed.get("operations"), list)

    def test_unrecoverable_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_json_object("no json here at all")

    def test_multiple_objects_takes_first_balanced(self) -> None:
        text = 'noise {"first": true} middle {"second": false} tail'
        # naive first{...last} slice fails; balanced scan recovers the first
        parsed = parse_json_object(text)
        assert parsed == {"first": True}


class TestAliasMiningAndGuards:
    def test_name_patterns(self) -> None:
        found = extract_self_name_aliases("My name is Klim and they call me Krill")
        names = [surface.lower() for surface, _weight in found]
        assert "klim" in names and "krill" in names

    def test_no_false_positives_on_plain_text(self) -> None:
        assert extract_self_name_aliases("likes watching movies a lot") == []

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("alice's brother Ivan came over", "Ivan"),
            ("someone named Ivan joined yesterday", "Ivan"),
            ("her friend Ivan is nice", "Ivan"),
            ("ivan likes tea", None),
        ],
    )
    def test_third_party_guard(self, text: str, expected: str | None) -> None:
        assert is_third_party_name_reference(text, expected or "Ivan") is (expected is not None)


class TestMaintenanceService:
    async def test_sweep_forget_prune_flow(self, fixed_clock) -> None:
        """Extracted-style facts (non-manual) get swept by the maintenance job."""
        from datetime import timedelta

        from discord_memory.adapters.in_memory.store import InMemoryStore

        store = InMemoryStore()
        config = __import__("discord_memory", fromlist=["MemoryConfig"]).MemoryConfig()
        service = MaintenanceService(store=store, config=config, clock=fixed_clock)
        now = fixed_clock.now()

        async def seed(fact_id: str, *, expires_in: timedelta | None) -> None:
            record = __import__(
                "discord_memory.models.facts",
                fromlist=["FactRecord"],
            ).FactRecord(
                id=fact_id,
                guild_id=GUILD,
                subject_id=ALICE,
                text=f"extracted observation {fact_id} for maintenance testing",
                category=__import__(
                    "discord_memory.models.facts",
                    fromlist=["FactCategory"],
                ).FactCategory.INTERESTS,
                strength=1.0,
                last_reinforced_at=now,
                created_at=now,
                updated_at=now,
                valid_from=now,
                expires_at=(now + expires_in) if expires_in else None,
            )
            await store.insert_fact(record)

        await seed("fct_exp", expires_in=timedelta(days=-1))  # already expired
        await seed("fct_keep", expires_in=timedelta(days=400))
        report = await service.run_guild(GUILD, force=True)
        assert report.expired == 1
        swept = await store.get_fact(GUILD, "fct_exp")
        assert swept is not None and not swept.is_active
        kept = await store.get_fact(GUILD, "fct_keep")
        assert kept is not None and kept.is_active

    async def test_throttling(self, fixed_clock) -> None:
        from discord_memory.adapters.in_memory.store import InMemoryStore
        from discord_memory.config import MemoryConfig

        service = MaintenanceService(
            store=InMemoryStore(),
            config=MemoryConfig(),
            clock=fixed_clock,
        )
        first = await service.run_guild("g")
        second = await service.run_guild("g")
        assert first.skipped_reason is None
        assert second.skipped_reason == "throttled"
        forced = await service.run_guild("g", force=True)
        assert forced.skipped_reason is None


class TestMentionLinksAndSummaryRefresh:
    async def test_mentioned_user_gets_link_row(self, make_client, event_factory):
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice was talking about carol during lunch break today",
                            "category": "relationships",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        },
                    ]
                )
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        CAROL = "300000000000000003"
        event = event_factory(
            content=("carol told me the funniest story at lunch about her camping trip"),
            author_id=ALICE,
            mentions=(CAROL,),
            display_name="alice",
        )
        await client.observe(event)
        await client.flush()
        from discord_memory.models.graph import NodeType

        linked = await client._store.links_for_node(GUILD, NodeType.USER, CAROL)
        assert any(row.kind.value in {"mentioned_with", "about_user"} for row, _record in linked)
        # third-party guard: carol's name must NOT become alice's alias
        aliases = await client.identity.aliases_of(GUILD, ALICE)
        assert all(record.alias_norm != "carol" for record in aliases)
        await client.close()

    async def test_summary_auto_refresh_after_enough_adds(
        self,
        make_client,
        event_factory,
    ):
        summary_text = "the asker enjoys chess and long walks"
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": f"fact batch {i}: likes chess walks",
                            "category": "interests",
                            "confidence": 0.85,
                            "source_message_indexes": [1],
                        }
                        for i in range(3)
                    ]
                ),
                "summarize": summary_text,
            }
        )
        from tests.conftest import make_config

        config = make_config(
            extraction={
                "auto_consolidate_after_adds": 2,
                "summary_sanity_threshold": 0.2,
            }
        )
        client, _ = make_client(llm=llm, config=config)
        await client.start()
        for i in range(3):
            await client.observe(
                event_factory(content=f"message number {i} mentioning chess and long walks")
            )
        await client.flush()
        doc = await client._store.get_summary(GUILD, ALICE)
        assert doc is not None and "chess" in doc.text
        await client.close()


class TestPairAndEntityHintRecall:
    async def test_pair_ids_intersect_recall(self, make_client, event_factory):
        BOB = "200000000000000002"
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p1",
                            "speaker_token": "p0",
                            "text": "bob got called out by alice after the ranked match ended",
                            "category": "relationships",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                            "relations": [
                                {"verb": "called_out", "from_token": "p0", "to_token": "p1"}
                            ],
                        },
                    ]
                )
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        event = event_factory(
            content=("@bob that was blatant hacking in the ranked match and everyone saw"),
            author_id=ALICE,
            mentions=(BOB,),
            display_name="alice",
        )
        await client.observe(event)
        await client.flush()

        result = await client.recall(
            __import__(
                "discord_memory.models.retrieval",
                fromlist=["RecallQuery"],
            ).RecallQuery(guild_id=GUILD, pair_ids=(ALICE, BOB))
        )
        texts = [sf.fact.text for sf in result.facts]
        assert any("called out" in t or "hacking" in t for t in texts), texts
        await client.close()

    async def test_entity_hint_seeds_aggregation(self, make_client, event_factory):
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice plays chess every single weekend",
                            "category": "interests",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                            "entities": [{"name": "Chess", "kind": "concept"}],
                            "relations": [
                                {"verb": "plays", "from_token": "p0", "to_entity": "Chess"}
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
                content="chess club meets on saturdays and i never miss it",
                author_id=ALICE,
                display_name="alice",
            )
        )
        await client.flush()
        result = await client.recall(
            __import__(
                "discord_memory.models.retrieval",
                fromlist=["RecallQuery"],
            ).RecallQuery(guild_id=GUILD, entity_hint="chess")
        )
        assert any("chess" in sf.fact.text.lower() for sf in result.facts)
        await client.close()


class TestCiteInstructions:
    def test_instruction_appended_when_citations_exist(self) -> None:
        from datetime import UTC, datetime

        from discord_memory.models.facts import FactRecord, SourceRef, SourceRole
        from discord_memory.models.retrieval import ScoredFact
        from discord_memory.retrieval.injection import (
            CITATION_INSTRUCTION,
            InjectionBuilder,
        )

        record = FactRecord(
            id="fct_ci",
            guild_id=GUILD,
            subject_id=ALICE,
            text="likes chess",
            created_at=datetime.now(UTC),
            citations=(
                SourceRef(
                    message_id="m9",
                    channel_id="c9",
                    guild_id=GUILD,
                    author_id=ALICE,
                    role=SourceRole.PRIMARY,
                ),
            ),
        )
        builder = InjectionBuilder()
        block, citations, _trimmed = builder.build(
            asker_id=ALICE,
            facts_by_section={"asker": (ScoredFact(fact=record, score=0.9),)},
            summaries={},
            token_budget=5_000,
            guild_id=GUILD,
        )
        assert CITATION_INSTRUCTION.split("\n")[0][:20] in block
        assert len(citations) == 1


class TestServerFactDedupAndCitations:
    """Regression: cross-batch server duplicates + provenance snapshots."""

    async def test_server_fact_deduped_across_batches(
        self,
        make_client,
        event_factory,
    ) -> None:
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice plays chess every weekend",
                            "category": "interests",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        },
                        {
                            "subject_token": "server",
                            "text": "the community bonds over late night gaming sessions",
                            "category": "culture",
                            "confidence": 0.85,
                            "source_message_indexes": [1],
                        },
                    ]
                )
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        await client.observe(
            event_factory(content="chess night went long but the games were great", author_id=ALICE)
        )
        await client.flush()
        BOB = "200000000000000002"
        event = event_factory(
            content="more chess talk happened later on", author_id=BOB, display_name="bob"
        )
        await client.observe(event)
        await client.flush()
        server = await client.facts.search(GUILD, "bonds", server_only=True)
        assert len(server) == 1, [(r.text,) for r, _ in server]
        await client.close()

    async def test_extracted_fact_carries_citation_snapshot(
        self,
        make_client,
        event_factory,
    ) -> None:
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice mains support in every ranked game she plays",
                            "category": "interests",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        },
                    ]
                )
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        event = event_factory(
            content="i always end up playing support no matter what team needs",
            author_id=ALICE,
            display_name="alice",
        )
        await client.observe(event)
        await client.flush()
        page = await client.facts.list_for_subject(GUILD, ALICE, include_server=False)
        fact = next(f for f in page.items if "support" in f.text)
        assert fact.citations, "extracted facts must carry provenance"
        primary = fact.citations[0]
        assert primary.message_id == event.message_id
        assert primary.guild_id == GUILD
        assert primary.role.value in {"primary", "supporting"}
        ctx = await client.prompt_context(guild_id=GUILD, asker_id=ALICE, text="ranked games")
        linkified = ctx.apply_citations("she mains support [mem:1]")
        assert "discord.com/channels/" in linkified or "[mem:1]" in linkified
        await client.close()
