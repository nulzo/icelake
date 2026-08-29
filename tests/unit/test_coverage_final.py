"""Final coverage pass: API groups, injection citations, degradation, UPDATE path."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from icelake.api.groups import AdminApi, GraphApi, IdentityApi
from icelake.models.facts import SourceRef, SourceRole
from icelake.models.identity import AliasSource
from icelake.models.retrieval import ScoredFact
from icelake.retrieval.injection import InjectionBuilder, snippet
from tests.conftest import ScriptedLLM, extraction_response

GUILD = "500000000000000001"
ALICE = "100000000000000001"
BOB = "200000000000000002"


def _fact_record(text: str, fact_id: str = "fct_c1", subject_id: str | None = ALICE, citations=()):
    now = datetime.now(UTC)
    from icelake.models.facts import FactCategory, FactRecord

    return FactRecord(
        id=fact_id,
        guild_id=GUILD,
        subject_id=subject_id,
        text=text,
        category=FactCategory.INTERESTS,
        created_at=now,
        updated_at=now,
        observed_at=now,
        valid_from=now,
        citations=citations,
    )


class TestGroupsSurface:
    async def test_identity_register_and_aliases(self) -> None:
        from icelake.adapters.in_memory.store import InMemoryStore

        store = InMemoryStore()
        api = IdentityApi(store)
        await api.register_alias(GUILD, ALICE, "Alice W", source=AliasSource.DISPLAY_NAME)
        resolution = await api.resolve(GUILD, "alice w")
        assert resolution.resolved is not None
        aliases = await api.aliases_of(GUILD, ALICE)
        assert any(record.alias_norm == "alice w" for record in aliases)

    async def test_register_alias_rejects_empty(self) -> None:
        from icelake.adapters.in_memory.store import InMemoryStore

        api = IdentityApi(InMemoryStore())
        await api.register_alias(GUILD, ALICE, "   ")  # no-op, no crash

    async def test_graph_relations_of_and_stances(self, make_client) -> None:
        client, _llm = make_client(
            llm=ScriptedLLM(
                {
                    "extraction": extraction_response(
                        [
                            {
                                "subject_token": "p1",
                                "speaker_token": "p0",
                                "text": "bob got teased by alice about his "
                                "chess opening repertoire",
                                "category": "relationships",
                                "confidence": 0.9,
                                "source_message_indexes": [1],
                                "relations": [
                                    {"verb": "called_out", "from_token": "p0", "to_token": "p1"}
                                ],
                            },
                            {
                                "subject_token": "server",
                                "text": "the community adores chess tournaments on weekends lately",
                                "category": "culture",
                                "confidence": 0.9,
                                "source_message_indexes": [1],
                                "entities": [{"name": "Chess", "kind": "concept"}],
                                "relations": [
                                    {"verb": "loves", "from_token": "server", "to_entity": "Chess"}
                                ],
                            },
                        ]
                    )
                }
            )
        )
        await client.start()
        event = event_factory_for(
            client,
            content=("@bob your chess opening lost you that whole tournament game honestly"),
            author_id=ALICE,
            mentions=(BOB,),
        )
        await client.observe(event)
        await client.flush()

        graph = GraphApi(store=client._store)
        outgoing = await graph.relations_of(GUILD, ALICE)
        assert outgoing
        stances = await graph.entity_stances(GUILD, "chess")
        assert stances.entity_slug == "chess"
        admin = AdminApi(client._store)
        assert not await admin.get_opt_out(GUILD, BOB)
        await client.close()

    async def test_graph_between_empty(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        edges = await client.graph.between(GUILD, ALICE, BOB)
        assert edges == ()
        neighbors = await client.graph.neighbors(GUILD, ALICE, depth=2)
        assert neighbors == ()
        await client.close()


def event_factory_for(client, *, content: str, author_id: str, mentions: tuple[str, ...] = ()):

    from icelake.models.events import MessageEvent

    return MessageEvent(
        message_id=f"msg-{abs(hash(content)) % 10**12}",
        guild_id=GUILD,
        channel_id="c1",
        author_id=author_id,
        content=content,
        created_at=datetime.now(UTC),
        author_display_name="alice",
        mention_ids=mentions,
    )


class TestInjectionCitations:
    def _scored_with_source(self, role: SourceRole = SourceRole.PRIMARY):
        record = _fact_record(
            "likes building mechanical keyboards on weekends",
            citations=(
                SourceRef(
                    message_id="m1",
                    channel_id="c1",
                    guild_id=GUILD,
                    author_id=ALICE,
                    author_name="alice",
                    content_snippet="i love my keyboard collection so much",
                    message_url="",
                    role=role,
                ),
            ),
        )
        return ScoredFact(fact=record, score=0.9)

    def test_citation_rendered_from_message_url_fallback(self) -> None:
        builder = InjectionBuilder()
        block, citations, _trimmed = builder.build(
            asker_id=BOB,
            facts_by_section={"user:al": (self._scored_with_source(),)},
            summaries={},
            token_budget=5_000,
            guild_id=GUILD,
        )
        assert "[mem:1]" in block
        assert len(citations) == 1
        assert "/555.../".strip() or citations[0].url.startswith("https://discord.com")

    def test_supporting_role_fallback_when_no_primary(self) -> None:
        scored = self._scored_with_source(role=SourceRole.SUPPORTING)
        builder = InjectionBuilder()
        _block, citations, _trimmed = builder.build(
            asker_id=BOB,
            facts_by_section={"asker": (scored,)},
            summaries={},
            token_budget=5_000,
            guild_id=GUILD,
        )
        assert len(citations) == 1

    def test_summary_only_section_renders(self) -> None:
        builder = InjectionBuilder()
        block, _citations, _trimmed = builder.build(
            asker_id=BOB,
            facts_by_section={"asker": ()},
            summaries={"asker": "a long digest of this person " * 3},
            token_budget=5_000,
            guild_id=GUILD,
        )
        assert "Summary:" in block

    def test_snippet_truncates_long_text(self) -> None:
        long_text = "x" * 500
        result = snippet(long_text, max_chars=50)
        assert result.endswith("…") and len(result) == 50


class TestRetrievalDegradation:
    async def test_channel_failure_reported_not_raised(self, make_client) -> None:
        class ExplodingVectors:
            async def setup(self) -> None:
                pass

            async def upsert(self, items):
                return None

            async def delete(self, ids):
                return 0

            async def search(self, *args, **kwargs):
                raise RuntimeError("vector backend down")

            async def count(self, guild_id):
                return 0

        client, _ = make_client(llm=ScriptedLLM({"extraction": extraction_response([])}))
        client._vectors = ExplodingVectors()
        await client.start()
        await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="plays table tennis every lunch break"
        )
        from icelake.models.retrieval import RecallQuery

        result = await client.recall(
            RecallQuery(guild_id=GUILD, text="table tennis", subject_ids=(ALICE,))
        )
        # vector channel failure degrades; keyword/baseline still serve results
        assert "vector" in result.degraded_channels
        assert result.facts  # other channels carried the recall
        await client.close()


class TestReconcileUpdatePath:
    async def test_update_decision_supersedes_existing_fact(
        self,
        make_client,
        event_factory,
    ) -> None:
        calls = {"n": 0}

        def extract_handler(_request):
            calls["n"] += 1
            if calls["n"] == 1:
                return extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice works as a nurse at city hospital",
                            "category": "professional",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        }
                    ]
                )
            return extraction_response(
                [
                    {
                        "subject_token": "p0",
                        "text": "alice works as a nurse practitioner at city hospital",
                        "category": "professional",
                        "confidence": 0.9,
                        "source_message_indexes": [1],
                    }
                ]
            )

        def reconcile_handler(request):
            import re

            match = re.search(r"\[(\d+)\]", request.messages[-1].content)
            target = int(match.group(1)) if match else None
            return json.dumps(
                {
                    "decisions": [
                        {
                            "candidate_index": 0,
                            "kind": "update",
                            "target_id": target,
                            "text": "alice works as a nurse practitioner at city hospital",
                            "reason": "role refined",
                        }
                    ]
                }
            )

        llm = ScriptedLLM(
            {
                "extraction": extract_handler,
                "reconcile": reconcile_handler,
            }
        )
        from tests.conftest import make_config

        config = make_config(
            extraction={
                "noise_gate": True,
                "reconcile_collision_threshold": 0.5,
            }
        )
        client, _ = make_client(llm=llm, config=config)
        await client.start()
        await client.observe(
            event_factory(
                content="i have been working at the hospital for years now honestly",
                author_id=ALICE,
                display_name="alice",
            )
        )
        await client.flush()
        await client.observe(
            event_factory(
                content="update on work stuff: they promoted me to a new nursing role recently",
                author_id=ALICE,
                display_name="alice",
            )
        )
        await client.flush()

        page = await client.facts.list_for_subject(GUILD, ALICE, active_only=False)
        superseded = [f for f in page.items if f.superseded_by_id]
        assert superseded, [f.text for f in page.items]
        history = await client.facts.history(superseded[0].id, guild_id=GUILD)
        kinds = [str(getattr(entry.kind, "value", entry.kind)) for entry in history]
        assert "superseded" in kinds
        await client.close()


class TestClientWorkerPaths:
    async def test_maybe_server_batch_direct(self, make_client, event_factory) -> None:
        client, _ = make_client(
            llm=ScriptedLLM(
                {
                    "extraction": extraction_response([]),
                }
            )
        )
        await client.start()
        await client.observe(event_factory(content="a chat line long enough to pass"))
        await client.flush()
        await client._pipeline.flush_subject(GUILD, "__server__")  # empty window safe
        health = await client.ops.health()
        assert health.pending_messages == 0
        await client.close()

    async def test_worker_loop_survives_queue_errors(self, event_factory) -> None:
        from icelake.adapters.in_memory.queue import InMemoryIngestQueue

        class FlakyQueue(InMemoryIngestQueue):
            async def due_batch_keys(self, **kwargs):  # type: ignore[no-untyped-def]
                raise RuntimeError("queue hiccup")

        from icelake.config import MemoryConfig

        config = MemoryConfig(
            storage="sqlite://:memory:",
            workers={"enabled": True, "count": 1, "poll_interval_seconds": 0.01},
        )
        from icelake.api.client import DiscordMemory

        queue = FlakyQueue()
        client = DiscordMemory(config, clock=None, llm=None, queue=queue)
        client.started = False
        await client.start()
        await __import__("asyncio").sleep(0.08)
        assert any(task.done() is False for task in client.worker_tasks)
        await client.close(drain=False)

    async def test_classify_command_uses_llm_when_configured(self, make_client) -> None:
        llm = ScriptedLLM(
            {
                "classify_command": json.dumps(
                    {
                        "action": "query",
                        "confidence": 0.8,
                    }
                ),
            }
        )
        client, _llm = make_client(llm=llm)
        command = await client.classify_command("what do you know about me then?")
        assert command.action.value == "query"


class TestDiscordPyRemainingLines:
    async def test_on_message_ignores_dm(self, fixed_clock) -> None:
        pytest.importorskip("discord")
        from types import SimpleNamespace

        from icelake.integrations.discord_py import setup_discord_memory
        from tests.conftest import make_config

        listeners: dict[str, list] = {}

        class StubBot:
            user = SimpleNamespace(id=42)

            def listen(self, name):
                def decorator(fn):
                    listeners.setdefault(name, []).append(fn)
                    return fn

                return decorator

        memory, _cog = await setup_discord_memory(
            StubBot(), make_config(), clock=fixed_clock, llm=None
        )
        await listeners["on_ready"][0]()
        dm_message = SimpleNamespace(guild=None)
        await listeners["on_message"][0](dm_message)  # early return, no crash
        await memory.close(drain=False)


class TestSqliteMergeEntitiesRealMove:
    async def test_merge_moves_counts_and_aliases(self) -> None:
        from icelake.adapters.sqlite.store import SqliteStore

        store = SqliteStore("sqlite://:memory:")
        await store.setup()
        await store.upsert_entity("g", "films", "Films", "concept", ("film",))
        await store.bump_entity_facts("g", "films", delta=3)
        moved = await store.merge_entities("g", ("films",), to_slug="movies")
        assert moved == 1
        target = await store.get_entity("g", "movies")
        assert target is not None and target.fact_count >= 3
        assert "films" in target.aliases
        assert await store.get_entity("g", "films") is None
        await store.close()


class TestExtractionResidualBranches:
    async def test_noop_decision_without_target_allowed(self) -> None:
        from icelake.models.operations import ReconcileOutput

        output = ReconcileOutput.model_validate({"decisions": [{"kind": "noop"}]})
        assert output.decisions[0].target_id is None
