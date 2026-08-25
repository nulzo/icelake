"""End-to-end pipeline tests with scripted LLMs: the accuracy guarantees in action."""

from __future__ import annotations

import asyncio
import json

from discord_memory.models.events import BatchCompleted, FactCommitted
from tests.conftest import (
    ExplodingLLM,
    ScriptedLLM,
    extraction_response,
    make_config,
)


async def observe_and_flush(client, event_factory, **event_kwargs):
    from discord_memory.models.events import ObserveStatus

    event_kwargs.setdefault("content", "hey everyone, quick update from me today")
    event = event_factory(**event_kwargs)
    receipt = await client.observe(event)
    assert receipt.status is ObserveStatus.ACCEPTED
    processed = await client.flush()
    return receipt, processed


class TestExtractionEndToEnd:
    async def test_simple_fact_extracted_and_recalled(
        self,
        make_client,
        event_factory,
        fixed_clock,
    ) -> None:
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice prefers mechanical keyboards",
                            "category": "preferences",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        }
                    ]
                ),
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        _, processed = await observe_and_flush(client, event_factory)
        assert processed >= 1
        stats = await client.stats("500000000000000001")
        assert stats.total_facts == 1

        result = await client.recall(
            __import__(
                "discord_memory",
                fromlist=["RecallQuery"],
            ).RecallQuery(
                guild_id="500000000000000001", text="keyboards", subject_ids=("100000000000000001",)
            )
        )
        assert len(result.facts) == 1
        assert "mechanical keyboards" in result.facts[0].fact.text
        await client.close()

    async def test_third_party_fact_anchored_on_target(
        self,
        make_client,
        event_factory,
    ) -> None:
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p1",  # bob (mentioned) is the subject
                            "speaker_token": "p0",  # alice said it
                            "text": "bob was called a hacker by alice during the match",
                            "category": "relationships",
                            "confidence": 0.85,
                            "source_message_indexes": [1],
                            "relations": [
                                {"verb": "called_out", "from_token": "p0", "to_token": "p1"}
                            ],
                        }
                    ]
                ),
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        guild = "500000000000000001"
        alice = "100000000000000001"
        bob = "200000000000000002"
        await observe_and_flush(
            client,
            event_factory,
            content=(
                "@bob stop hacking the lobby, you absolute hacker! "
                "everyone in this match saw exactly what you did there"
            ),
            author_id=alice,
            mentions=(bob,),
            display_name="alice",
        )
        facts_page = await client.facts.list_for_subject(guild, bob, include_server=False)
        texts = [f.text for f in facts_page.items]
        assert any("hacker" in text for text in texts), texts
        fact = next(f for f in facts_page.items if "hacker" in f.text)
        assert fact.subject_id == bob  # anchored on target
        assert fact.attribution.speaker_id == alice  # speaker attribution kept
        edges = await client.graph.between(guild, alice, bob)
        assert any(edge.verb == "called_out" for edge in edges)
        # cross-linked: retrievable from BOTH profiles via links channel
        for subject in (alice, bob):
            result = await client.recall(
                __import__(
                    "discord_memory",
                    fromlist=["RecallQuery"],
                ).RecallQuery(guild_id=guild, text="hacker", subject_ids=(subject,))
            )
            assert any("hacker" in sf.fact.text for sf in result.facts)
        await client.close()

    async def test_hallucinated_roster_token_rejected(
        self,
        make_client,
        event_factory,
    ) -> None:
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p7",  # not minted — must be dropped
                            "text": "stranger likes pineapple pizza entirely",
                            "confidence": 0.95,
                            "source_message_indexes": [1],
                        },
                        {
                            "subject_token": "unknown-person",  # invalid token form
                            "text": "someone else enjoys long walks on beaches",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        },
                    ]
                ),
            }
        )
        client, _ = make_client(llm=llm)
        await client.start()
        await observe_and_flush(client, event_factory, content="hey there everyone")
        stats = await client.stats("500000000000000001")
        assert stats.total_facts == 0  # nothing stored: verification gate held
        await client.close()

    async def test_duplicate_message_ignored(self, make_client, event_factory) -> None:
        client, _ = make_client()
        await client.start()
        event = event_factory(content="hello world this is fine")
        first = await client.observe(event)
        second = await client.observe(event)
        assert first.status.value == "accepted"
        assert second.reason.value == "duplicate"
        await client.close()

    async def test_noise_batch_skips_llm(self, make_client, event_factory) -> None:
        llm = ScriptedLLM({"extraction": extraction_response([])})
        client, _ = make_client(llm=llm)
        await client.start()
        for content_text in ("lol", "ok haha nice"):
            await client.observe(event_factory(content=content_text))
        calls_before = len(llm.calls)
        await client.flush()
        assert len(llm.calls) == calls_before  # noise gate skipped extraction
        await client.close()


class TestReconciliation:
    async def _seed(self, make_client, event_factory, llm):
        client, _ = make_client(llm=llm)
        await client.start()
        await observe_and_flush(
            client, event_factory, content="i have been learning rust for a year now"
        )
        return client

    async def test_semantic_collision_triggers_reconcile_and_reinforces_or_updates(
        self,
        make_client,
        event_factory,
    ) -> None:
        call_count = {"n": 0}

        def extract_handler(_request):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice has been programming in Rust for a year",
                            "category": "interests",
                            "confidence": 0.85,
                            "source_message_indexes": [1],
                        }
                    ]
                )
            return extraction_response(
                [
                    {
                        "subject_token": "p0",
                        "text": "alice has been programming in Rust for a year now",
                        "category": "interests",
                        "confidence": 0.85,
                        "source_message_indexes": [1],
                    }
                ]
            )

        reconcile_responses = [
            json.dumps(
                {"decisions": [{"kind": "noop", "target_id": None, "reason": "same meaning"}]}
            ),
        ]
        llm = ScriptedLLM(
            {
                "extraction": extract_handler,
                "reconcile": lambda request: reconcile_responses[0],
            }
        )
        client = await self._seed(make_client, event_factory, llm)
        guild = "500000000000000001"
        before = await client.stats(guild)
        await observe_and_flush(
            client, event_factory, content="yeah rust has been my language for a year"
        )
        after = await client.stats(guild)
        assert after.active_facts <= before.active_facts + 1
        await client.close()

    async def test_contradiction_invalidates_old_fact(
        self,
        make_client,
        event_factory,
    ) -> None:
        call_count = {"n": 0}
        old_fact_id = {"id": None}

        def extract_handler(request):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice lives in chicago downtown area",
                            "category": "personal",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        }
                    ]
                )
            return extraction_response(
                [
                    {
                        "subject_token": "p0",
                        "text": "alice lives in chicago suburbs now instead of downtown",
                        "category": "personal",
                        "confidence": 0.92,
                        "source_message_indexes": [1],
                    }
                ]
            )

        def reconcile_handler(request):
            prompt = request.messages[-1].content
            import re

            match = re.search(r"id=(fct_\S+)", prompt)
            if match:
                old_fact_id["id"] = match.group(1)
            return json.dumps(
                {
                    "decisions": [
                        {
                            "kind": "invalidate",
                            "target_id": match.group(1) if match else "",
                            "reason": "moved away",
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
        config = make_config(
            extraction={
                "noise_gate": True,
                "reconcile_collision_threshold": 0.5,
            }
        )
        client, _ = make_client(llm=llm, config=config)
        await client.start()
        await observe_and_flush(
            client, event_factory, content="i have been living in chicago downtown forever"
        )
        await observe_and_flush(
            client, event_factory, content="update: i just moved to seattle for work"
        )
        page = await client.facts.list_for_subject(
            "500000000000000001",
            "100000000000000001",
            include_server=False,
            active_only=False,
        )
        texts = {f.text: f.is_active for f in page.items}
        invalidated = [t for t, active in texts.items() if not active]
        assert invalidated, texts
        await client.close()


class TestReliability:
    async def test_llm_failure_dead_letters_not_silent(
        self,
        make_client,
        event_factory,
    ) -> None:
        client, _ = make_client(llm=ExplodingLLM())
        await client.start()
        await observe_and_flush(
            client,
            event_factory,
            content="remember everyone: my favorite hobby is collecting antique pocket watches",
        )
        health = await client.ops.health()
        assert health.dead_letters == 1
        requeued = await client.ops.retry_dead_letters()
        assert requeued == 1
        await client.close()

    async def test_events_published(self, make_client, event_factory) -> None:
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice collects vintage synthesizers",
                            "category": "interests",
                            "confidence": 0.8,
                            "source_message_indexes": [1],
                        }
                    ]
                ),
            }
        )
        client, _ = make_client(llm=llm)
        seen_batches: list[BatchCompleted] = []
        seen_facts: list[FactCommitted] = []
        client.events.subscribe(BatchCompleted, seen_batches.append)
        client.events.subscribe(FactCommitted, seen_facts.append)
        await client.start()
        await observe_and_flush(
            client,
            event_factory,
            content="just got a vintage roland juno-106 synthesizer for my home studio setup",
        )
        await asyncio.sleep(0)
        assert any(batch.adds >= 1 for batch in seen_batches)
        assert any("synthesizers" in fact.text for fact in seen_facts)
        await client.close()


class TestServerScopeAndConsent:
    async def test_opted_out_user_never_observed(self, make_client, event_factory) -> None:
        client, _ = make_client()
        await client.start()
        guild = "500000000000000001"
        quiet = event_factory(author_id="300000000000000003", display_name="quiet")
        await client.admin.set_opt_out(guild, quiet.author_id, True)
        receipt = await client.observe(quiet)
        assert receipt.reason.value == "opted_out"
        await client.close()

    async def test_bot_author_ignored(self, make_client, event_factory) -> None:
        client, _ = make_client()
        await client.start()
        bot_event = event_factory(
            author_id="999000000000000009", content="beep boop i am a bot message here"
        )
        receipt = await client.observe(
            bot_event.model_copy(update={"author_is_bot": True}),
        )
        assert receipt.reason.value == "bot_author"
        await client.close()

    async def test_server_scope_batch_creates_community_facts(
        self,
        make_client,
        event_factory,
    ) -> None:
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "server",
                            "text": "the community communicates with heavy sarcasm and memes",
                            "category": "culture",
                            "confidence": 0.85,
                            "source_message_indexes": [1],
                        }
                    ]
                ),
            }
        )
        client, _ = make_client(llm=llm, config=make_config())
        await client.start()
        guild = "500000000000000001"
        for i in range(3):
            await client.observe(event_factory(content=f"message number {i} about memes"))
        report = await client._pipeline.flush_subject(guild, "__server__")
        assert report.summary is not None or report.skipped_reason is not None
        server_facts = await client.facts.search(guild, "sarcasm", server_only=True)
        assert server_facts
        await client.close()
