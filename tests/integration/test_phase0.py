"""Phase 0 regression: observe gates, queue capacity, attribution types."""

from __future__ import annotations

from datetime import UTC, datetime

from discord_memory import DiscordMemory, MessageEvent
from discord_memory.models.events import IgnoreReason
from discord_memory.models.facts import AttributionType
from tests.conftest import ScriptedLLM, extraction_response, make_config

GUILD = "500000000000000001"


class TestObserveGates:
    async def test_min_message_chars_gate(self) -> None:
        memory = DiscordMemory(
            make_config(observe={"min_message_chars": 10}),
            llm=None,
        )
        await memory.start()
        receipt = await memory.observe(
            MessageEvent(
                message_id="1",
                guild_id=GUILD,
                channel_id="c",
                author_id="u1",
                content="too short",
                created_at=datetime.now(UTC),
            )
        )
        assert receipt.reason is IgnoreReason.EMPTY_CONTENT or receipt.reason.value in (
            "empty_content",
            "ignored_pattern",
        )
        await memory.close()

    async def test_ignore_patterns_gate(self) -> None:
        memory = DiscordMemory(
            make_config(observe={"ignore_patterns": (r"\blol\b", r"\bok\b")}),
            llm=None,
        )
        await memory.start()
        receipt = await memory.observe(
            MessageEvent(
                message_id="1",
                guild_id=GUILD,
                channel_id="c",
                author_id="u1",
                content="lol that is genuinely hilarious my friend wow so amazing",
                created_at=datetime.now(UTC),
            )
        )
        # "lol" matches the pattern at start → IGNORED_PATTERN
        assert receipt.reason is IgnoreReason.IGNORED_PATTERN
        await memory.close()


class TestAttributionTypes:
    def test_inferred_and_agent_exist(self) -> None:

        assert AttributionType.INFERRED.value == "inferred"
        assert AttributionType.AGENT.value == "agent"

    def test_inferred_and_agent_values(self) -> None:
        assert AttributionType.INFERRED.value == "inferred"
        assert AttributionType.AGENT.value == "agent"

    async def test_queue_capacity_enforcement(self) -> None:
        from datetime import datetime as dt

        from discord_memory.adapters.in_memory.queue import InMemoryIngestQueue
        from discord_memory.ports.queue import StoredMessage

        queue = InMemoryIngestQueue()
        now = dt.now(UTC)
        for i in range(3):
            await queue.put_message(
                StoredMessage(
                    message_id=f"cap{i}",
                    guild_id="g1",
                    author_id="u1",
                    subject_key="u1",
                    content=f"msg {i}",
                    created_at=now,
                )
            )
        # capacity = 3 already pending; new one should be rejected with False
        result = await queue.put_message(
            StoredMessage(
                message_id="cap3",
                guild_id="g1",
                author_id="u1",
                subject_key="u1",
                content="over capacity",
                created_at=now,
            ),
            max_depth=3,
        )
        assert result is False


class TestGetAllParity:
    async def test_get_all(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="fact alpha for testing purposes", actor_id="x"
        )
        items = await client.facts.get_all(GUILD, ALICE)
        assert len(items) == 1
        await client.close()


class TestExtractNow:
    async def test_extract_now_commits_synchronously(self) -> None:
        llm_obj = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "alice plays chess on tuesdays",
                            "category": "interests",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        },
                    ]
                )
            }
        )
        from tests.conftest import make_config as _mc

        memory_obj = DiscordMemory(_mc(), llm=llm_obj)
        await memory_obj.start()
        receipt = await memory_obj.extract_now(
            MessageEvent(
                message_id="en1",
                guild_id=GUILD,
                channel_id="c",
                author_id=ALICE,
                content="chess night every tuesday at the cafe",
                created_at=datetime.now(UTC),
                author_display_name="alice",
            )
        )
        assert receipt.status.value == "accepted"
        page = await memory_obj.facts.list_for_subject(GUILD, ALICE)
        assert any("chess" in f.text for f in page.items)
        await memory_obj.close()


ALICE = "100000000000000001"
