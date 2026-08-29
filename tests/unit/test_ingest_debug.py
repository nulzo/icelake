"""Debug logs and skip reasons for observe drops and empty extractions."""

from __future__ import annotations

import asyncio
import logging

from icelake.models.events import BatchCompleted, IgnoreReason, ObserveStatus
from tests.conftest import ScriptedLLM, extraction_response

SUBSTANTIVE = "this is a reasonably long message about nothing in particular today"


class TestObserveDropLogs:
    async def test_bot_author_logs_reason(self, make_client, event_factory, caplog) -> None:
        caplog.set_level(logging.DEBUG, logger="icelake.api.client")
        client, _ = make_client()
        await client.start()
        event = event_factory(content=SUBSTANTIVE).model_copy(update={"author_is_bot": True})
        receipt = await client.observe(event)
        assert receipt.status is ObserveStatus.IGNORED
        assert receipt.reason is IgnoreReason.BOT_AUTHOR
        assert "reason=bot_author" in caplog.text
        await client.close()

    async def test_opt_out_logs_reason(self, make_client, event_factory, caplog) -> None:
        caplog.set_level(logging.DEBUG, logger="icelake.api.client")
        client, _ = make_client()
        await client.start()
        event = event_factory(content=SUBSTANTIVE)
        await client.admin.set_opt_out(event.guild_id, event.author_id, True)
        receipt = await client.observe(event)
        assert receipt.reason is IgnoreReason.OPTED_OUT
        assert "reason=opted_out" in caplog.text
        await client.close()


class TestExtractionSkipLogs:
    async def test_noise_skip_logs_and_emits(self, make_client, event_factory, caplog) -> None:
        caplog.set_level(logging.DEBUG, logger="icelake.ingest.pipeline")
        client, _ = make_client()
        seen: list[BatchCompleted] = []
        client.events.subscribe(BatchCompleted, seen.append)
        await client.start()
        await client.observe(event_factory(content="lol"))
        await client.flush()
        await asyncio.sleep(0)
        assert "reason=noise" in caplog.text
        assert any(event.skipped_reason == "noise" for event in seen)
        await client.close()

    async def test_empty_ops_logs_and_emits(self, make_client, event_factory, caplog) -> None:
        caplog.set_level(logging.DEBUG, logger="icelake.ingest.pipeline")
        llm = ScriptedLLM({"extraction": extraction_response([])})
        client, _ = make_client(llm=llm)
        seen: list[BatchCompleted] = []
        client.events.subscribe(BatchCompleted, seen.append)
        await client.start()
        await client.observe(event_factory(content=SUBSTANTIVE))
        await client.flush()
        await asyncio.sleep(0)
        assert "reason=empty_ops" in caplog.text
        assert any(event.skipped_reason == "empty_ops" for event in seen)
        await client.close()

    async def test_gated_candidate_logs_reason(self, make_client, event_factory, caplog) -> None:
        caplog.set_level(logging.DEBUG, logger="icelake.ingest.pipeline")
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "what does alice like to eat for dinner tonight",
                            "category": "general",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        }
                    ]
                ),
            }
        )
        client, _ = make_client(llm=llm)
        seen: list[BatchCompleted] = []
        client.events.subscribe(BatchCompleted, seen.append)
        await client.start()
        await client.observe(event_factory(content=SUBSTANTIVE))
        await client.flush()
        await asyncio.sleep(0)
        assert "reason=question" in caplog.text
        assert any(event.skipped_reason == "gated" for event in seen)
        await client.close()
