"""N-process governance race: purge lands while an extraction LLM call is paused.

Two independent DiscordMemory instances share one disposable SQLite file, so
this exercises cross-process fencing without live services: the worker claims
a batch and blocks inside the LLM, the admin process opts out and purges, then
the LLM is released. The paused output must never become visible, and a later
opt-in must not revive it.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from icelake.api.client import DiscordMemory
from icelake.config import MemoryConfig
from icelake.models.events import MessageEvent
from icelake.ports.clock import SystemClock, UlidIdGen
from icelake.ports.llm import ChatRequest, ChatResponse
from icelake.ports.queue import BatchKey


class GatedLLM:
    """Extraction calls block until released; responses scripted per call."""

    model_name = "gated-test-model"

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.extraction_calls = 0

    async def complete(self, request: ChatRequest) -> ChatResponse:
        if request.purpose != "extraction":
            return ChatResponse(
                text="{}", prompt_tokens=1, completion_tokens=1, model=self.model_name
            )
        self.extraction_calls += 1
        self.entered.set()
        await self.release.wait()
        operation = (
            {
                "subject_token": "p0",
                "text": "secret fact from paused llm",
                "confidence": 0.9,
            }
            if self.extraction_calls == 1
            else {
                "subject_token": "p0",
                "text": "post optin fact",
                "confidence": 0.9,
            }
        )
        return ChatResponse(
            text=json.dumps({"operations": [operation]}),
            prompt_tokens=10,
            completion_tokens=5,
            model=self.model_name,
        )


def _event(message_id: str, author_id: str) -> MessageEvent:
    return MessageEvent(
        message_id=message_id,
        guild_id="g1",
        channel_id="c",
        author_id=author_id,
        content="i enjoy trivia nights",
        created_at=datetime.now(UTC),
    )


async def test_purge_during_paused_llm_never_replays_after_optin(tmp_path) -> None:
    db = f"sqlite:///{tmp_path / 'race.db'}"
    config = MemoryConfig(
        storage=db,
        batching={
            "batch_size_messages": 1,
            "max_age_seconds": 10,
            "lease_seconds": 120,
        },
        workers={"enabled": False},
        embeddings="hashing",
        extraction={"noise_gate": False},
    )
    llm = GatedLLM()
    admin = DiscordMemory(config, clock=SystemClock(), id_gen=UlidIdGen(), llm=None)
    worker = DiscordMemory(config, clock=SystemClock(), id_gen=UlidIdGen(), llm=llm)
    await admin.start()
    await worker.start()
    try:
        receipt = await admin.observe(_event("m1", "alice_id"))
        from icelake.models.events import ObserveStatus

        assert receipt.status is ObserveStatus.ACCEPTED

        batch_task = asyncio.create_task(
            worker._pipeline.process_key(BatchKey(guild_id="g1", subject_key="alice_id"))
        )
        await llm.entered.wait()
        await admin.admin.set_opt_out("g1", "alice_id", True)
        report = await admin.admin.purge_user("g1", "alice_id", dry_run=False)
        assert report.facts_removed == 0
        assert report.messages_removed == 1
        assert await worker._queue.pending_count("g1") == 0

        await admin.admin.set_opt_out("g1", "alice_id", False)
        llm.release.set()
        batch_report = await batch_task
        assert batch_report.skipped_reason == "consent_revoked"
        assert batch_report.summary is None
        assert (await admin._store.consent_state("g1", "alice_id")).generation == 1

        await admin.admin.set_opt_out("g1", "alice_id", False)
        assert (await admin._store.consent_state("g1", "alice_id")).generation == 1

        await admin.observe(_event("m2", "alice_id"))
        processed = await worker.flush("g1")
        assert processed == 1
        page = await admin._store.list_facts("g1", subject_id="alice_id", include_server=True)
        texts = [fact.text for fact in page.items]
        assert texts == ["post optin fact"]
        assert not any("paused" in text for text in texts)
    finally:
        await admin.close()
        await worker.close()
