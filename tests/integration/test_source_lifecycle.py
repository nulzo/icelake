"""Evidence must survive retries without surviving its own invalidation."""

from datetime import UTC, datetime, timedelta

import pytest

from icelake import Citations, DiscordMemory, MemoryConfig
from icelake.api.facts_api import SchemaError
from icelake.api.sources import SourceRevision, invalidate_source, source_is_current
from icelake.models.facts import SourceRef
from icelake.permissions import Audience
from icelake.ports.citations import CitationVerification

NOW = datetime(2026, 10, 23, tzinfo=UTC)


def source(message="m1", text="I like coffee", channel="public"):
    return SourceRef(
        message_id=message,
        guild_id="g",
        channel_id=channel,
        author_id="100000000000000101",
        content_snippet=text,
        created_at=NOW,
    )


@pytest.fixture
async def memory(store, monkeypatch):
    memory = DiscordMemory(
        MemoryConfig(workers={"enabled": False}), store=store, llm=None, embedder=None, vectors=None
    )
    await memory.start()
    yield memory
    from unittest.mock import AsyncMock

    with monkeypatch.context() as patch:
        patch.setattr(store, "close", AsyncMock())
        await memory.close()


async def test_correction_is_atomic_idempotent_and_keeps_separate_evidence(memory):
    old = await memory.facts.remember(
        guild_id="g", subject_id="100000000000000101", text="likes coffee", source_refs=(source(),)
    )
    kwargs = dict(
        guild_id="g",
        text="prefers tea now",
        actor_id="100000000000000101",
        source_refs=(source("m2", "I prefer tea now"),),
        idempotency_key="request-1",
    )
    new = await memory.facts.update(old.id, **kwargs)
    replay = await memory.facts.update(old.id, **kwargs)
    assert replay.id == new.id != old.id
    assert new.supersedes_id == old.id
    historical = await memory.facts.get("g", old.id)
    assert not historical.is_active and historical.citations == (source(),)
    assert new.citations == kwargs["source_refs"]
    with pytest.raises(SchemaError, match="different inputs"):
        await memory.facts.update(old.id, **{**kwargs, "text": "prefers juice now"})


async def test_source_edit_and_delete_fence_old_observations(memory, store):
    fact = await memory.facts.remember(
        guild_id="g", subject_id="100000000000000101", text="likes coffee", source_refs=(source(),)
    )
    assert (
        await invalidate_source(
            store, "g", "m1", SourceRevision(revision="v2", changed_at=NOW + timedelta(days=1))
        )
        == 1
    )
    assert not (await memory.facts.get("g", fact.id)).is_active
    assert not await source_is_current(store, "g", "m1", "")
    assert await source_is_current(store, "g", "m1", "v2")
    with pytest.raises(SchemaError, match="edited or deleted"):
        await memory.facts.remember(
            guild_id="g",
            subject_id="100000000000000101",
            text="likes stale coffee",
            source_refs=(source(),),
        )
    await memory.delete_source("g", "m1", deleted_at=NOW + timedelta(days=2))
    await invalidate_source(
        store, "g", "m1", SourceRevision(revision="v3", changed_at=NOW + timedelta(days=3))
    )
    assert not await source_is_current(store, "g", "m1", "v3")


async def test_mixed_citations_revalidate_after_verification(memory):
    await memory.facts.remember(
        guild_id="g", subject_id="100000000000000101", text="likes coffee", source_refs=(source(),)
    )
    context = await memory.prompt_context(
        guild_id="g",
        asker_id="100000000000000101",
        text="coffee",
        audience=Audience(channel_id="public", requester_id="100000000000000101"),
    )
    assert context.citations
    assert context.citations[0].snippet == "I like coffee"
    registry = Citations.from_prompt_context(context)
    registry.add_source("https://example.org/", excerpt="other evidence")

    class DeletingVerifier:
        async def verify(self, candidates):
            await memory.delete_source("g", "m1")
            return CitationVerification(supported_ids=tuple(c.id for c in candidates))

    reply = await memory.cite("This person likes coffee.", registry, verifier=DeletingVerifier())
    assert not any(s.kind == "memory" for s in reply.sources)
    assert "discord.com" not in reply.text


async def test_reinforcement_preserves_each_original_message(memory):
    first = await memory.facts.remember(
        guild_id="g", subject_id="100000000000000101", text="likes coffee", source_refs=(source(),)
    )
    await memory.facts.remember(
        guild_id="g",
        subject_id="100000000000000101",
        text="likes coffee",
        source_refs=(source("m2", "still love coffee"),),
    )
    current = await memory.facts.get("g", first.id)
    assert {s.message_id for s in current.citations} == {"m1", "m2"}
    context = await memory.prompt_context(
        guild_id="g",
        asker_id="100000000000000101",
        text="coffee",
        audience=Audience(channel_id="public", requester_id="100000000000000101"),
    )
    assert {c.snippet for c in context.citations} == {"I like coffee", "still love coffee"}


async def test_edit_to_empty_content_still_invalidates_old_memory(memory):
    from icelake import MessageEvent, ObserveStatus

    fact = await memory.facts.remember(
        guild_id="g",
        subject_id="100000000000000101",
        text="likes coffee",
        source_refs=(source(),),
    )
    receipt = await memory.observe(
        MessageEvent(
            message_id="m1",
            guild_id="g",
            channel_id="public",
            author_id="100000000000000101",
            content="",
            created_at=NOW,
            edited=True,
            edited_at=NOW + timedelta(hours=1),
        )
    )
    assert receipt.status is ObserveStatus.IGNORED
    assert not (await memory.facts.get("g", fact.id)).is_active


async def test_budget_reservations_share_ceiling_and_settle_in_original_period(store):
    from icelake.adapters.meter import UsageMeter
    from icelake.config import BudgetsConfig
    from icelake.errors import BudgetExceededError
    from icelake.ports.clock import FixedClock

    clock = FixedClock(NOW)
    a = UsageMeter(BudgetsConfig(guild_daily_prompt_tokens=100), clock, store=store)
    b = UsageMeter(BudgetsConfig(guild_daily_prompt_tokens=100), clock, store=store)
    reservation = await a.reserve("g", prompt_tokens=80)
    with pytest.raises(BudgetExceededError):
        await b.reserve("g", prompt_tokens=30)
    await a.settle(reservation, prompt_tokens=20)
    await b.reserve("g", prompt_tokens=80)
    assert await store.guild_token_usage("g", day_key="2026-10-23", month_key="2026-10") == (
        100,
        100,
    )
