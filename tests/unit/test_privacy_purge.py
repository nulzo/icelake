from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from icelake.adapters.in_memory.store import InMemoryStore
from icelake.adapters.sqlite.store import SqliteStore
from icelake.api.groups import AdminApi
from icelake.consolidation.service import ConsolidationService
from icelake.consolidation.validity import summary_is_valid, summary_source
from icelake.models.facts import FactHistoryEntry, FactHistoryKind, FactRecord, ProfileSummary
from icelake.ports.clock import FixedClock
from icelake.ports.llm import ChatRequest, ChatResponse
from icelake.ports.store import MemoryStore
from tests.conftest import ScriptedLLM, make_config


@pytest.mark.parametrize("backend", ["in_memory", "sqlite"])
async def test_purge_preserves_consent_and_erases_history(backend: str, tmp_path: Path) -> None:
    store = (
        InMemoryStore()
        if backend == "in_memory"
        else SqliteStore(f"sqlite:///{tmp_path / 'privacy.db'}")
    )
    await store.setup()
    try:
        await store.insert_fact(FactRecord(id="f", guild_id="g", subject_id="u", text="private"))
        await store.append_history(
            "g",
            "f",
            FactHistoryEntry(
                at=datetime.now(UTC), kind=FactHistoryKind.CREATED, detail="private history"
            ),
        )
        await store.put_summary(ProfileSummary(guild_id="g", subject_id="u", text="private"))
        await store.set_opt_out("g", "u", True)
        admin = AdminApi(store)
        await admin.purge_user("g", "u", dry_run=True)
        assert await store.get_opt_out("g", "u")
        assert await store.get_fact("g", "f") is not None
        assert await store.get_history("g", "f")
        await admin.purge_user("g", "u", dry_run=False)
        assert await store.get_summary("g", "u") is None
        assert await store.get_opt_out("g", "u")
        assert await store.get_fact("g", "f") is None
        assert await store.get_history("g", "f") == ()
    finally:
        await store.close()


@pytest.mark.parametrize("backend", ["in_memory", "sqlite"])
@pytest.mark.parametrize("mutation", ["update", "forget", "expire", "purge", "optout"])
async def test_stale_summary_unavailable(backend: str, mutation: str, tmp_path: Path) -> None:
    store = (
        InMemoryStore()
        if backend == "in_memory"
        else SqliteStore(f"sqlite:///{tmp_path / 'summary.db'}")
    )
    await store.setup()
    try:
        fact = FactRecord(id="f", guild_id="g", subject_id="u", text="private")
        await store.insert_fact(fact)
        summary = ProfileSummary(
            guild_id="g", subject_id="u", text="private", sources=(summary_source(fact),)
        )
        await store.put_summary(summary)
        assert await store.get_summary("g", "u") == summary
        now = datetime.now(UTC)
        if mutation == "update":
            await store.update_fact_fields("g", "f", text="changed", updated_at=now)
        elif mutation == "forget":
            await store.transition_fact("g", "f", valid_until=now, updated_at=now)
        elif mutation == "expire":
            await store.update_fact_fields(
                "g", "f", expires_at=now - timedelta(seconds=1), updated_at=now
            )
        elif mutation == "purge":
            await store.purge_user_data("g", "u", dry_run=False)
        else:
            await store.set_opt_out("g", "u", True)
        assert await store.get_summary("g", "u") is None
        await store.put_summary(summary)
        assert await store.get_summary("g", "u") is None
    finally:
        await store.close()


@pytest.mark.parametrize("owner", ["other", None])
@pytest.mark.parametrize("subject", ["u", None])
async def test_summary_requires_source_owner(
    store: MemoryStore, owner: str | None, subject: str | None
) -> None:
    fact = FactRecord(
        id="f", guild_id="g", subject_id=owner, text="private", related_user_ids=("u",)
    )
    await store.insert_fact(fact)
    summary = ProfileSummary(
        guild_id="g", subject_id=subject, text="private", sources=(summary_source(fact),)
    )
    await store.put_summary(summary)
    assert (await store.get_summary("g", subject) is not None) == (owner == subject)


async def test_summary_validity_batches_sources_and_ignores_access_changes(
    store: MemoryStore,
) -> None:
    now = datetime.now(UTC)
    facts = tuple(
        FactRecord(id=str(i), guild_id="g", subject_id="u", text=str(i)) for i in range(3)
    )
    for fact in facts:
        await store.insert_fact(fact)
    summary = ProfileSummary(
        guild_id="g",
        subject_id="u",
        text="digest",
        sources=tuple(summary_source(fact) for fact in facts),
    )
    await store.put_summary(summary)
    await store.touch_facts("g", tuple(fact.id for fact in facts), accessed_at=now)
    await store.reinforce_fact(
        "g",
        "0",
        occurrences_delta=1,
        strength=5,
        last_reinforced_at=now,
        expires_at=None,
        tier=facts[0].tier,
        confidence=0.9,
    )
    with (
        patch.object(store, "get_fact", side_effect=AssertionError("per-fact read")),
        patch.object(store, "get_facts", wraps=store.get_facts) as batch,
    ):
        assert await store.get_summary("g", "u") == summary
        batch.assert_awaited_once_with("g", ("0", "1", "2"))


@pytest.mark.parametrize("offset_hours", [None, 0, -7, 5])
async def test_summary_expiry_uses_supplied_time(offset_hours: int | None) -> None:
    store = InMemoryStore()
    clock = FixedClock(datetime(2100, 1, 1, tzinfo=UTC))
    expiry = clock.now() + timedelta(seconds=1)
    expiry = (
        expiry.replace(tzinfo=None)
        if offset_hours is None
        else expiry.astimezone(timezone(timedelta(hours=offset_hours)))
    )
    fact = FactRecord(id="f", guild_id="g", subject_id="u", text="fact", expires_at=expiry)
    await store.insert_fact(fact)
    summary = ProfileSummary(
        guild_id="g", subject_id="u", text="digest", sources=(summary_source(fact),)
    )
    assert await summary_is_valid(store, summary, now=clock.now().replace(tzinfo=None))
    clock.advance(1)
    assert not await summary_is_valid(store, summary, now=clock.now())
    await store.put_summary(summary)
    assert await store.get_summary("g", "u") == summary


@pytest.mark.parametrize("mutation", ["update", "optout", "expire"])
async def test_summary_changed_during_llm_is_not_published(
    store: MemoryStore, mutation: str
) -> None:
    clock = FixedClock(datetime(2100, 1, 1, tzinfo=UTC))
    facts = tuple(
        FactRecord(
            id=str(i),
            guild_id="g",
            subject_id="u",
            text=str(i),
            expires_at=clock.now() + timedelta(seconds=1),
        )
        for i in range(2)
    )
    for fact in facts:
        await store.insert_fact(fact)

    async def complete(request: ChatRequest) -> ChatResponse:
        if mutation == "update":
            await store.update_fact_fields("g", "0", text="changed", updated_at=clock.now())
        elif mutation == "optout":
            await store.set_opt_out("g", "u", True)
        else:
            clock.advance(2)
        return ChatResponse(text="stale digest", model="test")

    llm = ScriptedLLM()
    service = ConsolidationService(
        store=store,
        llm=llm,
        embedder=None,
        config=make_config(),
        clock=clock,
    )
    with (
        patch.object(llm, "complete", side_effect=complete) as completion,
        patch.object(store, "put_summary", wraps=store.put_summary) as publish,
    ):
        assert await service.regenerate_profile(guild_id="g", subject_id="u") is None
        completion.assert_awaited_once()
        publish.assert_not_awaited()


async def test_consolidation_strict_bounded_sources_and_batched_embeddings(
    store: MemoryStore,
) -> None:
    clock = FixedClock(datetime(2100, 1, 1, tzinfo=UTC))
    facts = (
        FactRecord(id="weak", guild_id="g", subject_id="u", text="weak"),
        FactRecord(id="strong", guild_id="g", subject_id="u", text="strong", strength=3),
        FactRecord(
            id="other",
            guild_id="g",
            subject_id="other",
            text="private",
            related_user_ids=("u",),
            strength=10,
        ),
    )
    for fact in facts:
        await store.insert_fact(fact)
    llm = ScriptedLLM({"summarize": "digest"})
    embedder = AsyncMock()
    embedder.embed.return_value = ((1.0, 0.0),) * 3
    config = make_config(extraction={"auto_consolidate_after_adds": 3})
    service = ConsolidationService(
        store=store,
        llm=llm,
        embedder=embedder,
        config=config,
        clock=clock,
    )
    with (
        patch.object(store, "top_strength_facts", side_effect=AssertionError("inclusive read")),
        patch.object(store, "list_facts", wraps=store.list_facts) as listing,
    ):
        assert await service.maybe_refresh_profile(guild_id="g", subject_id="u", adds=1) is None
        assert not llm.calls
        summary = await service.regenerate_profile(guild_id="g", subject_id="u")
        assert summary is not None
        assert tuple(source.fact_id for source in summary.sources) == ("strong", "weak")
        assert summary.source_fact_count == 2
        assert listing.await_count == 2
        listing.assert_awaited_with("g", subject_id="u", limit=config.lifecycle.max_facts_per_user)
    embedder.embed.assert_awaited_once_with(("digest", "strong", "weak"))
    assert "private" not in llm.calls[0].messages[-1].content
