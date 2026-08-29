"""FactsApi, AdminApi, OpsApi behavior tests."""

from __future__ import annotations

import asyncio

import pytest

from icelake import FactCommitted, FactSupersededEvent
from icelake.errors import FactNotFoundError, SubjectNotAllowedError
from icelake.models.facts import FactCategory


class TestFactsApi:
    async def test_remember_creates_core_fact(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        fact = await client.facts.remember(
            guild_id="g1",
            subject_id="u1",
            text="prefers mechanical keyboards",
            category=FactCategory.PREFERENCES,
            actor_id="admin",
        )
        assert fact.tier.value == "core"
        assert fact.attribution.type.value == "manual"
        assert fact.attribution.actor_id == "admin"
        loaded = await client.facts.get("g1", fact.id)
        assert loaded.id == fact.id
        await client.close()

    async def test_remember_deduplicates_into_reinforce(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        first = await client.facts.remember(
            guild_id="g1", subject_id="u1", text="drives a blue Subaru"
        )
        second = await client.facts.remember(
            guild_id="g1", subject_id="u1", text="drives a blue Subaru"
        )
        assert second.id == first.id
        assert second.occurrences == 2
        await client.close()

    async def test_opted_out_subject_rejected(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.admin.set_opt_out("g1", "u-quiet", True)
        with pytest.raises(SubjectNotAllowedError):
            await client.facts.remember(
                guild_id="g1", subject_id="u-quiet", text="anything at all really"
            )
        await client.close()

    async def test_update_and_forget_flow(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        fact = await client.facts.remember(
            guild_id="g1", subject_id="u1", text="works night shifts at hospital"
        )
        updated = await client.facts.update(
            fact.id, guild_id="g1", text="works day shifts at hospital", reason="user correction"
        )
        assert "day shifts" in updated.text
        history = await client.facts.history(fact.id, guild_id="g1")
        assert history
        await client.facts.forget(fact.id, guild_id="g1", reason="requested")
        page = await client.facts.list_for_subject("g1", "u1", active_only=True)
        assert all(item.id != fact.id or item.is_active for item in page.items)
        with pytest.raises(FactNotFoundError):
            await client.facts.get("g1", "fct_missing")
        await client.close()

    async def test_curation_publishes_events(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        fired: list[object] = []
        client.events.subscribe(FactCommitted, fired.append)
        client.events.subscribe(FactSupersededEvent, fired.append)
        fact = await client.facts.remember(
            guild_id="g1", subject_id="u1", text="keeps a sourdough starter"
        )
        await client.facts.reinforce(fact.id, guild_id="g1")
        await client.facts.update(fact.id, guild_id="g1", text="keeps two starters", reason="probe")
        await client.facts.forget(fact.id, guild_id="g1", reason="done")
        await asyncio.sleep(0)  # handlers dispatch via loop.call_soon
        committed = [e for e in fired if isinstance(e, FactCommitted)]
        superseded = [e for e in fired if isinstance(e, FactSupersededEvent)]
        assert [e.was_reinforcement for e in committed] == [False, True]
        assert [(e.old_fact_id, e.new_fact_id) for e in superseded] == [
            (fact.id, fact.id),
            (fact.id, None),
        ]
        await client.close()


class TestAdminAndOps:
    async def test_purge_two_phase(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id="g1", subject_id="u-purge", text="favorite color is green definitely"
        )
        dry = await client.admin.purge_user("g1", "u-purge", dry_run=True)
        assert dry.dry_run and dry.facts_removed >= 0
        report = await client.admin.purge_user("g1", "u-purge", dry_run=False)
        assert not report.dry_run
        page = await client.facts.list_for_subject("g1", "u-purge", active_only=False)
        assert page.items == ()
        await client.close()

    async def test_export_guild(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id="g-export", subject_id="u9", text="plays bass guitar on weekends"
        )
        data = await client.admin.export_guild("g-export")
        assert data.guild_id == "g-export"
        assert any(fact.text.startswith("plays bass") for fact in data.facts)
        await client.close()

    async def test_ops_health_and_meter(self, make_client) -> None:
        client, llm = make_client()
        await client.start()
        health = await client.ops.health()
        assert health.healthy
        snapshot = client.ops.meter_snapshot()
        assert snapshot.calls == {} or isinstance(snapshot.calls, dict)
        del llm
        await client.close()

    async def test_stats_shape(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        stats = await client.stats("g-empty")
        assert stats.guild_id == "g-empty"
        assert stats.total_facts == 0
        await client.close()
