"""Maintenance is one durable, retryable state transition across workers."""

import asyncio
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest
from pymongo.errors import OperationFailure

from icelake.config import MemoryConfig
from icelake.lifecycle.maintenance import MaintenanceService
from icelake.models.facts import FactRecord


def service(store, clock):
    return MaintenanceService(store=store, config=MemoryConfig(), clock=clock)


async def seed(store, clock):
    fact = FactRecord(
        id="old", guild_id="g", text="Alice likes tea", created_at=clock.now() - timedelta(days=365)
    )
    await store.insert_fact(fact)
    return fact


async def test_throttle_survives_new_worker_and_expires(store, fixed_clock):
    assert (await service(store, fixed_clock).run_guild("g")).skipped_reason is None
    assert (await service(store, fixed_clock).run_guild("g")).skipped_reason == "throttled"
    fixed_clock.advance(7200)
    assert (await service(store, fixed_clock).run_guild("g")).skipped_reason is None
    assert (await service(store, fixed_clock).run_guild("g", force=True)).skipped_reason is None


async def test_failed_sweep_rolls_back_facts_and_checkpoint(store, fixed_clock, monkeypatch):
    fact = await seed(store, fixed_clock)
    worker = service(store, fixed_clock)
    with monkeypatch.context() as patch:
        patch.setattr(store, "prune_to_caps", AsyncMock(side_effect=RuntimeError("sweep failed")))
        with pytest.raises(RuntimeError, match="sweep failed"):
            await worker.run_guild("g")
    assert (await store.get_fact("g", fact.id)) == fact
    assert await store.get_cursor("g", "maintenance_completed_at") is None
    assert (await worker.run_guild("g")).forgotten == 1


async def test_competing_workers_retire_once(store, fixed_clock):
    fact = await seed(store, fixed_clock)
    workers = [service(store, fixed_clock), service(store, fixed_clock)]
    reports = await asyncio.gather(*(w.run_guild("g") for w in workers), return_exceptions=True)
    # Mongo aborts a conflicting snapshot transaction; the worker's next poll
    # retries it. SQLite/reference serialize. Neither may double-retire a fact.
    for index, report in enumerate(reports):
        if isinstance(report, BaseException):
            assert isinstance(report, OperationFailure)
            assert report.has_error_label("TransientTransactionError")
            reports[index] = await workers[index].run_guild("g")
    assert sorted(r.forgotten for r in reports) == [0, 1]
    current = await store.get_fact("g", fact.id)
    assert not current.is_active
    assert current.version == fact.version + 1


async def test_recent_reinforcement_survives_sweep(store, fixed_clock):
    fact = await seed(store, fixed_clock)
    await store.reinforce_fact(
        "g",
        fact.id,
        occurrences_delta=1,
        strength=2.0,
        last_reinforced_at=fixed_clock.now(),
        expires_at=None,
        tier=fact.tier,
        confidence=1.0,
    )
    assert (await service(store, fixed_clock).run_guild("g")).forgotten == 0
    assert (await store.get_fact("g", fact.id)).is_active


async def test_local_workers_share_the_sweep_without_transaction_conflicts(store, fixed_clock):
    await seed(store, fixed_clock)
    worker = service(store, fixed_clock)
    reports = await asyncio.gather(*(worker.run_guild("g") for _ in range(4)))
    assert sum(report.forgotten for report in reports) == 1
    assert sum(report.skipped_reason == "throttled" for report in reports) == 3


async def test_mongo_reinforcement_during_sweep_aborts_stale_retirement(
    mongo_store, fixed_clock, monkeypatch
):
    fact = await seed(mongo_store, fixed_clock)
    worker = service(mongo_store, fixed_clock)
    snapshot_read, reinforced = asyncio.Event(), asyncio.Event()
    original = mongo_store.apply_forgetting

    async def pause_after_snapshot(*args, **kwargs):
        await mongo_store.get_fact("g", fact.id)
        snapshot_read.set()
        await reinforced.wait()
        return await original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(mongo_store, "apply_forgetting", pause_after_snapshot)
        pending = asyncio.create_task(worker.run_guild("g"))
        try:
            await asyncio.wait_for(snapshot_read.wait(), timeout=5)
            await mongo_store.reinforce_fact(
                "g",
                fact.id,
                occurrences_delta=1,
                strength=2.0,
                last_reinforced_at=fixed_clock.now(),
                expires_at=None,
                tier=fact.tier,
                confidence=1.0,
            )
        finally:
            reinforced.set()
        with pytest.raises(OperationFailure) as failure:
            await pending
        assert failure.value.has_error_label("TransientTransactionError")
    assert await mongo_store.get_cursor("g", "maintenance_completed_at") is None
    assert (await worker.run_guild("g")).forgotten == 0
    current = await mongo_store.get_fact("g", fact.id)
    assert current.is_active and current.strength == 2.0
