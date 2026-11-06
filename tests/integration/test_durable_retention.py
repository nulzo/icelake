"""Durable memory outlives silence; explicit expiry and user deletion still win."""

from datetime import timedelta

import pytest

from icelake.config import LifecycleConfig, MemoryConfig
from icelake.lifecycle.maintenance import MaintenanceService
from icelake.lifecycle.tiers import assign_tier
from icelake.models.facts import FactCategory, FactRecord, MemoryTier


@pytest.mark.parametrize(
    "category",
    [
        FactCategory.PERSONAL,
        FactCategory.PREFERENCES,
        FactCategory.INTERESTS,
        FactCategory.PROFESSIONAL,
        FactCategory.RELATIONSHIPS,
        FactCategory.PERSONALITY,
        FactCategory.EXPERIENCES,
    ],
)
def test_durable_information_has_no_calendar_expiry(category):
    tier, expiry = assign_tier(
        text="likes making furniture",
        category=category,
        confidence=0.8,
        occurrences=1,
        manual=False,
        is_server_fact=False,
        lifecycle=LifecycleConfig(),
    )
    assert tier is MemoryTier.LONG_TERM
    assert expiry is None


def test_explicit_long_term_retention_configuration_still_works():
    _, expiry = assign_tier(
        text="likes coffee",
        category=FactCategory.PREFERENCES,
        confidence=0.8,
        occurrences=1,
        manual=False,
        is_server_fact=False,
        lifecycle=LifecycleConfig(long_term_days=730),
    )
    assert expiry == timedelta(days=730)


async def test_silence_does_not_forget_durable_facts_but_expiry_and_deletion_do(store, fixed_clock):
    now = fixed_clock.now()
    old = now - timedelta(days=730)
    kept = FactRecord(
        id="durable",
        guild_id="g",
        subject_id="u",
        text="likes coffee",
        tier=MemoryTier.LONG_TERM,
        created_at=old,
        last_reinforced_at=old,
    )
    temporary = kept.model_copy(
        update={
            "id": "temporary",
            "tier": MemoryTier.SHORT_TERM,
            "expires_at": now - timedelta(days=1),
        }
    )
    deleted = kept.model_copy(update={"id": "deleted", "valid_until": now - timedelta(days=1)})
    for fact in (kept, temporary, deleted):
        await store.insert_fact(fact)
    worker = MaintenanceService(store=store, config=MemoryConfig(), clock=fixed_clock)
    report = await worker.run_guild("g")
    assert report.forgotten == 0
    assert report.expired == 1
    assert (await store.get_fact("g", "durable")).is_active
    assert not (await store.get_fact("g", "temporary")).is_active
    assert not (await store.get_fact("g", "deleted")).is_active


def test_retention_extension_is_monotonic_and_does_not_resurrect():
    from datetime import UTC, datetime

    from icelake.lifecycle.tiers import retention_extension
    from icelake.models.facts import Attribution, AttributionType

    now = datetime(2026, 11, 5, tzinfo=UTC)
    fact = FactRecord(
        id="f",
        guild_id="g",
        subject_id="u",
        text="likes coffee",
        category=FactCategory.PREFERENCES,
        tier=MemoryTier.MID_TERM,
        created_at=now - timedelta(days=5),
        expires_at=now + timedelta(days=40),
    )
    update = retention_extension(fact, now=now, lifecycle=LifecycleConfig())
    assert update is not None and update.expires_at is None
    assert update.tier is MemoryTier.LONG_TERM
    assert (
        retention_extension(
            fact.model_copy(update=update.model_dump()), now=now, lifecycle=LifecycleConfig()
        )
        is None
    )
    for changes in (
        {"valid_until": now},
        {"superseded_by_id": "new"},
        {"expires_at": now},
        {"tier": MemoryTier.SHORT_TERM},
        {"attribution": Attribution(type=AttributionType.MANUAL)},
        {"text": "going on a trip tomorrow"},
        {"expires_at": None},
    ):
        assert (
            retention_extension(
                fact.model_copy(update=changes), now=now, lifecycle=LifecycleConfig()
            )
            is None
        )
    general = fact.model_copy(update={"category": FactCategory.GENERAL})
    extended = retention_extension(general, now=now, lifecycle=LifecycleConfig())
    assert extended is not None and extended.expires_at == fact.created_at + timedelta(days=365)
    assert (
        retention_extension(general, now=now, lifecycle=LifecycleConfig(mid_term_days=10)) is None
    )
