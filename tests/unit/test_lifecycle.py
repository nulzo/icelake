from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from icelake.config import LifecycleConfig
from icelake.lifecycle.strength import (
    reinforced_strength,
    retention,
    should_forget,
    strength_signal,
)
from icelake.lifecycle.tiers import assign_tier
from icelake.models.facts import FactCategory, FactRecord, MemoryTier


class TestTierAssignment:
    def test_manual_is_core_and_never_expires(self) -> None:
        tier, expiry = assign_tier(
            text="anything",
            category=FactCategory.GENERAL,
            confidence=0.6,
            occurrences=1,
            manual=True,
            is_server_fact=False,
            lifecycle=LifecycleConfig(),
        )
        assert tier.value == "core"
        assert expiry is None

    def test_server_rules_high_confidence_core(self) -> None:
        tier, expiry = assign_tier(
            text="no politics in general",
            category=FactCategory.RULES,
            confidence=0.9,
            occurrences=1,
            manual=False,
            is_server_fact=True,
            lifecycle=LifecycleConfig(),
        )
        assert tier.value == "core"
        assert expiry is None

    def test_time_horizon_short_term(self) -> None:
        tier, expiry = assign_tier(
            text="party at my place tomorrow",
            category=FactCategory.EXPERIENCES,
            confidence=0.9,
            occurrences=1,
            manual=False,
            is_server_fact=False,
            lifecycle=LifecycleConfig(),
        )
        assert tier.value == "short_term"
        assert expiry is not None and expiry.days == 7

    def test_short_term_days_knob_applies_to_week_scale_horizon(self) -> None:
        tier, expiry = assign_tier(
            text="party at my place tomorrow",
            category=FactCategory.EXPERIENCES,
            confidence=0.9,
            occurrences=1,
            manual=False,
            is_server_fact=False,
            lifecycle=LifecycleConfig(short_term_days=14),
        )
        assert tier.value == "short_term"
        assert expiry is not None and expiry.days == 14

    def test_tonight_horizon_three_days(self) -> None:
        tier, expiry = assign_tier(
            text="movie night tonight",
            category=FactCategory.EXPERIENCES,
            confidence=0.9,
            occurrences=1,
            manual=False,
            is_server_fact=False,
            lifecycle=LifecycleConfig(),
        )
        assert tier.value == "short_term"
        assert expiry is not None and expiry.days == 3

    def test_durable_category_long_term(self) -> None:
        tier, _ = assign_tier(
            text="works as a nurse",
            category=FactCategory.PROFESSIONAL,
            confidence=0.8,
            occurrences=1,
            manual=False,
            is_server_fact=False,
            lifecycle=LifecycleConfig(),
        )
        assert tier.value == "long_term"

    def test_default_mid_term(self) -> None:
        tier, expiry = assign_tier(
            text="enjoys hiking",
            category=FactCategory.INTERESTS,
            confidence=0.7,
            occurrences=1,
            manual=False,
            is_server_fact=False,
            lifecycle=LifecycleConfig(),
        )
        assert tier.value == "mid_term"
        assert expiry is not None and expiry.days == 45

    def test_high_confidence_repeated_personal_graduates_core(self) -> None:
        tier, expiry = assign_tier(
            text="has two siblings",
            category=FactCategory.PERSONAL,
            confidence=0.96,
            occurrences=4,
            manual=False,
            is_server_fact=False,
            lifecycle=LifecycleConfig(),
        )
        assert tier.value == "core"
        assert expiry is None


class TestStrengthDecay:
    NOW = datetime(2026, 8, 24, tzinfo=UTC)

    def test_full_retention_immediately_after_reinforcement(self) -> None:
        value = retention(
            last_reinforced_at=self.NOW,
            now=self.NOW,
            strength=2.0,
        )
        assert value == 1.0

    def test_decay_is_monotonic_in_time(self) -> None:
        base = self.NOW
        r1 = retention(last_reinforced_at=base, now=base + timedelta(days=1), strength=2.0)
        r5 = retention(last_reinforced_at=base, now=base + timedelta(days=5), strength=2.0)
        assert 0 < r5 < r1 < 1.0

    def test_stronger_memory_decays_slower(self) -> None:
        base = self.NOW
        later = base + timedelta(days=10)
        weak = retention(last_reinforced_at=base, now=later, strength=1.0)
        strong = retention(last_reinforced_at=base, now=later, strength=8.0)
        assert strong > weak

    def test_stability_days_slows_decay(self) -> None:
        base = self.NOW
        later = base + timedelta(days=10)
        fast = retention(last_reinforced_at=base, now=later, strength=1.0)
        slow = retention(last_reinforced_at=base, now=later, strength=1.0, stability_days=7.0)
        assert slow > fast

    def test_stability_days_scales_the_curve_exactly(self) -> None:
        import math

        base = self.NOW
        later = base + timedelta(days=3)
        value = retention(last_reinforced_at=base, now=later, strength=2.0, stability_days=7.0)
        assert value == pytest.approx(math.exp(-3 / 14))

    def test_default_stability_preserves_legacy_curve(self) -> None:
        import math

        base = self.NOW
        later = base + timedelta(days=3)
        assert retention(
            last_reinforced_at=base, now=later, strength=1.0
        ) == pytest.approx(math.exp(-3))

    def test_reinforcement_adds_strength(self) -> None:
        assert reinforced_strength(2.0) > 2.0

    def test_forgetting_gate(self) -> None:
        assert should_forget(
            retention_value=0.01,
            tier=MemoryTier.MID_TERM,
            manual=False,
            forget_retention_floor=0.05,
        )
        assert not should_forget(
            retention_value=0.01, tier=MemoryTier.CORE, manual=False, forget_retention_floor=0.05
        )
        assert not should_forget(
            retention_value=0.01, tier=MemoryTier.MID_TERM, manual=True, forget_retention_floor=0.05
        )
        assert not should_forget(
            retention_value=0.5, tier=MemoryTier.MID_TERM, manual=False, forget_retention_floor=0.05
        )

    def test_strength_signal_bounded(self) -> None:
        signal = strength_signal(strength=100.0, retention_value=1.0)
        assert 0.0 <= signal <= 1.0


class TestPruneVictimSelection:
    def test_weakest_and_short_term_go_first(self) -> None:
        from datetime import UTC, datetime

        from icelake.lifecycle.prune import select_prune_victims
        from icelake.models.facts import (
            Attribution,
            AttributionType,
            FactCategory,
            FactRecord,
            MemoryTier,
        )

        now = datetime(2026, 8, 28, tzinfo=UTC)

        def fact(
            fact_id: str, *, tier: MemoryTier, strength: float, manual: bool = False
        ) -> FactRecord:
            return FactRecord(
                id=fact_id,
                guild_id="g1",
                subject_id="u1",
                text=f"fact {fact_id} with enough words to be real",
                category=FactCategory.INTERESTS,
                tier=tier,
                strength=strength,
                confidence=0.8,
                created_at=now,
                attribution=Attribution(
                    type=AttributionType.MANUAL if manual else AttributionType.SELF
                ),
            )

        victims = select_prune_victims(
            (
                fact("weak", tier=MemoryTier.SHORT_TERM, strength=1.0),
                fact("mid", tier=MemoryTier.MID_TERM, strength=2.0),
                fact("core", tier=MemoryTier.CORE, strength=2.0),
                fact("pinned", tier=MemoryTier.SHORT_TERM, strength=1.0, manual=True),
            ),
            cap=2,
        )
        assert tuple(v.id for v in victims) == ("weak",)


class TestSelectForgottenFacts:
    NOW = datetime(2026, 8, 28, tzinfo=UTC)

    def _fact(
        self,
        fact_id: str,
        *,
        days_old: float,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
        manual: bool = False,
        valid_until: datetime | None = None,
    ) -> FactRecord:
        from icelake.models.facts import Attribution, AttributionType

        at = self.NOW - timedelta(days=days_old)
        return FactRecord(
            id=fact_id,
            guild_id="g1",
            subject_id="u1",
            text=f"fact {fact_id} with enough words to be real",
            category=FactCategory.INTERESTS,
            tier=tier,
            strength=1.0,
            confidence=0.8,
            created_at=at,
            last_reinforced_at=at,
            valid_until=valid_until,
            attribution=Attribution(
                type=AttributionType.MANUAL if manual else AttributionType.SELF
            ),
        )

    def _select(self, records, *, stability_days: float = 1.0):
        from icelake.lifecycle.forget import select_forgotten_facts

        return select_forgotten_facts(
            records,
            now=self.NOW,
            retention_floor=0.05,
            stability_days=stability_days,
        )

    def test_stale_fact_forgotten_fresh_fact_kept(self) -> None:
        victims = self._select(
            (self._fact("stale", days_old=10), self._fact("fresh", days_old=1))
        )
        assert tuple(v.id for v in victims) == ("stale",)

    def test_core_and_manual_facts_are_exempt(self) -> None:
        victims = self._select(
            (
                self._fact("core", days_old=365, tier=MemoryTier.CORE),
                self._fact("pinned", days_old=365, manual=True),
            )
        )
        assert victims == ()

    def test_inactive_facts_are_skipped(self) -> None:
        victims = self._select(
            (self._fact("gone", days_old=10, valid_until=self.NOW),)
        )
        assert victims == ()

    def test_stability_days_extends_lifetime(self) -> None:
        records = (self._fact("borderline", days_old=10),)
        assert len(self._select(records)) == 1
        assert self._select(records, stability_days=7.0) == ()
