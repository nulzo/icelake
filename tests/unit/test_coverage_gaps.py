"""Always-on coverage for branches that live-service tests do not hit."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from icelake.adapters.in_memory.queue import InMemoryIngestQueue
from icelake.adapters.mongo import mapping as mongo_mapping
from icelake.api.classify import CommandAction, CommandClassifier
from icelake.api.client import _build_store, _snippet
from icelake.errors import ConfigError
from icelake.graph.writes import DirectRoster
from icelake.identity.aliases import alias_slug, is_valid_alias
from icelake.identity.guards import BotGuard
from icelake.lifecycle.strength import retention
from icelake.lifecycle.tiers import _mentions_horizon
from icelake.models.admin import MeterSnapshot
from icelake.models.common import TokenUsage
from icelake.models.facts import FactCategory, FactRecord
from icelake.models.identity import AliasSource, Resolution, ResolvedCandidate
from icelake.ports.queue import BatchKey, MessageStatus, StoredMessage
from icelake.ports.vectors import cosine
from tests.conftest import make_config

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)


def _msg(message_id: str, *, guild: str = "g1", author: str = "u1") -> StoredMessage:
    return StoredMessage(
        message_id=message_id,
        guild_id=guild,
        author_id=author,
        subject_key=author,
        content=message_id,
        created_at=NOW,
    )


class TestCosine:
    def test_mismatched_or_empty_or_zero(self) -> None:
        assert cosine((1.0,), (1.0, 0.0)) == 0.0
        assert cosine((), ()) == 0.0
        assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0


class TestClassifyWithoutLlm:
    async def test_regex_forget_and_none(self) -> None:
        classifier = CommandClassifier(None)
        forget = await classifier.classify("forget that I ever said that")
        assert forget.action is CommandAction.FORGET
        none = await classifier.classify("what do you remember about tuesday")
        assert none.action is CommandAction.NONE
        assert none.confidence == pytest.approx(0.3)


class TestClientInternals:
    def test_snippet_truncates(self) -> None:
        long = "word " * 80
        result = _snippet(long)
        assert result.endswith("…")
        assert len(result) == 160

    def test_build_store_rejects_postgres_and_unknown(self) -> None:
        from types import SimpleNamespace

        with pytest.raises(ConfigError, match="postgresql storage is not implemented"):
            _build_store(make_config(storage="postgresql://localhost/db"))
        fake = SimpleNamespace(storage=SimpleNamespace(backend="cassandra", url="cassandra://x"))
        with pytest.raises(ConfigError, match="storage backend"):
            _build_store(fake)  # type: ignore[arg-type]


class TestMongoMappingDates:
    def test_dt_in_datetime_string_and_garbage(self) -> None:
        instant = datetime(2026, 1, 2, tzinfo=UTC)
        assert mongo_mapping._dt_in(instant) is instant
        parsed = mongo_mapping._dt_in("2026-01-02T00:00:00+00:00")
        assert parsed is not None and parsed.year == 2026
        assert mongo_mapping._dt_in("not-a-date") is None
        assert mongo_mapping._dt_in(12) is None


class TestSmallHelpers:
    def test_direct_roster(self) -> None:
        roster = DirectRoster("u1", "u2")
        assert roster.knows("u1") and not roster.knows("u3")
        assert roster.user_id_for("u2") == "u2"
        assert roster.user_id_for("missing") is None
        assert roster.bind_names("hello") == "hello"
        assert roster.display_name("u1") is None

    def test_token_usage_and_meter_snapshot(self) -> None:
        assert TokenUsage(prompt_tokens=3, completion_tokens=7).total == 10
        dumped = MeterSnapshot(calls={"extraction": 2}).as_dict()
        assert dumped["calls"]["extraction"] == 2

    def test_resolution_basis(self) -> None:
        unresolved = Resolution(identifier="bob")
        assert unresolved.basis == "unresolved"
        candidate = ResolvedCandidate(
            user_id="u1",
            matched_alias="robert",
            source=AliasSource.DISPLAY_NAME,
            weight=0.9,
            confidence=0.8,
        )
        fuzzy = Resolution(identifier="bob", resolved=candidate, candidates=(candidate,))
        assert fuzzy.basis.startswith("fuzzy:")

    def test_fact_with_updates_and_inactive(self) -> None:
        fact = FactRecord(
            id="fct_1",
            guild_id="g",
            subject_id="u",
            text="likes tea",
            category=FactCategory.INTERESTS,
        )
        updated = fact.with_updates(superseded_by_id="fct_2")
        assert not updated.is_active
        assert updated.superseded_by_id == "fct_2"

    def test_bot_guard_register_many(self) -> None:
        guard = BotGuard()
        guard.register_many(("11", "22"))
        assert guard.is_bot("11") and guard.exclude(("11", "33")) == ("33",)

    def test_alias_validation_and_slug(self) -> None:
        assert is_valid_alias("a") is False
        assert alias_slug("!!!") == "unknown"

    def test_retention_floors_strength(self) -> None:
        value = retention(last_reinforced_at=NOW, now=NOW + timedelta(days=1), strength=0.1)
        assert 0.0 < value <= 1.0

    def test_mentions_horizon_this_month(self) -> None:
        assert _mentions_horizon("planning this month", short_term_days=14) == 21
        assert _mentions_horizon("no time words here", short_term_days=14) is None


class TestInMemoryQueueEdges:
    async def test_max_depth_and_max_pending(self) -> None:
        queue = InMemoryIngestQueue(max_pending=2)
        assert await queue.put_message(_msg("m1"), max_depth=1) is True
        assert await queue.put_message(_msg("m2"), max_depth=1) is False
        capped = InMemoryIngestQueue(max_pending=1)
        assert await capped.put_message(_msg("a")) is True
        assert await capped.put_message(_msg("b")) is False

    async def test_claim_empty_and_complete_unknown(self) -> None:
        queue = InMemoryIngestQueue()
        key = BatchKey(guild_id="g1", subject_key="nobody")
        claim = await queue.claim_batch(key, now=NOW, lease_seconds=30, owner="w1", limit=4)
        assert claim.messages == ()
        assert await queue.complete_messages(("missing",), owner="w1") == 0
        assert await queue.dead_letter_messages(("missing",), owner="w1") == 0

    async def test_lease_renew_release_prune_recent(self) -> None:
        queue = InMemoryIngestQueue()
        await queue.put_message(_msg("keep"))
        key = BatchKey(guild_id="g1", subject_key="u1")
        claim = await queue.claim_batch(key, now=NOW, lease_seconds=30, owner="w1", limit=1)
        assert await queue.renew_lease(key, owner="other", now=NOW, lease_seconds=60) is False
        assert await queue.renew_lease(key, owner="w1", now=NOW, lease_seconds=60) is True
        await queue.release_key(key, owner="w1")
        await queue.complete_messages(tuple(m.message_id for m in claim.messages), owner="w1")
        recent = await queue.recent_messages("g1", 10)
        assert recent[0].message_id == "keep"
        pruned = await queue.prune_processed(older_than=NOW + timedelta(seconds=1))
        assert pruned == 1

    async def test_requeue_filters_guild(self) -> None:
        queue = InMemoryIngestQueue()
        await queue.put_message(_msg("poison", guild="g2"))
        key = BatchKey(guild_id="g2", subject_key="u1")
        claim = await queue.claim_batch(key, now=NOW, lease_seconds=30, owner="w1", limit=1)
        await queue.dead_letter_messages(tuple(m.message_id for m in claim.messages), owner="w1")
        assert queue._messages["poison"].status is MessageStatus.DEAD
        assert await queue.requeue_dead_letters("g1") == 0
        assert await queue.requeue_dead_letters("g2") == 1
