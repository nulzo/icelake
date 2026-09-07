"""Boundary model contract tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from icelake.ids import prefixed, ulid
from icelake.models.common import ensure_aware
from icelake.models.events import MessageEvent
from icelake.models.facts import FactRecord, MemoryTier, ProfileSummary
from icelake.models.identity import AliasSource
from icelake.models.retrieval import (
    CHANNELS_DEFAULT,
    ChannelName,
    channels,
    discovery_pairs,
)


def test_ulid_shape_and_uniqueness() -> None:
    first = ulid()
    second = ulid()
    assert len(first) == 26 and len(second) == 26
    assert first != second
    assert prefixed("fct").startswith("fct_")


def test_message_event_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError, match="naive"):
        MessageEvent(
            message_id="1",
            guild_id="g",
            channel_id="c",
            author_id="a",
            content="hi",
            created_at=datetime(2026, 1, 1),  # naive
        )


def test_message_event_accepts_aware_datetime() -> None:
    event = MessageEvent(
        message_id="1",
        guild_id="g",
        channel_id="c",
        author_id="a",
        content="hi",
        created_at=datetime.now(UTC),
    )
    assert event.created_at.tzinfo is not None


def test_message_event_metadata_bounds() -> None:
    base = dict(
        message_id="1",
        guild_id="g",
        channel_id="c",
        author_id="a",
        content="x",
        created_at=datetime.now(UTC),
    )
    with pytest.raises(ValidationError):
        MessageEvent(**base, metadata={f"k{i}": "v" for i in range(20)})
    with pytest.raises(ValidationError):
        MessageEvent(**base, metadata={"k": "v" * 300})


def test_ensure_aware_rejects_naive() -> None:
    with pytest.raises(ValueError):
        ensure_aware(datetime.now())


def test_fact_record_active_and_server_properties() -> None:
    record = FactRecord(id="fct_1", guild_id="g", text="likes movies")
    assert record.is_server_fact is True
    assert record.is_active is True
    owned = FactRecord(id="fct_2", guild_id="g", subject_id="u", text="likes tea")
    assert not owned.is_server_fact


def test_tier_prune_priority_order() -> None:
    assert (
        MemoryTier.SHORT_TERM.prune_priority
        < MemoryTier.MID_TERM.prune_priority
        < MemoryTier.LONG_TERM.prune_priority
        < MemoryTier.CORE.prune_priority
    )


def test_alias_source_ranking() -> None:
    assert (
        AliasSource.DISCORD_USERNAME.rank
        > AliasSource.DISPLAY_NAME.rank
        > AliasSource.ENTITY_TAG.rank
    )


def test_channel_set_composition() -> None:
    custom = channels(ChannelName.VECTOR, ChannelName.KEYWORD)
    assert ChannelName.VECTOR in custom
    assert ChannelName.GRAPH_HOP not in CHANNELS_DEFAULT
    assert CHANNELS_DEFAULT | {ChannelName.GRAPH_HOP}


def test_profile_summary_key_semantics() -> None:
    server = ProfileSummary(guild_id="g", subject_id=None, text="digest")
    assert server.subject_id is None


def test_discovery_pairs_empty_when_solo() -> None:
    assert discovery_pairs("alice") == ()
    assert discovery_pairs("alice", ("alice",), ("alice",)) == ()


def test_discovery_pairs_all_combinations_first_seen_order() -> None:
    pairs = discovery_pairs("alice", ("bob",), ("carol", "bob"))
    assert pairs == (("alice", "bob"), ("alice", "carol"), ("bob", "carol"))


def test_discovery_pairs_thread_only_enables_related() -> None:
    assert discovery_pairs("alice", (), ("bob",)) == (("alice", "bob"),)
