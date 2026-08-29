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
    Citation,
    PromptContext,
    channels,
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


def _prompt_context() -> PromptContext:
    return PromptContext(
        injection_block="[MEMORY CONTEXT]",
        citations=(
            Citation(ref="mem:1", fact_id="fct_a", url="https://discord.com/channels/g/c/m1"),
            Citation(ref="mem:2", fact_id="fct_b", url=""),
        ),
    )


def test_apply_citations_resolves_and_strips() -> None:
    ctx = _prompt_context()
    out = ctx.apply_citations("they game [mem:1] a lot [mem:2] and [mem:9] gone")
    assert "[[mem:1]](https://discord.com/channels/g/c/m1)" in out
    assert "a lot [mem:2]" in out  # citation without url keeps its plain tag
    assert "mem:9" not in out  # unknown refs stripped


def test_apply_citations_unknown_refs_removed() -> None:
    ctx = _prompt_context()
    out = ctx.apply_citations("cite [mem:42] end")
    assert "mem:42" not in out
    assert out == "cite  end"


def test_apply_citations_leaves_non_mem_brackets() -> None:
    ctx = _prompt_context()
    text = "array[0] and [not a citation]"
    assert ctx.apply_citations(text) == text
