from __future__ import annotations

import pytest

from icelake.adapters.in_memory.store import InMemoryStore
from icelake.identity.aliases import (
    alias_slug,
    is_valid_alias,
    normalize_alias,
    strongest_alias,
    weight_for_source,
)
from icelake.identity.guards import BotGuard, ConsentPolicy, SubjectGate
from icelake.identity.resolver import IdentityResolver
from icelake.models.identity import AliasSource


def test_normalize() -> None:
    assert normalize_alias("  Alice   Wong ") == "alice wong"


def test_snowflake_aliases_rejected() -> None:
    assert not is_valid_alias("123456789012345678")
    assert is_valid_alias("1234")  # short digit strings allowed


def test_slug_generation() -> None:
    assert alias_slug("Rust Programming!") == "rust-programming"
    assert alias_slug("!!!") == "unknown"


def test_strongest_alias_prefers_rank_then_weight() -> None:
    from icelake.models.identity import AliasRecord

    def record(alias: str, source: AliasSource, weight: float) -> AliasRecord:
        return AliasRecord(
            guild_id="g",
            alias_norm=alias,
            user_id="u",
            source=source,
            weight=weight,
        )

    assert strongest_alias(()) is None
    records = (
        record("ally", AliasSource.MENTION, 0.9),
        record("alice", AliasSource.DISCORD_USERNAME, 0.6),
    )
    # Higher source rank wins even at lower weight (rank is the authority).
    assert strongest_alias(records) == "alice"
    same_rank = (
        record("ally", AliasSource.MENTION, 0.9),
        record("al", AliasSource.MENTION, 0.6),
    )
    assert strongest_alias(same_rank) == "ally"


def test_weight_for_source_ordering() -> None:
    assert weight_for_source(AliasSource.DISCORD_USERNAME) >= weight_for_source(
        AliasSource.DISPLAY_NAME,
    )


@pytest.fixture()
async def populated_store():
    store = InMemoryStore()
    await store.upsert_alias("g", "alice", "u-alice", AliasSource.DISCORD_USERNAME, 1.0)
    await store.upsert_alias("g", "alice", "u-other", AliasSource.DISPLAY_NAME, 0.7)
    await store.upsert_alias("g", "alicia", "u-alicia", AliasSource.SUBJECT_USERNAME, 0.95)
    return store


async def test_exact_resolution_prefers_strong_source(populated_store: InMemoryStore) -> None:
    resolver = IdentityResolver(populated_store)
    resolution = await resolver.resolve("g", "Alice")
    # discord_username (1.0) beats display_name (0.7): strong source wins, not ambiguous
    assert not resolution.ambiguous
    assert resolution.resolved is not None
    assert resolution.resolved.user_id == "u-alice"


async def test_rank_beats_weight_when_sources_conflict() -> None:
    """A higher-weight low-rank alias must not steal resolution from a lower-weight
    high-rank alias. Store order is weight DESC; resolution must follow source rank."""
    store = InMemoryStore()
    # display_name has the higher weight but the lower rank.
    await store.upsert_alias("g", "klim", "u-display", AliasSource.DISPLAY_NAME, 0.9)
    await store.upsert_alias("g", "klim", "u-username", AliasSource.DISCORD_USERNAME, 0.7)
    resolution = await IdentityResolver(store).resolve("g", "klim")
    assert not resolution.ambiguous
    assert resolution.resolved is not None
    assert resolution.resolved.user_id == "u-username"


async def test_equal_weight_candidates_are_ambiguous() -> None:
    store = InMemoryStore()
    await store.upsert_alias("g", "alex", "u-alex-1", AliasSource.DISPLAY_NAME, 0.9)
    await store.upsert_alias("g", "alex", "u-alex-2", AliasSource.DISPLAY_NAME, 0.88)
    resolution = await IdentityResolver(store).resolve("g", "Alex")
    assert resolution.ambiguous
    assert resolution.resolved is None
    assert {c.user_id for c in resolution.candidates} == {"u-alex-1", "u-alex-2"}


async def test_unambiguous_resolution(populated_store: InMemoryStore) -> None:
    resolver = IdentityResolver(populated_store)
    resolution = await resolver.resolve("g", "alicia")
    assert not resolution.ambiguous
    assert resolution.resolved is not None
    assert resolution.resolved.user_id == "u-alicia"


async def test_prefix_fallback(populated_store: InMemoryStore) -> None:
    resolver = IdentityResolver(populated_store)
    # 'alici' uniquely prefixes only 'alicia'
    resolution = await resolver.resolve("g", "alici")
    assert not resolution.ambiguous
    assert resolution.resolved is not None
    assert resolution.resolved.user_id == "u-alicia"


async def test_ambiguous_prefix_never_guesses(populated_store: InMemoryStore) -> None:
    resolver = IdentityResolver(populated_store)
    resolution = await resolver.resolve("g", "alic")  # matches alice AND alicia
    assert resolution.resolved is None


async def test_snowflake_passthrough(populated_store: InMemoryStore) -> None:
    resolver = IdentityResolver(populated_store)
    snowflake = "123456789012345678"
    resolution = await resolver.resolve("g", snowflake)
    assert resolution.resolved is not None
    assert resolution.resolved.user_id == snowflake
    assert resolution.basis == "alias:mention"


async def test_unknown_identifier_resolves_none(populated_store: InMemoryStore) -> None:
    resolver = IdentityResolver(populated_store)
    resolution = await resolver.resolve("g", "zzzzzz")
    assert resolution.resolved is None
    assert not resolution.ambiguous


class TestBotGuard:
    def test_registered_and_observed_bots(self) -> None:
        guard = BotGuard()
        guard.register("bot-1")
        guard.note_author("bot-2", is_bot=True)
        guard.note_author("human", is_bot=False)
        assert guard.is_bot("bot-1")
        assert guard.is_bot("bot-2")
        assert not guard.is_bot("human")
        assert not guard.is_bot(None)

    def test_exclude_drops_bots_preserving_order(self) -> None:
        guard = BotGuard()
        guard.register("bot-1")
        assert guard.exclude(("alice", "bot-1", "bob", "alice")) == ("alice", "bob")


async def test_subject_gate_blocks_opted_out() -> None:
    store = InMemoryStore()
    await store.set_opt_out("g", "quiet-user", True)
    gate = SubjectGate(BotGuard(), ConsentPolicy(store))
    assert not await gate.allows("g", "quiet-user")
    assert await gate.allows("g", "regular-user")
    assert await gate.allows("g", None)  # server scope always allowed
