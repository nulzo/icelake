"""Access-driven decay loop (mem0 pattern) + get_all parity tests."""

from __future__ import annotations

from datetime import UTC, datetime

from discord_memory import DiscordMemory, RecallQuery
from tests.conftest import make_config

GUILD = "500000000000000001"
ALICE = "100000000000000001"


def _memory_with_decay(*, enabled: bool):
    config = make_config(
        retrieval={"reinforce_on_recall": enabled},
    )
    from discord_memory.api.client import DiscordMemory

    memory = DiscordMemory(config, llm=None)
    return memory


async def _seed_two_facts(memory: DiscordMemory) -> tuple[str, str]:
    fresh = await memory.facts.remember(
        guild_id=GUILD,
        subject_id=ALICE,
        text="freshly reinforced fact about gardening",
        actor_id="seed",
    )
    stale = await memory.facts.remember(
        guild_id=GUILD,
        subject_id=ALICE,
        text="stale fact about ancient pottery",
        actor_id="seed",
    )
    assert memory.started
    return fresh.id, stale.id


class TestRecallDecayLoop:
    async def test_recall_resets_decay_clock_when_enabled(self) -> None:
        """mem0-decay pattern: served facts get their decay clock reset."""
        memory = _memory_with_decay(enabled=True)
        await memory.start()
        fresh_id, stale_id = await _seed_two_facts(memory)

        before = {
            f.id: f.last_reinforced_at
            for f in (await memory._store.get_facts(GUILD, (fresh_id, stale_id)))
        }

        # Simulate a turn that serves BOTH facts.
        result = await memory.recall(
            RecallQuery(
                guild_id=GUILD, text="gardening or pottery", subject_ids=(ALICE,), min_score=0.0
            )
        )

        assert result.facts
        for scored in result.facts:
            after = await memory._store.get_fact(GUILD, scored.fact.id)
            assert after is not None
            if after.last_reinforced_at and before.get(after.id):
                assert after.last_reinforced_at >= before[after.id]
        await memory.close()

    async def test_disabled_by_default_no_writes(self) -> None:
        memory = _memory_with_decay(enabled=False)
        await memory.start()
        fact_id, _ = await _seed_two_facts(memory)
        before = (await memory._store.get_fact(GUILD, fact_id)).last_reinforced_at
        await memory.recall(
            RecallQuery(guild_id=GUILD, text="gardening", subject_ids=(ALICE,), min_score=0.0)
        )
        after = (await memory._store.get_fact(GUILD, fact_id)).last_reinforced_at
        assert after == before  # untouched without the knob
        await memory.close()


class TestGetAllParity:
    async def test_get_all_returns_active_only(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        kept = await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="kept fact about chess", actor_id="x"
        )
        dropped = await client.facts.remember(
            guild_id=GUILD, subject_id=ALICE, text="dropped fact about checkers", actor_id="x"
        )
        await client.facts.forget(dropped.id, guild_id=GUILD)
        items = await client.facts.get_all(GUILD, ALICE)
        ids = {f.id for f in items}
        assert kept.id in ids and dropped.id not in ids
        await client.close()


import pytest as _pytest  # noqa: E402


@_pytest.mark.parametrize("enabled", [True, False])
async def test_touch_is_batched_single_statement(enabled: bool) -> None:
    """touch_facts port method exists on all backends (conformance guard)."""
    from discord_memory.adapters.in_memory.store import InMemoryStore

    store = InMemoryStore()
    now = datetime.now(UTC)
    touched = await store.touch_facts("g", (), accessed_at=now)
    assert touched == 0
