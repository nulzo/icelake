"""The rest of the public surface: facts, identity, admin, ops, events, recall.

Runnable WITHOUT Discord or an LLM. ``python examples/lifecycle.py``

Graph queries live in ``examples/relationship_queries.py``. Name lookup lives
in ``examples/name_lookup_tool.py``. Citations live in ``examples/citations.py``.
This file is the remaining capability groups on one client: curate facts,
resolve names, govern a guild, watch events, and operate the worker plane.

``Memory`` is the canonical class. ``DiscordMemory`` is the same object.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from icelake import (
    AliasSource,
    CommandAction,
    FactCategory,
    FactCommitted,
    FactSupersededEvent,
    Memory,
    MemoryConfig,
    MessageEvent,
    ObserveStatus,
    RecallQuery,
    Scope,
    discovery_pairs,
)

GUILD = "555"
ALICE = "100000000000000001"
BOB = "200000000000000002"
CAROL = "300000000000000003"


def _event(author_id: str, name: str, content: str, message_id: str) -> MessageEvent:
    return MessageEvent(
        message_id=message_id,
        guild_id=GUILD,
        channel_id="general",
        author_id=author_id,
        content=content,
        created_at=datetime.now(UTC),
        author_display_name=name,
    )


async def demo_identity(memory: Memory) -> None:
    print("=== 1. Identity: backfill, aliases, rename, resolve ===")
    registered = await memory.ops.backfill_aliases(
        GUILD,
        (
            (ALICE, "alice", "alice"),
            (BOB, "bob", "bobby"),
            (CAROL, "carol", "carol"),
        ),
    )
    await memory.identity.register_alias(GUILD, ALICE, "ali", source=AliasSource.DISPLAY_NAME)
    await memory.identity.handle_member_rename(GUILD, BOB, "robert")
    print(f"  backfill registered {registered} aliases")
    print(f"  alice display_name: {await memory.identity.display_name(GUILD, ALICE)}")
    aliases = await memory.identity.aliases_of(GUILD, BOB)
    print(f"  bob aliases: {[record.alias_norm for record in aliases]}")
    for ident in ("ali", "bobby", "robert", "nobody"):
        resolution = await memory.identity.resolve(GUILD, ident)
        if resolution.ambiguous:
            print(f"  {ident!r} is ambiguous, refusing to guess")
        elif resolution.resolved is None:
            print(f"  {ident!r} matched nobody")
        else:
            print(f"  {ident!r} -> {resolution.resolved.user_id}")
    print()


async def demo_facts(memory: Memory) -> None:
    print("=== 2. Facts CRUD (remember, search, update, reinforce, forget, history) ===")
    rust = await memory.facts.remember(
        guild_id=GUILD,
        subject_id=ALICE,
        text="alice has been learning rust for about a year",
        category=FactCategory.INTERESTS,
        actor_id=ALICE,
        subject_username="alice",
    )
    await memory.facts.remember(
        guild_id=GUILD,
        subject_id=BOB,
        text="bob hosts sunday minecraft builds",
        category=FactCategory.INTERESTS,
        actor_id=BOB,
    )
    hits = await memory.facts.search(GUILD, "rust", subject_ids=(ALICE,))
    print(f"  search rust: {[fact.text for fact, _score in hits]}")
    page = await memory.facts.list_for_subject(GUILD, ALICE, include_server=False)
    print(f"  list_for_subject alice: {len(page.items)} fact(s), cursor={page.next_cursor}")
    all_alice = await memory.facts.get_all(GUILD, ALICE)
    print(f"  get_all alice: {len(all_alice)}")

    loaded = await memory.facts.get(GUILD, rust.id)
    stronger = await memory.facts.reinforce(rust.id, guild_id=GUILD)
    updated = await memory.facts.update(
        rust.id,
        guild_id=GUILD,
        text="alice has been learning rust for about a year and likes cargo",
        reason="member correction",
        actor_id=ALICE,
    )
    await memory.facts.forget(updated.id, guild_id=GUILD, reason="user request", actor_id=ALICE)
    history = await memory.facts.history(updated.id, guild_id=GUILD)
    kinds = [entry.kind.value for entry in history]
    print(f"  get id={loaded.id}  strength {loaded.strength} -> {stronger.strength}")
    print(f"  history after update+forget: {kinds}")
    print()


async def demo_events(memory: Memory) -> None:
    print("=== 3. Events (sync handlers, keep them short) ===")
    committed: list[str] = []
    superseded: list[str] = []

    @memory.events.on(FactCommitted)
    def _on_commit(event: FactCommitted) -> None:
        committed.append(event.fact_id)

    memory.events.subscribe(
        FactSupersededEvent,
        lambda event: superseded.append(event.reason or "(none)"),
    )

    fact = await memory.facts.remember(
        guild_id=GUILD,
        subject_id=CAROL,
        text="carol hosts movie night every friday",
        actor_id=CAROL,
    )
    await memory.facts.update(
        fact.id,
        guild_id=GUILD,
        text="carol hosts movie night every friday night",
        reason="refine wording",
    )
    await asyncio.sleep(0)
    print(f"  FactCommitted fired: {len(committed)}")
    print(f"  FactSuperseded reasons: {superseded}")
    print()


async def demo_recall(memory: Memory) -> None:
    print("=== 4. Recall, discovery_pairs, entity_hint ===")
    pairs = discovery_pairs(ALICE, mentioned_ids=(BOB,), thread_participant_ids=(CAROL,))
    print(f"  discovery_pairs(alice, bob, carol) = {pairs}")
    result = await memory.recall(
        RecallQuery(
            guild_id=GUILD,
            text="minecraft",
            subject_ids=(BOB,),
            scope=Scope.SUBJECTS,
            entity_hint="minecraft",
        )
    )
    print(f"  recall minecraft: {[scored.fact.text for scored in result.facts]}")
    print(f"  warnings={list(result.warnings)}  degraded={list(result.degraded_channels)}")
    print()


async def demo_commands(memory: Memory) -> None:
    print("=== 5. classify_command (library classifies, you decide whether to execute) ===")
    command = await memory.classify_command("hey bot remember that I hate pineapple")
    print(
        f"  action={command.action.value}  text={command.target_text!r}  conf={command.confidence}"
    )
    if command.action is CommandAction.REMEMBER:
        print("  bot-side policy would call facts.remember here. 0.85 is the usual floor.")
    chatter = await memory.classify_command("what are we playing tonight")
    print(f"  chatter action={chatter.action.value}")
    print()


async def demo_governance(memory: Memory) -> None:
    print("=== 6. Governance: opt-out, purge preview, export/import ===")
    await memory.admin.set_opt_out(GUILD, CAROL, True)
    print(f"  carol opted out: {await memory.admin.get_opt_out(GUILD, CAROL)}")
    receipt = await memory.observe(_event(CAROL, "carol", "carol just got a puppy", "obs-1"))
    ignored = receipt.status is ObserveStatus.IGNORED
    print(f"  observe after opt-out ignored={ignored} reason={receipt.reason}")

    dry = await memory.admin.purge_user(GUILD, BOB, dry_run=True)
    print(
        f"  purge dry-run bob: facts={dry.facts_removed} aliases={dry.aliases_removed} "
        f"dry_run={dry.dry_run}"
    )
    export = await memory.admin.export_guild(GUILD)
    print(f"  export: {len(export.facts)} facts, {len(export.entities)} entities")

    replica = Memory(MemoryConfig(storage="sqlite://:memory:", llm=None))
    async with replica:
        inserted = await replica.admin.import_guild(export)
        stats = await replica.stats(GUILD)
        print(f"  import into a fresh store: {inserted} facts, active={stats.active_facts}")

    gone = await memory.admin.purge_user(GUILD, BOB, dry_run=False)
    leftover = await memory.facts.get_all(GUILD, BOB)
    print(f"  purge bob for real: removed {gone.facts_removed}, leftover={len(leftover)}")
    print()


async def demo_ops(memory: Memory) -> None:
    print("=== 7. Ops: health, meters, stats ===")
    health = await memory.ops.health()
    snapshot = memory.ops.meter_snapshot()
    stats = await memory.stats(GUILD)
    print(
        f"  healthy={health.healthy}  pending={health.pending_messages}  dead={health.dead_letters}"
    )
    for component in health.components:
        print(f"  {component.component}: {component.status.value} {component.detail}")
    print(f"  meter calls={dict(snapshot.calls)}  (empty: no LLM in this file)")
    print(
        f"  stats active={stats.active_facts} users={stats.user_count} "
        f"entities={stats.entity_count}"
    )
    print("  workers off + ops.run_pending() is the split bot/worker topology.")
    print("  retry_dead_letters re-drives poison jobs after you fix the cause.")


async def main() -> None:
    memory = Memory(MemoryConfig(storage="sqlite://:memory:", llm=None))
    async with memory:
        memory.register_bot_id("42")
        await demo_identity(memory)
        await demo_facts(memory)
        await demo_events(memory)
        await demo_recall(memory)
        await demo_commands(memory)
        await demo_governance(memory)
        await demo_ops(memory)


if __name__ == "__main__":
    asyncio.run(main())
