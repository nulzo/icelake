"""Comprehensive end-to-end suite: the memory system as a bot experiences it.

Drives the REAL library — real OpenRouter LLM, real embeddings, real sqlite —
through ONLY the public DiscordMemory API, exactly the way omni_style_bot.py
calls it. Nothing is stubbed, scripted, shaped, or reached into internals:
every extraction and every reconcile decision is the model's genuine behavior,
and every check asserts an outcome a production bot depends on.

  Suite A (workers off, deterministic — cron-style flush mode):
    passive adds         observe() -> batch -> extract -> commit
    noise gauntlet       acks/emoji/bot authors/empty/dup ids never reach the LLM
    style gauntlet       CAPS, slang, questions, hypotheticals, ephemeral, negation
    reinforcement        exact + paraphrase repeats grow occurrences, not rows
    refinement           "promoted to charge nurse" updates the nurse fact
    conflicts            "i moved to seattle" retires "lives in omaha"
    contradictions       "i'm over red bull" retires "loves red bull"
    cross-user pollution mentions anchor facts to the RIGHT user; no bleed-over
    name guarding        third-party names never bind to the speaker
    manual curation      remember/update/reinforce/forget/history + extract_now
    identity             real-name mining, resolve, ambiguity, alias backfill
    governance           opt-out (meter-verified), dry-run + real purge
    multitenancy         guild B facts/aliases never leak into guild A
    retrieval            channels, strength, time travel, caps, budget trimming
    commands/events/ops  classify_command intent, event bus, health, stats
    lifecycle            idempotent start/close, clean rejection after close
    budgets              daily ceiling degrades to skip-extraction, no overspend
    failure resilience   poison batches dead-letter, event fires, requeue works

  Suite B (workers on — the production path):
    response cache       identical LLM requests replay for zero tokens
    community scope      worker heartbeat extracts server-scope facts (game night)
    age trigger          below-size batches flush via max_age_seconds, no flush()

Requires OPENROUTER_API_KEY. Run:

    .venv/bin/python examples/e2e_simulation.py

Checks come in two tiers: hard checks (system guarantees — gating, dedup,
anchoring, governance, ranking rules) drive the non-zero exit code; expectations
(model-decided outcomes — extraction recall, reinforcement, reconcile judgment)
are reported as WEAK when unmet but never fail the suite. A persistent WEAK is
a prompt/model finding to investigate, not a system bug. Inspect the database
afterwards:

    sqlite3 e2e.db "SELECT text, occurrences, valid_until FROM dm_facts;"
    sqlite3 e2e.db "SELECT alias_norm, user_id, source FROM dm_aliases;"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from icelake import (
    BatchCompleted,
    ChannelName,
    CommandAction,
    DiscordMemory,
    ExtractionFailed,
    FactCommitted,
    FactScope,
    FactSupersededEvent,
    LlmConfig,
    MemoryConfig,
    MessageEvent,
    MeterSnapshot,
    RecallQuery,
    Scope,
)

GUILD = "900000000000000001"
GUILD2 = "900000000000000099"  # isolation probe; only the multitenancy phase uses it
CHANNEL = "900000000000000002"
ALICE = "100000000000000001"  # username/display: "nulzo"
BOB = "100000000000000002"  # username/display: "bobby"
CAROL = "100000000000000003"  # username/display: "carol"
DAVE = "100000000000000004"  # never speaks; cold-start backfill only
BOT = "100000000000000009"  # a bot account; never a subject
PROBE = "100000000000000010"  # deterministic curation probes; never speaks
PEOPLE = ((ALICE, "nulzo"), (BOB, "bobby"), (CAROL, "carol"))

# Same provider URLs as examples/omni_style_bot.py.
DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
LLM_URL_TEMPLATE = "openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model={model}"
EMBEDDINGS_URL = (
    "openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=openai/text-embedding-3-small"
)


class Suite:
    """Hard checks gate the exit code; expectations report model behavior.

    The library's GUARANTEES (gating, dedup, anchoring, governance, curation,
    ranking rules) are hard checks. Outcomes the LLM freely decides (which facts
    it extracts, whether it re-emits to reinforce, how it phrases an update) are
    expectations: reported as WEAK when unmet, but they never fail the suite —
    a persistent WEAK is a model/prompt finding, not a system bug.
    """

    def __init__(self) -> None:
        self.checks = 0
        self.failures = 0
        self.expectations = 0
        self.weak = 0
        self.failed_names: list[str] = []
        self.weak_names: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks += 1
        if ok:
            print(f"  PASS {name}")
        else:
            self.failures += 1
            self.failed_names.append(name)
            print(f"  FAIL {name}  {detail}")

    def expect(self, name: str, ok: bool, detail: str = "") -> None:
        self.expectations += 1
        if ok:
            print(f"  PASS {name}")
        else:
            self.weak += 1
            self.weak_names.append(name)
            print(f"  WEAK {name}  {detail}")

    def report(self) -> dict[str, object]:
        return {
            "checks": self.checks,
            "failures": self.failures,
            "failed": self.failed_names,
            "expectations": self.expectations,
            "weak": self.weak,
            "weak_checks": self.weak_names,
        }


class Simulator:
    """Feeds scripted chat through observe() exactly like a bot's on_message."""

    def __init__(
        self,
        memory: DiscordMemory,
        *,
        id_prefix: str = "m",
        start_at: datetime | None = None,
    ) -> None:
        self.memory = memory
        self._clock = start_at or datetime(2026, 8, 27, 12, 0, 0, tzinfo=UTC)
        self._seq = 0
        self._prefix = id_prefix

    @property
    def now(self) -> datetime:
        return self._clock

    def event(
        self,
        user_id: str,
        username: str,
        content: str,
        *,
        mentions: tuple[str, ...] = (),
        bot: bool = False,
        guild: str = GUILD,
        channel: str = CHANNEL,
    ) -> MessageEvent:
        self._seq += 1
        self._clock += timedelta(seconds=20)
        return MessageEvent(
            message_id=f"{self._prefix}_{self._seq:04d}",
            guild_id=guild,
            channel_id=channel,
            author_id=user_id,
            content=content,
            created_at=self._clock,
            author_username=username,
            author_display_name=username,
            author_is_bot=bot,
            mention_ids=mentions,
        )

    async def say(
        self,
        user_id: str,
        username: str,
        content: str,
        *,
        mentions: tuple[str, ...] = (),
        bot: bool = False,
        guild: str = GUILD,
    ) -> str:
        receipt = await self.memory.observe(
            self.event(user_id, username, content, mentions=mentions, bot=bot, guild=guild)
        )
        detail = receipt.status.value if receipt.reason is None else receipt.reason.value
        print(f"  {username}: {content}  [{detail}]")
        return detail

    async def drain(self) -> None:
        """Flush until no batch reports a summary (one flush = one batch per subject)."""
        while await self.memory.flush():
            pass


async def active_facts(memory: DiscordMemory, user_id: str, guild: str = GUILD) -> list[str]:
    page = await memory.facts.list_for_subject(
        guild, user_id, include_server=False, active_only=True
    )
    return [fact.text for fact in page.items]


async def all_facts(memory: DiscordMemory, user_id: str, guild: str = GUILD):
    page = await memory.facts.list_for_subject(
        guild, user_id, include_server=False, active_only=False
    )
    return list(page.items)


def mentions(facts: list[str], pattern: str) -> list[str]:
    return [text for text in facts if re.search(pattern, text, re.IGNORECASE)]


async def print_state(memory: DiscordMemory) -> None:
    for user_id, name in PEOPLE:
        facts = await all_facts(memory, user_id)
        aliases = await memory.identity.aliases_of(GUILD, user_id)
        if not facts and not aliases:
            continue
        print(f"  {name}:")
        for fact in facts:
            state = "active  " if fact.is_active else "INACTIVE"
            print(f"    [{state}] x{fact.occurrences} {fact.text}")
        if aliases:
            names = ", ".join(f"{a.alias_norm} ({a.source.value})" for a in aliases)
            print(f"    aliases: {names}")


async def check_no_duplicates(suite: Suite, memory: DiscordMemory) -> None:
    for user_id, name in PEOPLE:
        facts = await all_facts(memory, user_id)
        norms = [fact.text_normalized for fact in facts if fact.is_active]
        suite.check(
            f"{name}: no duplicate active facts",
            len(norms) == len(set(norms)),
            detail=str(norms),
        )


# --------------------------------------------------------------------------- #
# Suite A phases: deterministic, workers off (cron-style flush mode).         #
# --------------------------------------------------------------------------- #


async def phase_adds(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 1: passive adds + real-name mining ==")
    await sim.say(ALICE, "nulzo", "hey guys my name is nolan gregory")
    await sim.say(ALICE, "nulzo", "but everyone call me nolan")
    await sim.say(ALICE, "nulzo", "i love writing go")
    await sim.say(ALICE, "nulzo", "i live in omaha")
    await sim.say(ALICE, "nulzo", "i love red bull")
    await sim.say(ALICE, "nulzo", "the yellow cans are elite")
    await sim.say(ALICE, "nulzo", "i work as a nurse at city hospital, been there 3 years now")
    await sim.say(BOB, "bobby", "hey, call me bobby")
    await sim.say(BOB, "bobby", "i'm a designer")
    await sim.say(BOB, "bobby", "i love coffee")
    await sim.drain()
    await print_state(memory)

    facts = await active_facts(memory, ALICE)
    suite.expect("alice: a fact about her name", bool(mentions(facts, r"nolan")), str(facts))
    suite.expect("alice: a fact about go", bool(mentions(facts, r"\bgo\b|golang")), str(facts))
    suite.expect("alice: a fact about omaha", bool(mentions(facts, r"omaha")), str(facts))
    suite.expect("alice: a fact about red bull", bool(mentions(facts, r"red bull")), str(facts))
    suite.expect("alice: a fact about nursing", bool(mentions(facts, r"nurse")), str(facts))
    bob_facts = await active_facts(memory, BOB)
    suite.expect("bob: a fact about design", bool(mentions(bob_facts, r"design")), str(bob_facts))
    suite.expect("bob: a fact about coffee", bool(mentions(bob_facts, r"coffee")), str(bob_facts))

    aliases = {a.alias_norm: a.source.value for a in await memory.identity.aliases_of(GUILD, ALICE)}
    suite.check(
        "alice alias nolan gregory (real_name)",
        aliases.get("nolan gregory") == "real_name",
        str(aliases),
    )
    suite.check("alice alias nolan (real_name)", aliases.get("nolan") == "real_name", str(aliases))
    suite.check("alice alias nulzo (display_name)", "nulzo" in aliases, str(aliases))
    await check_no_duplicates(suite, memory)


async def phase_noise(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 2: noise gauntlet — never reaches the LLM ==")
    before = dict(memory.ops.meter_snapshot().calls)
    before_facts = (await memory.stats(GUILD)).total_facts

    await sim.say(ALICE, "nulzo", "lol")
    await sim.say(ALICE, "nulzo", "haha")
    await sim.say(ALICE, "nulzo", "nice one")
    await sim.say(ALICE, "nulzo", "🔥🔥🔥")
    detail = await sim.say(ALICE, "nulzo", "   ")
    suite.check("empty content is ignored", detail == "empty_content", detail)
    detail = await sim.say(BOT, "announce-bot", "i am a bot announcing stuff", bot=True)
    suite.check("bot author is ignored", detail == "bot_author", detail)

    duplicate = sim.event(ALICE, "nulzo", "lol")  # noise content: keeps the batch gated
    first = await memory.observe(duplicate)
    second = await memory.observe(duplicate)
    suite.check(
        "same message id twice is a duplicate",
        first.status.value == "accepted"
        and second.reason is not None
        and second.reason.value == "duplicate",
        f"{first.status} / {second.status} {second.reason}",
    )
    await sim.drain()

    after = dict(memory.ops.meter_snapshot().calls)
    suite.check("noise gauntlet made zero LLM calls", before == after, f"{before} -> {after}")
    suite.check(
        "noise gauntlet added zero facts",
        (await memory.stats(GUILD)).total_facts == before_facts,
    )


async def phase_styles(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 3: style gauntlet — durable signal vs non-facts ==")
    await sim.say(ALICE, "nulzo", "I JUST GOT A PUPPY NAMED BISCUIT I'M SO HAPPY")
    await sim.say(ALICE, "nulzo", "ngl biscuit is my whole world now 🐶💕")
    await sim.say(ALICE, "nulzo", "do you guys think i should get another dog?")
    await sim.say(ALICE, "nulzo", "i wish i could live in japan someday honestly")
    await sim.say(ALICE, "nulzo", "ugh i'm so tired today, pulled an all nighter")
    await sim.say(ALICE, "nulzo", "i don't drink coffee at all, never have")
    await sim.drain()
    await print_state(memory)

    facts = await active_facts(memory, ALICE)
    suite.expect(
        "caps/slang: a fact about the puppy", bool(mentions(facts, r"biscuit|puppy")), str(facts)
    )
    biscuit = [
        f
        for f in await all_facts(memory, ALICE)
        if f.is_active and re.search(r"biscuit|puppy", f.text, re.IGNORECASE)
    ]
    suite.expect(
        "puppy restatement merged into ONE active fact (update/reinforce, not add)",
        len(biscuit) == 1,
        str([(f.text, f.occurrences) for f in biscuit]),
    )
    suite.expect(
        "question did not become a fact about a second dog",
        not mentions(facts, r"another dog|second dog|two dogs"),
        str(facts),
    )
    japan = mentions(facts, r"japan")
    suite.expect(
        "hypothetical kept wish-qualified, not stated as residence",
        not [
            t
            for t in japan
            if not re.search(r"wish|hope|dream|someday|wants? to", t, re.IGNORECASE)
        ],
        str(japan),
    )
    suite.expect(
        "ephemeral state did not become a durable fact",
        not mentions(facts, r"\btired\b|all.?nighter"),
        str(facts),
    )
    coffee = mentions(facts, r"coffee")
    suite.expect(
        "negation: no false coffee affinity for alice",
        not [
            t
            for t in coffee
            if not re.search(r"don't|does not|never|no longer|hate|dislike", t, re.IGNORECASE)
        ],
        str(coffee),
    )
    await check_no_duplicates(suite, memory)


async def phase_duplicates(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 4: repetition reinforces instead of duplicating ==")
    before = await all_facts(memory, ALICE)
    await sim.say(ALICE, "nulzo", "i love writing go")
    await sim.say(ALICE, "nulzo", "i love writing go")
    await sim.say(ALICE, "nulzo", "i really love writing go")
    await sim.say(ALICE, "nulzo", "go is the best")
    await sim.say(ALICE, "nulzo", "i've been writing go for like 5 years now, it's my main one")
    await sim.say(ALICE, "nulzo", "i write all my side projects in go these days")
    await sim.drain()
    await print_state(memory)
    after = await all_facts(memory, ALICE)
    go_facts = [f for f in after if re.search(r"\bgo\b|golang", f.text, re.IGNORECASE)]
    suite.expect(
        "go knowledge reinforced (occurrences grew)",
        any(f.occurrences >= 2 for f in go_facts),
        str([(f.text, f.occurrences) for f in go_facts]),
    )
    suite.expect(
        "repeats did not pile up new rows",
        len(after) <= len(before) + 1,
        f"{len(before)} -> {len(after)}",
    )
    await check_no_duplicates(suite, memory)


async def phase_update(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 5: a promotion refines the job fact (UPDATE path) ==")
    await sim.say(ALICE, "nulzo", "i just got promoted to charge nurse at the hospital")
    await sim.say(ALICE, "nulzo", "been working toward this for years honestly")
    await sim.say(ALICE, "nulzo", "the pay bump is nice too")
    await sim.drain()
    await print_state(memory)
    facts = await all_facts(memory, ALICE)
    nurse = [f for f in facts if re.search(r"nurse", f.text, re.IGNORECASE)]
    active_nurse = [f for f in nurse if f.is_active]
    suite.expect(
        "an active fact reflects the charge nurse promotion",
        any(re.search(r"charge nurse", f.text, re.IGNORECASE) for f in active_nurse),
        str([(f.text, f.is_active) for f in nurse]),
    )
    suite.expect(
        "one active nursing fact, not a pile of versions",
        len(active_nurse) == 1,
        str([(f.text, f.is_active) for f in nurse]),
    )
    await check_no_duplicates(suite, memory)


async def phase_conflict(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 6: a move retires the old location ==")
    await sim.say(ALICE, "nulzo", "i moved to seattle")
    await sim.say(ALICE, "nulzo", "the move went well")
    await sim.say(ALICE, "nulzo", "seattle is rainy")
    await sim.drain()
    await print_state(memory)
    facts = await active_facts(memory, ALICE)
    suite.expect(
        "an active fact places alice in seattle",
        bool(mentions(facts, r"seattle")),
        str(facts),
    )
    suite.expect(
        "no active fact still claims omaha",
        not mentions(facts, r"lives in omaha|living in omaha"),
        str(facts),
    )
    stale = [
        f
        for f in await all_facts(memory, ALICE)
        if re.search(r"omaha", f.text, re.IGNORECASE) and not f.is_active
    ]
    if stale:
        history = await memory.facts.history(stale[0].id, guild_id=GUILD)
        suite.expect(
            "retired omaha fact has an audit trail",
            any(entry.kind in {"superseded", "invalidated"} for entry in history),
            str([entry.kind for entry in history]),
        )
    await check_no_duplicates(suite, memory)


async def phase_contradiction(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 7: a contradiction retires the old preference ==")
    await sim.say(ALICE, "nulzo", "i'm over red bull, i don't drink it anymore honestly")
    await sim.say(ALICE, "nulzo", "quit drinking it last month and i feel so much better")
    await sim.say(ALICE, "nulzo", "my sleep schedule has improved a lot since then")
    await sim.drain()
    await print_state(memory)
    facts = await active_facts(memory, ALICE)
    suite.expect(
        "no active fact still claims alice loves red bull",
        not [
            t
            for t in mentions(facts, r"red bull")
            if re.search(r"love|like|enjoy", t, re.IGNORECASE)
            and not re.search(r"no longer|quit|stopped|over", t, re.IGNORECASE)
        ],
        str(facts),
    )
    stale = [
        f
        for f in await all_facts(memory, ALICE)
        if re.search(r"red bull", f.text, re.IGNORECASE) and not f.is_active
    ]
    suite.expect("the old red bull fact was retired, not deleted", bool(stale))
    await check_no_duplicates(suite, memory)


async def phase_pollution(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 8: cross-user pollution — facts anchor to the right person ==")
    await sim.say(
        ALICE,
        "nulzo",
        "bobby is obsessed with mechanical keyboards, he has like twelve",
        mentions=(BOB,),
    )
    await sim.say(BOB, "bobby", "nolan taught me go basics last weekend", mentions=(ALICE,))
    # Rapid interleaved conversation in one batch window via the bulk entry point.
    receipts = await memory.observe_many(
        (
            sim.event(ALICE, "nulzo", "my favorite color is purple"),
            sim.event(BOB, "bobby", "i'm learning to play the drums"),
            sim.event(CAROL, "carol", "i just adopted a cat named whiskers"),
            sim.event(ALICE, "nulzo", "purple everything — even my keyboard is purple"),
            sim.event(BOB, "bobby", "drum practice every evening this week"),
            sim.event(CAROL, "carol", "whiskers is already destroying my couch lol"),
        )
    )
    suite.check(
        "observe_many accepted the interleaved batch",
        all(r.status.value == "accepted" for r in receipts),
        str([(r.status.value, r.reason) for r in receipts]),
    )
    await sim.drain()
    await print_state(memory)

    alice_facts = await active_facts(memory, ALICE)
    bob_facts = await active_facts(memory, BOB)
    carol_facts = await active_facts(memory, CAROL)
    suite.expect(
        "keyboards anchored to bob (the mention), not alice (the speaker)",
        bool(mentions(bob_facts, r"mechanical keyboard"))
        and not mentions(alice_facts, r"mechanical keyboard|twelve"),
        f"bob={bob_facts} alice={alice_facts}",
    )
    suite.expect("alice: purple present", bool(mentions(alice_facts, r"purple")), str(alice_facts))
    suite.check(
        "alice: no bleed from bob's drums or carol's cat",
        not mentions(alice_facts, r"drum") and not mentions(alice_facts, r"whiskers|\bcat\b"),
        str(alice_facts),
    )
    suite.expect("bob: drums present", bool(mentions(bob_facts, r"drum")), str(bob_facts))
    suite.check(
        "bob: no bleed from alice's purple or carol's cat",
        not mentions(bob_facts, r"purple") and not mentions(bob_facts, r"whiskers|\bcat\b"),
        str(bob_facts),
    )
    suite.expect(
        "carol: cat present", bool(mentions(carol_facts, r"whiskers|\bcat\b")), str(carol_facts)
    )
    suite.check(
        "carol: no bleed from alice's purple or bob's drums",
        not mentions(carol_facts, r"purple") and not mentions(carol_facts, r"drum"),
        str(carol_facts),
    )

    pair = await memory.recall(
        RecallQuery(guild_id=GUILD, text="keyboards", pair_ids=((ALICE, BOB),))
    )
    suite.expect(
        "pair recall surfaces a fact linking alice and bob",
        bool(pair.facts),
        str([f.fact.text for f in pair.facts]),
    )
    await check_no_duplicates(suite, memory)


async def phase_name_guards(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 9: third-party names never bind to the speaker ==")
    bob_aliases_before = {a.alias_norm for a in await memory.identity.aliases_of(GUILD, BOB)}
    await sim.say(BOB, "bobby", "nolan's last name is gregory, right?")
    await sim.say(CAROL, "carol", "my friend steve is visiting this weekend")
    await sim.say(CAROL, "carol", "he hasn't been here in years")
    await sim.drain()

    bob_aliases = {a.alias_norm for a in await memory.identity.aliases_of(GUILD, BOB)}
    suite.check(
        "bob stating nolan's surname bound nothing to bob",
        bob_aliases == bob_aliases_before and "gregory" not in bob_aliases,
        str(bob_aliases),
    )
    carol_aliases = {a.alias_norm for a in await memory.identity.aliases_of(GUILD, CAROL)}
    suite.check(
        "carol's friend steve did not become carol's alias",
        "steve" not in carol_aliases,
        str(carol_aliases),
    )
    alice_aliases = {a.alias_norm for a in await memory.identity.aliases_of(GUILD, ALICE)}
    suite.check(
        "alice's real-name aliases are intact",
        {"nolan", "nolan gregory"} <= alice_aliases,
        str(alice_aliases),
    )


async def phase_manual_curation(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 10: manual curation (the /memory edit entry points) ==")
    fact = await memory.facts.remember(
        guild_id=GUILD,
        subject_id=BOB,
        text="bobby is a night owl",
        actor_id=BOB,
    )
    suite.check("remember: fact committed", fact.is_active)

    updated = await memory.facts.update(
        fact.id, guild_id=GUILD, text="bobby is a night owl on weekdays", actor_id=BOB
    )
    suite.check("update: text refined", updated.text == "bobby is a night owl on weekdays")

    reinforced = await memory.facts.reinforce(fact.id, guild_id=GUILD)
    suite.check("reinforce: occurrences incremented", reinforced.occurrences == 2)

    await memory.facts.forget(fact.id, guild_id=GUILD, reason="bob asked", actor_id=BOB)
    gone = await memory.facts.get(GUILD, fact.id)
    suite.check("forget: fact soft-removed", not gone.is_active)

    history = await memory.facts.history(fact.id, guild_id=GUILD)
    suite.check("history: full audit trail", len(history) >= 2, str([e.kind for e in history]))

    # extract_now: the synchronous bypass for messages that must land immediately.
    receipt = await memory.extract_now(
        sim.event(ALICE, "nulzo", "i'm allergic to shellfish, found out last year")
    )
    suite.check("extract_now accepted", receipt.status.value == "accepted", receipt.status)
    suite.expect(
        "extract_now committed without a flush",
        bool(mentions(await active_facts(memory, ALICE), r"shellfish")),
        str(await active_facts(memory, ALICE)),
    )

    # Pagination over the public listing API — driven by curation probes so the
    # check tests the mechanism, not the model's extraction volume.
    for i in range(3):
        await memory.facts.remember(
            guild_id=GUILD, subject_id=PROBE, text=f"probe user pagination fact {i}"
        )
    page1 = await memory.facts.list_for_subject(GUILD, PROBE, include_server=False, limit=2)
    suite.check("pagination: first page respects limit", len(page1.items) <= 2)
    suite.check("pagination: enough facts to paginate", page1.next_cursor is not None, str(page1))
    if page1.next_cursor is not None:
        page2 = await memory.facts.list_for_subject(
            GUILD, PROBE, include_server=False, limit=2, cursor=page1.next_cursor
        )
        overlap = {f.id for f in page1.items} & {f.id for f in page2.items}
        suite.check("pagination: pages do not overlap", not overlap, str(overlap))
    await check_no_duplicates(suite, memory)


async def phase_identity(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 11: identity resolution, ambiguity, cold-start backfill ==")
    await sim.say(CAROL, "carol", "hi i'm carol")
    await sim.say(CAROL, "carol", "my name is nolan")
    await sim.say(CAROL, "carol", "great to be here")
    await sim.drain()
    await print_state(memory)

    nolan = await memory.identity.resolve(GUILD, "nolan")
    suite.check("resolve('nolan') is ambiguous (two users)", nolan.ambiguous)
    full = await memory.identity.resolve(GUILD, "nolan gregory")
    suite.check(
        "resolve('nolan gregory') -> alice",
        full.resolved is not None and full.resolved.user_id == ALICE and not full.ambiguous,
    )
    bobby = await memory.identity.resolve(GUILD, "bobby")
    suite.check(
        "resolve('bobby') -> bob",
        bobby.resolved is not None and bobby.resolved.user_id == BOB,
    )

    registered = await memory.ops.backfill_aliases(GUILD, [(DAVE, "dave", "Dave")])
    suite.check("backfill registered dave's directory aliases", registered >= 1, str(registered))
    dave = await memory.identity.resolve(GUILD, "dave")
    suite.check(
        "cold-start: resolve('dave') works before dave ever spoke",
        dave.resolved is not None and dave.resolved.user_id == DAVE,
    )


async def phase_governance(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 12: governance — opt-out and purge ==")
    await memory.admin.set_opt_out(GUILD, CAROL, True)
    before = dict(memory.ops.meter_snapshot().calls)
    detail = await sim.say(CAROL, "carol", "i love hiking")
    suite.check("opted-out observe is rejected", detail == "opted_out", detail)
    await sim.drain()
    after = dict(memory.ops.meter_snapshot().calls)
    suite.check("opted-out message made zero LLM calls", before == after, f"{before} -> {after}")

    dry = await memory.admin.purge_user(GUILD, CAROL, dry_run=True)
    still_there = await memory.identity.aliases_of(GUILD, CAROL)
    suite.check(
        "dry-run purge reports but deletes nothing",
        dry.aliases_removed >= 1 and bool(still_there),
        f"{dry} aliases still {still_there}",
    )

    report = await memory.admin.purge_user(GUILD, CAROL, dry_run=False)
    suite.check("purge removed carol's aliases", report.aliases_removed >= 1, str(report))
    carol_aliases = await memory.identity.aliases_of(GUILD, CAROL)
    suite.check("carol has no aliases left", not carol_aliases, str(carol_aliases))
    carol_facts = await all_facts(memory, CAROL)
    suite.check("carol has no facts left", not carol_facts, str([f.text for f in carol_facts]))

    nolan = await memory.identity.resolve(GUILD, "nolan")
    suite.check(
        "after purge, resolve('nolan') -> alice uniquely",
        nolan.resolved is not None and nolan.resolved.user_id == ALICE and not nolan.ambiguous,
    )


async def phase_multitenancy(suite: Suite, memory: DiscordMemory, sim: Simulator) -> None:
    print("\n== Phase 13: multi-guild isolation ==")
    await sim.say(ALICE, "nulzo", "hey, call me shadowfax", guild=GUILD2)
    await sim.say(ALICE, "nulzo", "i'm really into rocket league", guild=GUILD2)
    await sim.say(ALICE, "nulzo", "been playing rocket league for years", guild=GUILD2)
    await sim.drain()

    g2_facts = await active_facts(memory, ALICE, guild=GUILD2)
    g1_facts = await active_facts(memory, ALICE)
    suite.expect(
        "guild2: rocket league fact exists",
        bool(mentions(g2_facts, r"rocket league")),
        str(g2_facts),
    )
    suite.check(
        "guild1: no rocket league leak", not mentions(g1_facts, r"rocket league"), str(g1_facts)
    )
    shadowfax_g2 = await memory.identity.resolve(GUILD2, "shadowfax")
    suite.check(
        "guild2: shadowfax resolves to alice",
        shadowfax_g2.resolved is not None and shadowfax_g2.resolved.user_id == ALICE,
    )
    shadowfax_g1 = await memory.identity.resolve(GUILD, "shadowfax")
    suite.check("guild1: shadowfax alias does not leak", shadowfax_g1.resolved is None)
    leak = await memory.recall(
        RecallQuery(guild_id=GUILD, text="rocket league", subject_ids=(ALICE,))
    )
    suite.check(
        "guild1 recall only returns guild1 facts",
        all(f.fact.guild_id == GUILD for f in leak.facts)
        and not mentions([f.fact.text for f in leak.facts], r"rocket league"),
        str([f.fact.text for f in leak.facts]),
    )


async def phase_retrieval(suite: Suite, memory: DiscordMemory, pre_move_wall: datetime) -> None:
    print("\n== Phase 14: retrieval deep-dive ==")
    lexical = await memory.recall(RecallQuery(guild_id=GUILD, text="biscuit", subject_ids=(ALICE,)))
    suite.expect(
        "lexical: exact-word query surfaces the puppy fact",
        bool(mentions([f.fact.text for f in lexical.facts], r"biscuit|puppy")),
        str([f.fact.text for f in lexical.facts]),
    )
    if lexical.facts:
        suite.expect(
            "lexical: keyword channel participated",
            any(ChannelName.KEYWORD in f.matched_channels for f in lexical.facts),
            str([f.matched_channels for f in lexical.facts]),
        )

    semantic = await memory.recall(RecallQuery(guild_id=GUILD, text="caffeine", subject_ids=(BOB,)))
    suite.expect(
        "semantic: synonym query (caffeine) surfaces the coffee fact",
        bool(mentions([f.fact.text for f in semantic.facts], r"coffee")),
        str([f.fact.text for f in semantic.facts]),
    )

    past = await memory.recall(
        RecallQuery(
            guild_id=GUILD,
            text="where does nolan live",
            subject_ids=(ALICE,),
            as_of=pre_move_wall,
        )
    )
    past_texts = [f.fact.text for f in past.facts]
    suite.expect(
        "time travel: as_of before the move still shows omaha",
        bool(mentions(past_texts, r"omaha")),
        str(past_texts),
    )
    now = await memory.recall(
        RecallQuery(guild_id=GUILD, text="where does nolan live", subject_ids=(ALICE,))
    )
    now_texts = [f.fact.text for f in now.facts]
    suite.expect(
        "present: seattle, and no live omaha claim",
        bool(mentions(now_texts, r"seattle"))
        and not mentions(now_texts, r"lives in omaha|living in omaha"),
        str(now_texts),
    )

    capped = await memory.recall(
        RecallQuery(
            guild_id=GUILD,
            text="hobbies and interests",
            subject_ids=(ALICE, BOB),
            max_per_subject=1,
            top_k=8,
        )
    )
    per_subject: dict[str, int] = {}
    for scored in capped.facts:
        subject = scored.fact.subject_id or "__server__"
        per_subject[subject] = per_subject.get(subject, 0) + 1
    suite.check(
        "max_per_subject=1 caps each subject at one fact",
        all(count <= 1 for count in per_subject.values()),
        str(per_subject),
    )

    strict = await memory.recall(
        RecallQuery(guild_id=GUILD, text="programming", subject_ids=(ALICE,), min_score=0.95)
    )
    suite.check(
        "min_score=0.95 filters weak matches",
        all(f.score >= 0.95 for f in strict.facts),
        str([f.score for f in strict.facts]),
    )

    # Strength signal: manual reinforcement must outrank unreinforced peers.
    go_fact = next(
        (
            f
            for f in await all_facts(memory, ALICE)
            if f.is_active and re.search(r"\bgo\b|golang", f.text, re.IGNORECASE)
        ),
        None,
    )
    if go_fact is not None:
        for _ in range(3):
            await memory.facts.reinforce(go_fact.id, guild_id=GUILD)
        ranked = await memory.recall(
            RecallQuery(guild_id=GUILD, text="what does nolan enjoy", subject_ids=(ALICE,), top_k=8)
        )
        strengths = {f.fact.id: f.components.strength for f in ranked.facts}
        others = [s for fact_id, s in strengths.items() if fact_id != go_fact.id]
        suite.expect(
            "reinforced fact carries the top strength signal",
            go_fact.id in strengths and bool(others) and strengths[go_fact.id] >= max(others),
            str(strengths),
        )

    # Channel restriction: a vector-only recall must not report keyword hits.
    vector_only = await memory.recall(
        RecallQuery(
            guild_id=GUILD,
            text="biscuit",
            subject_ids=(ALICE,),
            channels=(ChannelName.VECTOR,),
        )
    )
    suite.expect(
        "channel restriction still surfaces the puppy fact",
        bool(vector_only.facts),
        "empty result",
    )
    suite.check(
        "channel restriction excludes the keyword channel",
        all(ChannelName.KEYWORD not in f.matched_channels for f in vector_only.facts),
        str([f.matched_channels for f in vector_only.facts]),
    )

    excluded = await memory.recall(
        RecallQuery(guild_id=GUILD, text="go", subject_ids=(ALICE,), exclude_ids=(ALICE,))
    )
    suite.check(
        "exclude_ids removes the subject from results",
        not [f for f in excluded.facts if f.fact.subject_id == ALICE],
        str([f.fact.text for f in excluded.facts]),
    )

    await memory.regenerate_summaries(GUILD, (BOB,))
    ctx = await memory.prompt_context(
        guild_id=GUILD,
        asker_id=BOB,
        text="what does nolan like?",
        mentioned_ids=(ALICE,),
        token_budget_tokens=800,
    )
    suite.expect(
        "prompt_context injects alice's facts for a turn about her",
        bool(re.search(r"\bgo\b|golang", ctx.injection_block, re.IGNORECASE)),
        detail=ctx.injection_block[:200],
    )
    suite.expect(
        "prompt_context binds citations for injected facts",
        bool(ctx.citations),
        str(len(ctx.citations)),
    )
    suite.check(
        "apply_citations strips unknown mem refs",
        ctx.apply_citations("see [mem:999]") == "see ",
        repr(ctx.apply_citations("see [mem:999]")),
    )
    suite.expect(
        "profile summary generated for the asker",
        ctx.asker_summary is not None and bool(ctx.asker_summary.strip()),
        str(ctx.asker_summary),
    )
    ctx_tight = await memory.prompt_context(
        guild_id=GUILD,
        asker_id=BOB,
        text="what does nolan like?",
        mentioned_ids=(ALICE,),
        token_budget_tokens=60,
    )
    suite.expect(
        "token budget trims the injection block",
        bool(ctx.injection_block) and len(ctx_tight.injection_block) < len(ctx.injection_block),
        f"{len(ctx.injection_block)} -> {len(ctx_tight.injection_block)} chars",
    )


async def phase_ops(
    suite: Suite,
    memory: DiscordMemory,
    events: dict[str, int],
) -> None:
    print("\n== Phase 15: commands, event bus, ops ==")
    remember = await memory.classify_command("remember that i like tea")
    suite.expect(
        "classify: 'remember that i like tea' -> remember",
        remember.action is CommandAction.REMEMBER and "tea" in remember.target_text.lower(),
        str(remember),
    )
    query = await memory.classify_command("what do you remember about me?")
    suite.expect(
        "classify: 'what do you remember about me?' -> query",
        query.action is CommandAction.QUERY,
        str(query),
    )
    before = sum(memory.ops.meter_snapshot().calls.values())
    none = await memory.classify_command("nice weather today")
    after = sum(memory.ops.meter_snapshot().calls.values())
    suite.check(
        "classify: plain chat -> none without an LLM call (regex gate)",
        none.action is CommandAction.NONE and before == after,
        f"{none} calls {before} -> {after}",
    )

    # Event bus: deterministic probes via the public curation API. These test
    # the publish mechanism, not extraction volume, so model quality can't
    # turn a library guarantee into a flaky check.
    fired: dict[str, list[object]] = {"committed": [], "superseded": []}
    memory.events.subscribe(FactCommitted, lambda e: fired["committed"].append(e))
    memory.events.subscribe(FactSupersededEvent, lambda e: fired["superseded"].append(e))

    probe = await memory.facts.remember(
        guild_id=GUILD, subject_id=PROBE, text="probe user keeps a sourdough starter"
    )
    await memory.facts.reinforce(probe.id, guild_id=GUILD)
    await memory.facts.update(
        probe.id, guild_id=GUILD, text="probe user keeps two sourdough starters", reason="probe"
    )
    await memory.facts.forget(probe.id, guild_id=GUILD, reason="probe complete")
    await asyncio.sleep(0.2)  # event handlers dispatch via loop.call_soon

    committed = fired["committed"]
    suite.check(
        "event bus: curation publishes FactCommitted (add then reinforce)",
        len(committed) == 2
        and all(getattr(e, "fact_id", None) == probe.id for e in committed)
        and [getattr(e, "was_reinforcement", None) for e in committed] == [False, True],
        str(committed),
    )
    superseded = fired["superseded"]
    transitions = [
        (getattr(e, "old_fact_id", None), getattr(e, "new_fact_id", "missing")) for e in superseded
    ]
    suite.check(
        "event bus: curation publishes FactSupersededEvent (refine then retire)",
        transitions == [(probe.id, probe.id), (probe.id, None)],
        str(superseded),
    )
    suite.check("event bus: BatchCompleted fired", events["batches"] >= 10, str(events))

    health = await memory.ops.health()
    suite.check(
        "health report is healthy with component coverage",
        health.healthy and bool(health.components),
        str(health),
    )
    stats = await memory.stats(GUILD)
    suite.check(
        "stats: active <= total, users tracked (carol purged -> 2)",
        stats.active_facts <= stats.total_facts and stats.user_count >= 2,
        str(stats),
    )


async def phase_lifecycle(suite: Suite, llm_url: str) -> None:
    print("\n== Phase 16: lifecycle robustness ==")
    memory = DiscordMemory(
        MemoryConfig(storage="sqlite:///:memory:", llm=llm_url, workers={"enabled": False})
    )
    await memory.start()
    await memory.start()  # idempotent — a crash here fails the suite loudly
    await memory.close()
    await memory.close()
    receipt = await memory.observe(
        MessageEvent(
            message_id="life_1",
            guild_id=GUILD,
            channel_id=CHANNEL,
            author_id=ALICE,
            content="my name is late",
            created_at=datetime.now(UTC),
            author_username="nulzo",
            author_display_name="nulzo",
        )
    )
    suite.check(
        "observe after close rejects cleanly (never raises)",
        receipt.status.value == "rejected"
        and receipt.reason is not None
        and receipt.reason.value == "storage_unavailable",
        str(receipt),
    )


async def phase_budget(suite: Suite, llm_url: str) -> None:
    print("\n== Phase 17: budget ladder — degradation instead of overspend ==")
    memory = DiscordMemory(
        MemoryConfig(
            storage="sqlite:///:memory:",
            llm=llm_url,
            budgets={"guild_daily_prompt_tokens": 1},
            workers={"enabled": False},
        )
    )
    await memory.start()
    try:
        sim = Simulator(memory, id_prefix="c")
        await sim.say(ALICE, "nulzo", "my name is budget tester")
        await sim.drain()  # first extraction spends well over the 1-token ceiling
        calls_after_first = sum(memory.ops.meter_snapshot().calls.values())
        suite.expect("first batch extracted before the budget bound", calls_after_first >= 1)

        await sim.say(ALICE, "nulzo", "i work as a budget analyst")
        await sim.drain()
        snap = memory.ops.meter_snapshot()
        suite.check(
            "extraction skipped once the daily budget bound",
            sum(snap.calls.values()) == calls_after_first,
            f"{calls_after_first} -> {sum(snap.calls.values())}",
        )
        suite.check(
            "no facts committed after the budget bound",
            not mentions(await active_facts(memory, ALICE), r"analyst"),
            str(await active_facts(memory, ALICE)),
        )
    finally:
        await memory.close()


async def phase_failure(suite: Suite) -> None:
    print("\n== Phase 18: failure resilience — poison batches dead-letter ==")
    memory = DiscordMemory(
        MemoryConfig(
            storage="sqlite:///:memory:",
            llm="openai://sk-invalid-key@openrouter.ai/api/v1?model=google/gemini-3.7-flash",
            workers={"enabled": False},
        )
    )
    await memory.start()
    failures = {"n": 0}

    def bump(_e: object) -> None:
        failures["n"] += 1

    memory.events.subscribe(ExtractionFailed, bump)
    try:
        sim = Simulator(memory, id_prefix="d")
        await sim.say(ALICE, "nulzo", "my name is resilient tester")
        await sim.drain()
        await asyncio.sleep(0.2)  # event handlers dispatch via loop.call_soon
        stats = await memory.stats(GUILD)
        suite.check(
            "failed batch dead-lettered instead of crashing", stats.dead_letters >= 1, str(stats)
        )
        suite.check("ExtractionFailed event fired", failures["n"] >= 1, str(failures))
        requeued = await memory.ops.retry_dead_letters(GUILD)
        suite.check("dead letters requeue for retry", requeued >= 1, str(requeued))
    finally:
        await memory.close(drain=False)


async def phase_cache(suite: Suite, db_path: Path, llm_url: str) -> None:
    print("\n== Phase 19: LLM response cache — identical requests replay free ==")
    llm_config = LlmConfig.from_url(llm_url).model_copy(update={"cache_responses": True})
    memory = DiscordMemory(
        MemoryConfig(storage=f"sqlite:///{db_path}", llm=llm_config, workers={"enabled": False})
    )
    await memory.start()
    try:
        first = await memory.classify_command("remember that i like tea")  # miss or prior-run hit
        mid = memory.ops.meter_snapshot()
        second = await memory.classify_command("remember that i like tea")  # must be a hit
        after = memory.ops.meter_snapshot()
        suite.check(
            "cached replay spends zero tokens",
            sum(after.prompt_tokens.values()) == sum(mid.prompt_tokens.values()),
            f"{sum(mid.prompt_tokens.values())} -> {sum(after.prompt_tokens.values())}",
        )
        suite.check(
            "cached replay returns the same classification",
            first.action == second.action,
            f"{first.action} vs {second.action}",
        )
    finally:
        await memory.close()


# --------------------------------------------------------------------------- #
# Suite B: workers enabled — the production path, including the community     #
# window pass that only the worker heartbeat triggers.                        #
# --------------------------------------------------------------------------- #


async def quiesce(memory: DiscordMemory) -> None:
    """Wait for the worker loop to reach quiescence: nothing pending, nothing
    in-flight (claimed-but-uncommitted), and a stable fact count. All three are
    publicly observable via stats(); in_flight covers the extraction window that
    pending_messages alone cannot see."""
    last_total = -1
    for _ in range(120):
        stats = await memory.stats(GUILD)
        idle = stats.pending_messages == 0 and stats.in_flight_messages == 0
        if idle and stats.total_facts == last_total:
            return
        last_total = stats.total_facts
        await asyncio.sleep(0.5 if not idle else 1.0)


async def phase_community(suite: Suite, db_path: Path, llm_url: str) -> MeterSnapshot:
    print("\n== Phase 20: community scope via the worker loop (production mode) ==")
    memory = DiscordMemory(
        MemoryConfig(
            storage=f"sqlite:///{db_path}",
            llm=llm_url,
            embeddings=EMBEDDINGS_URL,
            batching={"batch_size_messages": 4, "max_age_seconds": 10},
            workers={"enabled": True, "poll_interval_seconds": 0.2, "heartbeat_seconds": 5},
        )
    )
    await memory.start()
    try:
        # Clock starts after suite A so the server-window watermark (ordered by
        # message time) can never filter these messages out as "already seen".
        sim = Simulator(memory, id_prefix="b", start_at=datetime(2026, 8, 27, 13, 0, 0, tzinfo=UTC))

        async def game_night_wave() -> None:
            for user_id, name in PEOPLE[:2]:
                await sim.say(user_id, name, "game night this friday at 8pm, don't forget")
                await sim.say(user_id, name, "i'll bring snacks for game night")

        # Wave 1 lets the first heartbeat absorb suite A's backlog and set the
        # watermark; later waves are then pure game-night windows.
        await game_night_wave()
        await asyncio.sleep(6)
        await quiesce(memory)
        await game_night_wave()
        await quiesce(memory)

        for _ in range(4):
            found = await memory.facts.search(GUILD, "game night", server_only=True)
            if found:
                break
            await game_night_wave()  # top up so the next heartbeat has fresh volume
            await asyncio.sleep(5.5)
            await quiesce(memory)
        else:
            found = ()
        suite.expect(
            "worker heartbeat extracted a server-scope game night fact",
            bool(found),
            "no server facts matched 'game night'",
        )

        server_recall = await memory.recall(
            RecallQuery(guild_id=GUILD, text="when is game night", scope=Scope.SERVER)
        )
        suite.expect(
            "server-scope recall surfaces the community fact",
            bool(mentions([f.fact.text for f in server_recall.facts], r"game night|friday")),
            str([f.fact.text for f in server_recall.facts]),
        )
        stats = await memory.stats(GUILD)
        suite.expect(
            "stats track the server scope",
            stats.by_scope.get(FactScope.SERVER, 0) >= 1,
            str(stats.by_scope),
        )

        # Age trigger: a below-size batch must flush via max_age_seconds with no
        # manual flush — quiesce() only returns once the worker has drained it.
        await sim.say(ALICE, "nulzo", "my new hobby is playing the piano")
        await sim.say(ALICE, "nulzo", "i started learning piano last month")
        await sim.say(ALICE, "nulzo", "practicing piano scales every morning now")
        await quiesce(memory)
        suite.expect(
            "age trigger flushed a below-size batch (piano fact)",
            bool(mentions(await active_facts(memory, ALICE), r"piano")),
            str(await active_facts(memory, ALICE)),
        )
    finally:
        meter = memory.ops.meter_snapshot()
        await memory.close(drain=True)
    return meter


async def run(memory: DiscordMemory, llm_url: str) -> Suite:
    suite = Suite()
    sim = Simulator(memory)
    events = {"batches": 0}

    def bump(name: str):
        return lambda _e: events.__setitem__(name, events[name] + 1)

    memory.events.subscribe(BatchCompleted, bump("batches"))

    await phase_adds(suite, memory, sim)
    await phase_noise(suite, memory, sim)
    await phase_styles(suite, memory, sim)
    await phase_duplicates(suite, memory, sim)
    await phase_update(suite, memory, sim)
    pre_move_wall = datetime.now(UTC)  # validity windows use commit wall-time
    await phase_conflict(suite, memory, sim)
    await phase_contradiction(suite, memory, sim)
    await phase_pollution(suite, memory, sim)
    await phase_name_guards(suite, memory, sim)
    await phase_manual_curation(suite, memory, sim)
    await phase_identity(suite, memory, sim)
    await phase_governance(suite, memory, sim)
    await phase_multitenancy(suite, memory, sim)
    await phase_retrieval(suite, memory, pre_move_wall)
    await phase_ops(suite, memory, events)
    await phase_lifecycle(suite, llm_url)
    await phase_budget(suite, llm_url)
    await phase_failure(suite)

    stats = await memory.stats(GUILD)
    print(
        f"\n== {suite.checks - suite.failures}/{suite.checks} hard checks passed; "
        f"{suite.expectations - suite.weak}/{suite.expectations} model expectations met; "
        f"{stats.total_facts} total fact rows =="
    )
    print_cost_report(memory)
    return suite


def print_cost_report(memory: DiscordMemory) -> None:
    """Per-purpose usage from the library's meter (chat calls; embeddings are
    not metered — the Embedder port returns no usage)."""
    meter = memory.ops.meter_snapshot()
    print("\n== cost report ==")
    print(f"  {'purpose':<14} {'calls':>5} {'prompt':>8} {'completion':>10} {'est. cost':>10}")
    total_cost = 0.0
    for purpose in sorted(meter.calls):
        cost = meter.estimated_cost_usd.get(purpose, 0.0)
        total_cost += cost
        print(
            f"  {purpose:<14} {meter.calls[purpose]:>5} "
            f"{meter.prompt_tokens.get(purpose, 0):>8} "
            f"{meter.completion_tokens.get(purpose, 0):>10} "
            f"${cost:>9.6f}"
        )
    print(
        f"  {'TOTAL':<14} {sum(meter.calls.values()):>5} "
        f"{sum(meter.prompt_tokens.values()):>8} "
        f"{sum(meter.completion_tokens.values()):>10} "
        f"${total_cost:>9.6f}"
    )


def _meter_totals(*snapshots: MeterSnapshot) -> dict[str, object]:
    """Merge meter snapshots across clients (suite A + suite B) into totals."""
    calls: dict[str, int] = {}
    prompt: dict[str, int] = {}
    completion: dict[str, int] = {}
    cost: dict[str, float] = {}
    for snap in snapshots:
        for purpose in snap.calls:
            calls[purpose] = calls.get(purpose, 0) + snap.calls[purpose]
            prompt[purpose] = prompt.get(purpose, 0) + snap.prompt_tokens.get(purpose, 0)
            completion[purpose] = completion.get(purpose, 0) + snap.completion_tokens.get(
                purpose, 0
            )
            cost[purpose] = cost.get(purpose, 0.0) + snap.estimated_cost_usd.get(purpose, 0.0)
    return {
        "calls": sum(calls.values()),
        "prompt_tokens": sum(prompt.values()),
        "completion_tokens": sum(completion.values()),
        "cost_usd": round(sum(cost.values()), 6),
        "by_purpose": {
            purpose: {
                "calls": calls[purpose],
                "prompt_tokens": prompt[purpose],
                "completion_tokens": completion[purpose],
                "cost_usd": round(cost[purpose], 6),
            }
            for purpose in sorted(calls)
        },
    }


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines); real environment wins."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="e2e.db", help="sqlite file to write (default e2e.db)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter chat model to exercise (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--reasoning",
        choices=("none", "minimal", "low", "medium", "high"),
        default=None,
        help="OpenRouter reasoning.effort; 'none' disables thinking when the model allows it",
    )
    parser.add_argument(
        "--temperature",
        default=None,
        help="sampling temperature, or 'none' to omit the parameter entirely "
        "(required by reasoning-model endpoints that reject it, e.g. gpt-5.6-luna)",
    )
    parser.add_argument(
        "--structured-outputs",
        choices=("strict", "json_object"),
        default=None,
        help="strict json_schema (default) or json_object if the endpoint cannot enforce a schema",
    )
    parser.add_argument(
        "--report",
        default=None,
        metavar="PATH",
        help="also write the machine-readable run report (JSON) here — used by bench_models.py",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logs from the pipeline")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        for noisy in ("httpx", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    _load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("set OPENROUTER_API_KEY (or add it to .env); this suite runs the real LLM")

    llm_url = LLM_URL_TEMPLATE.format(model=args.model)
    if args.reasoning:
        llm_url += f"&reasoning={args.reasoning}"
    if args.temperature is not None:
        llm_url += f"&temperature={args.temperature}"
    if args.structured_outputs:
        llm_url += f"&structured_outputs={args.structured_outputs}"
    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()  # deterministic starting state
        print(f"fresh db: {db_path}")

    # Mirror omni_style_bot.py's configuration. Workers stay OFF for suite A so
    # drain() is deterministic; suite B runs its own client with workers ON.
    memory = DiscordMemory(
        MemoryConfig(
            storage=f"sqlite:///{db_path}",
            llm=llm_url,
            embeddings=EMBEDDINGS_URL,
            batching={"batch_size_messages": 12, "max_age_seconds": 90},
            extraction={"auto_consolidate_after_adds": 6},
            retrieval={"default_token_budget": 800},
            budgets={"guild_daily_prompt_tokens": 150_000},
            workers={"enabled": False},
        )
    )

    async def _run() -> int:
        started = time.monotonic()
        await memory.start()
        try:
            suite_a = await run(memory, llm_url)
            stats = await memory.stats(GUILD)
            meter_a = memory.ops.meter_snapshot()
        finally:
            await memory.close(drain=True)
        suite_b = Suite()
        await phase_cache(suite_b, db_path, llm_url)
        meter_b = await phase_community(suite_b, db_path, llm_url)
        met = suite_b.expectations - suite_b.weak
        print(
            f"\n== suite B: {suite_b.checks - suite_b.failures}/{suite_b.checks} hard checks; "
            f"{met}/{suite_b.expectations} model expectations met =="
        )
        if args.report:
            report = {
                "model": args.model,
                "duration_seconds": round(time.monotonic() - started, 1),
                "facts": stats.total_facts,
                "suites": {"A": suite_a.report(), "B": suite_b.report()},
                "llm": _meter_totals(meter_a, meter_b),
            }
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
            print(f"report: {args.report}")
        return 1 if suite_a.failures or suite_b.failures else 0

    raise SystemExit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
