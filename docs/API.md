# discord-memory: Consumer API Contract

Normative specification of the public interface. Implementation plan lives in
[`PLAN.md`](./PLAN.md); where the two differ, this document wins for anything
consumer-visible. Every exported name below is public API under SemVer; everything
not listed here is internal.

Design priorities carried over: **Accuracy → Cost → Performance**, and one more for
the surface itself: **simple things simple, hard things possible**.

---

## Part 1 — API Design Principles

1. **Four-method hot path.** A consumer needs `start`, `observe`, `prompt_context`,
   `close`. Everything else is progressive disclosure.
2. **Async everywhere, never blocking.** All I/O methods are coroutines. CPU-bound work
   is off-loop by contract; consumers' event loops are sacred (discord.py shares it).
3. **Typed at every boundary.** Inputs and outputs are frozen Pydantic models; fixed
   vocabularies are enums; keyword-only arguments everywhere; mypy strict clean.
4. **Receipts, not surprises.** Fire-and-forget paths report outcomes as values
   (`ObserveReceipt`); exceptions are reserved for caller mistakes (bad config, bad
   arguments) and unrecoverable infrastructure states.
5. **Transparency over magic.** Any name-based resolution returns *who was matched and
   why*; any degradation is visible in the result object; every injected fact carries
   its citations.
6. **Composable through ports, extended through config.** Customization = implementing a
   Protocol and passing it in. No inheritance hierarchies, no monkeypatching, no plugin
   registries to learn.
7. **Namespaced facade.** One class, few hot verbs at top level, capability groups
   underneath (`memory.facts.*`, `memory.graph.*`) — the discord.py `client.guilds`
   pattern. Flat enough to guess, grouped enough to grow.

---

## Part 2 — Import Map (the entire public surface)

```python
from discord_memory import (
    # lifecycle + facade
    DiscordMemory,
    MemoryConfig,
    # inputs
    MessageEvent,
    RecallQuery, ChannelName, CHANNELS_DEFAULT, CHANNELS_DISCOVERY, channels,
    ManualFact, FactUpdate,
    # results
    ObserveReceipt, RecallResult, ScoredFact, PromptContext,
    Citation, TokenUsage, HealthReport, Page,
    # identity
    Resolution, ResolvedCandidate,
    # graph
    RelationEdge, StanceSummary, NeighborInfo, Polarity, NodeType, EdgeKind,
    # ops/governance
    FactRecord, FactHistoryEntry, GuildStats, MemoryExport,
    # events
    BatchCompleted, FactCommitted, FactSupersededEvent,
    ExtractionFailed, BudgetWarning, ComponentDegraded,
    # errors
    DiscordMemoryError, ConfigError, StorageUnavailableError,
    BudgetExceededError, IdentityAmbiguousError, FactNotFoundError,
    SubjectNotAllowedError,
)

# integrations extra (`pip install discord-memory[discord]`)
from discord_memory.integrations import setup_discord_memory
```

That is the complete root namespace. If it isn't importable from here, it isn't
public — internals evolve freely between releases.

---

## Part 3 — Client Lifecycle

```python
class DiscordMemory:
    def __init__(self, config: MemoryConfig, **overrides: PortOverrides) -> None: ...
    async def start(self) -> None: ...
    async def close(self, *, drain: bool = True, timeout_seconds: float = 30.0) -> None: ...
    async def __aenter__(self) -> DiscordMemory: ...   # == start()
    async def __aexit__(...) -> None: ...              # == close()
```

- `__init__` builds adapters from config (cheap, no I/O). `PortOverrides` lets any
  port be replaced by keyword: `DiscordMemory(cfg, llm=MyLLM(), clock=FakeClock())`.
  This is *the* extension and testing seam — one mechanism, uniformly applied.
- `start()` opens pools, ensures store schema (migrating if needed), launches worker
  tasks. Idempotent; safe after fork (workers spawn lazily).
- `close(drain=True)` stops accepting jobs, finishes leased work within
  `timeout_seconds`, flushes meters, releases leases. `drain=False` abandons in-flight
  work to lease-expiry reclaim (crash semantics).
- Multi-process deployments run `DiscordMemory` in each process; leases make workers
  cooperative. A process may run with `config.workers.enabled = False` to be
  read/write-only (API process) while a dedicated worker process extracts.

### Capability groups (attributes on the client)

| Attribute | Concern |
|---|---|
| `memory.observe_*` (top level) | ingestion |
| `memory.recall`, `memory.prompt_context` (top level) | retrieval |
| `memory.facts` | CRUD, history, search |
| `memory.identity` | name ↔ ID resolution |
| `memory.graph` | relations, stances, discovery |
| `memory.admin` | consent, purge, export, stats |
| `memory.ops` | worker control, health, budgets |
| `memory.events` | hook subscription |

---

## Part 4 — Configuration

```python
config = MemoryConfig(
    storage="sqlite:///data/memory.db",       # required
    llm="openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=google/gemini-2.0-flash",
    embeddings="local",                        # or "openai://KEY@endpoint/model"
)
memory = DiscordMemory(config)
```

`MemoryConfig` is a validated, frozen, nested Pydantic model. Full groups (defaults in
parentheses; every duration suffixed `_seconds/_minutes/_days`, every budget
`_tokens`):

| Group | Key knobs |
|---|---|
| `storage` | backend parsed from URL; `pool_size (10)`; `schema_auto_migrate (True)` |
| `llm` | endpoint URL(s); `extraction_model`; `consolidation_model (None→same)`; `temperature (0.0)`; `request_timeout_seconds (30)`; `max_retries (2)` |
| `embeddings` | provider; `dimensions`; `batch_size (32)`; `cache_enabled (True)` |
| `batching` | `batch_size_messages (10)`; `max_age_seconds (300)`; `lease_seconds (120)`; `server_scope_window (100)` |
| `extraction` | `min_confidence (0.55)`; `max_candidates_per_batch (12)`; `reconcile_collision_threshold (0.85)`; `noise_gate (True)` |
| `lifecycle` | tier retention days `{short:7, mid:45, long:180}`; `strength_forget_threshold`; caps `{per_user:300, server:500}` |
| `retrieval` | `channels (ChannelSet.DEFAULT)`; `rrf_k (60)`; weights dict; `default_token_budget (600)`; `max_per_subject (4)`; `hop_depth (2)`; `fan_out_per_node (24)` |
| `budgets` | `guild_daily_tokens (None=off)`; `guild_monthly_tokens`; `degradation_ladder` order |
| `privacy` | `default_opt_out (False)`; `retention_days (365)`; `store_raw_messages (True)` |
| `workers` | `enabled (True)`; `count (2)`; `poll_interval_seconds (1.0)`; `heartbeat_seconds (20)` |
| `observability` | `meter ("log")`; `log_level`; `slow_query_ms (250)` |

Unknown keys raise `ConfigError` (typo-protection by construction). Every knob maps to
a documented tradeoff in PLAN.md Parts 4–9.

---

## Part 5 — Ingestion (write path, hot)

```python
@frozen_model
class MessageEvent:
    message_id: str                      # Discord snowflake
    guild_id: str
    channel_id: str
    author_id: str
    content: str
    created_at: datetime                 # tz-aware UTC required; naive → TypeError
    author_username: str = ""
    author_display_name: str = ""
    author_is_bot: bool = False
    mention_ids: tuple[str, ...] = ()
    reply_to_message_id: str | None = None
    thread_parent_id: str | None = None
    edited: bool = False
    metadata: Mapping[str, str] = {}     # bounded (≤16 keys, ≤256 chars each)
```

```python
receipt = await memory.observe(event)            # p99 < 5 ms: validate + persist + enqueue
receipts = await memory.observe_many(events)     # bulk: history backfill, replay
await memory.flush(guild_id=...)                 # force pending batches now (optional guild filter)
```

```python
@frozen_model
class ObserveReceipt:
    message_id: str
    status: ObserveStatus        # ACCEPTED | IGNORED | REJECTED
    reason: IgnoreReason | None  # BOT_AUTHOR | OPTED_OUT | EMPTY_CONTENT | DUPLICATE
    error: RejectReason | None   # QUEUE_OVER_CAPACITY | STORAGE_UNAVAILABLE
```

Contract:
- `observe` **never raises** operational errors — infra problems come back as
  `REJECTED` receipts so a message listener can never crash a bot. Caller mistakes
  (malformed event, naive datetime) raise `SchemaValidationError` immediately because
  they indicate broken integration code, not runtime conditions.
- `IGNORED` reasons are policy outcomes (bot authors, opted-out users, empties);
  `DUPLICATE` guards re-delivery by `(guild_id, message_id)` idempotency key.
- `observe_many` is transaction-per-chunk with per-item receipts — designed for
  backfill imports where historical messages arrive with old timestamps; they enter
  extraction queues normally (bitemporal `observed_at` preserved).
- Raw-message retention honors `privacy.store_raw_messages`; disabling stores hashes
  only, trading replayability for footprint.

---

## Part 6 — Retrieval (read path)

### 6.1 The convenience everyone uses first

```python
ctx = await memory.prompt_context(
    guild_id=gid,
    asker_id=user.id,                    # who is talking to the bot
    text=message.content,                # their message (query + entity hints)
    mentioned_ids=(m.id for m in message.mentions),
    thread_participant_ids=(),           # optional, tightens roster
    token_budget_tokens=600,
)

system_prompt += ctx.injection_block      # labeled, budgeted, cite-tagged block
# after generation, resolve [mem:N] tags the model echoed:
final_text = ctx.apply_citations(reply_text)   # → markdown jump links
```

`PromptContext` (frozen):

| Field | Type | Notes |
|---|---|---|
| `injection_block` | `str` | ready to concatenate; includes CURRENT_ASKER / REFERENCED_USER / SERVER section labels |
| `facts` | `tuple[ScoredFact, ...]` | what was injected, with scores |
| `citations` | `Mapping[str, Citation]` | `"mem:1"` → jump link + snippet + subject |
| `asker_summary` | `str \| None` | profile-summary paragraph when enabled |
| `resolution` | `tuple[Resolution, ...]` | every name→ID decision made, with confidence |
| `usage` | `TokenUsage` | prompt tokens consumed vs budget |
| `warnings` | `tuple[RecallWarning, ...]` | e.g. `IDENTITY_AMBIGUOUS`, `BUDGET_TRIMMED`, `DEGRADED_CHANNEL` |

This method composes: intent-lite heuristics (no LLM by default), subject selection,
`RecallQuery`, injection building, citation pool finalization. Power users drop to 6.2.

### 6.2 Explicit query

```python
result = await memory.recall(RecallQuery(
    guild_id=gid,
    text="who called whom a hacker last week?",
    subject_ids=(alice_id,),             # hardened IDs; empty ⇒ auto-scope
    pair_ids=None,                       # relationship mode (Q3): resolved (a, b)
    entity_hint="movies",                # Q5 aggregation seed
    scope=Scope.SUBJECTS,                # SUBJECTS | GUILD | SERVER
    exclude_ids=(bot.user.id,),
    top_k=8,
    max_per_subject=4,
    min_score=0.35,
    token_budget_tokens=600,
    channels=CHANNELS_DEFAULT,           # or channels(ChannelName.VECTOR, ChannelName.KEYWORD)
))
result.facts        # tuple[ScoredFact, ...]: fact + score + subject + citations + valid window
result.degraded     # tuple[str, ...]: channels that failed, if any
result.usage        # TokenUsage
```

Five canonical shapes from PLAN.md §5.4 map directly: Q1 `subject_ids`, Q2 two
subjects + intersect channel, Q3 `pair_ids`, Q4 `channels=ChannelSet.DISCOVERY`
(adds graph-hop), Q5 `entity_hint`.

### 6.3 Scoring transparency

`ScoredFact.score` ∈ [0,1] from the single calibrated formula (PLAN.md §5.3), and each
fact exposes `score_components` (semantic/lexical/entity/strength) — debuggable ranking,
never opaque.

---

## Part 7 — Facts: CRUD, audit, explicit memory

The programmatic equivalent of ChatGPT's "remember this", for bots that expose
commands or natural-language memory requests:

```python
fact = await memory.facts.remember(
    guild_id=gid,
    subject_id=user.id,
    text="prefers mechanical keyboards",
    category=Category.PREFERENCES,       # enum, closed set
    confidence=1.0,                      # manual facts default high
    actor_id=admin.id,                   # who commanded it (attribution: MANUAL)
)
# manual facts land in CORE tier: never expire, exempt from pruning — same as bot's
# proven `/memory remember` behavior.

updated = await memory.facts.update(
    fact.id, text="…", reason="user correction", actor_id=admin.id,
)                                        # supersedes; history preserved
await memory.facts.forget(fact.id, reason="user request", actor_id=admin.id)
                                         # soft-invalidate; reversible via history
await memory.facts.reinforce(fact.id)    # explicit strength bump
hist = await memory.facts.history(fact.id)   # audit chain: creation, reinforcements,
                                             # supersessions, invalidation — full lineage
page = await memory.facts.list_for_subject(gid, user.id, limit=50, cursor=None)
page = await memory.facts.search(gid, text="keyboards", scope=Scope.GUILD)
```

Contracts:
- All mutations are attributed (`actor_id`) and append-only in effect: `update` and
  `forget` create supersession/invalidation records; nothing vanishes until
  `admin.purge`.
- Dedup applies to manual facts too (exact/semantic match ⇒ `REINFORCE` existing +
  return it, flagged `was_deduplicated: True` in `FactRecord`) — prevents manual spam
  duplicates.
- `subject_id` must pass bot-guard and consent checks → else `SubjectNotAllowedError`.

### Natural-language memory commands (opt-in helper)

```python
cmd = await memory.classify_command("hey bot, forget I said I like pineapple")
# → UserMemoryCommand(action=FORGET, target_text="I like pineapple", confidence=0.91)
#   action ∈ REMEMBER | FORGET | UPDATE | QUERY | NONE  (NONE below confidence floor)
```

A thin LLM call (uses extraction model) so consumers can build ChatGPT-style "just say
it" flows without writing prompts. Returns a typed command; **executing it is the
consumer's choice** — the library never mutates from chat text unless asked.

---

## Part 8 — Identity API

```python
res = await memory.identity.resolve(
    guild_id=gid,
    identifier="klim",            # snowflake | @mention | username | display name | saved name
)
# Resolution:
#   resolved: ResolvedCandidate | None   # (user_id, matched_alias, source_rank, weight, confidence)
#   candidates: tuple[ResolvedCandidate, ...]
#   ambiguous: bool
#   basis: str                           # "snowflake" | "alias:discord_username" | ...

await memory.identity.register_alias(gid, user_id, "dr. krill",
                                     source=AliasSource.DISPLAY_NAME)
await memory.identity.handle_member_rename(gid, user_id, old_handle, new_handle)
members = await memory.identity.aliases_of(gid, user_id)     # audit what links them
```

Rules encoded in signatures:
- Snowflake-shaped input resolves to itself (validated shape, no DB hit).
- Ambiguity never guesses: `resolve` returns `ambiguous=True` + candidates;
  `prompt_context` records the ambiguity in `warnings` and scopes the fact out of
  subject attribution.
- Rename handling keeps old aliases valid-but-decaying (PLAN.md B12 fix) and is wired
  automatically by the discord.py integration's `on_member_update`.

---

## Part 9 — Graph API (relations, stances, discovery)

All methods resolve flexible identifiers exactly like `identity.resolve` and attach the
`Resolution` to the response — transparent, auditable matching:

```python
edges = await memory.graph.between(gid, "alice_id", "bob")      # Q3: RelationEdge*
                                                                # verb, polarity, weight,
                                                                # evidence fact ids, validity

stances = await memory.graph.entity_stances(gid, "movies")      # Q5 aggregation
# StanceSummary(entity_slug, positive=(user_id, weight)..., negative=(...),
#               mixed_verbs={...}, total_evidence=17)

neighbors = await memory.graph.neighbors(gid, user_id,          # Q4 hop expansion
                                         depth=2, limit_per_hop=24)
# NeighborInfo(node, relation_path, strength) — path kept for honest phrasing

similar = await memory.graph.similar_users(gid, user_id, limit=10)
# Jaccard-over-entity-adjacency, capped merge-join (§4.7 scale mechanics)
```

Every returned edge carries `evidence_ids` — consumers (and our own injection builder)
can only ever show backed-up claims. Hub nodes served top-k-by-weight, never full
adjacency dumps.

---

## Part 10 — Governance & Admin

```python
await memory.admin.set_opt_out(gid, user.id, opted_out=True)   # enforced everywhere, instantly
report = await memory.admin.purge_user(gid, user.id, dry_run=True)
# PurgeReport: facts, summaries, aliases, vector rows, link/edge rows to be removed
await memory.admin.purge_user(gid, user.id, dry_run=False)     # PLAN §9.3 semantics
export = await memory.admin.export_guild(gid)                  # MemoryExport: JSONL-friendly,
                                                               # facts + provenance + entities
stats = await memory.stats(gid)                                # GuildStats: fact counts by tier/scope,
                                                               # queue depth, spend-to-budget, dedup rate
await memory.admin.set_retention(gid, days=180)                # per-guild override
```

Purge is the **only** destructive operation and is two-phase (dry-run report first) by
design. Export satisfies "give me my data" flows without raw DB access.

---

## Part 11 — Ops, Workers, Health

```python
await memory.ops.run_pending(limit_batches=10)   # cron/external-scheduler mode
                                                 # (workers.enabled=False deployments)
health = await memory.health()                   # HealthReport: storage ok, vector ok,
                                                 # queue depths, dead-letter count,
                                                 # last_worker_heartbeat, degraded components
snapshot = memory.ops.meter_snapshot()           # cumulative tokens/cost by purpose
await memory.ops.retry_dead_letters(gid=None)    # re-drive poison jobs after fixes
```

Worker loop internals (leases, heartbeats, reclaim) are invisible; consumers choose
only *where* it runs (embedded tasks vs dedicated process vs cron).

---

## Part 12 — Events (hooks)

Single subscription point, typed payloads, sync handlers dispatched via
`loop.call_soon` (handlers must be quick; heavy work should enqueue its own tasks):

```python
@memory.events.on(BatchCompleted)
def log_batch(evt: BatchCompleted) -> None:
    metrics.increment("memory.batches", tags={"guild": evt.guild_id})

# Event types: BatchCompleted(guild_id, subject_key, adds, reinforces,
#                              supersessions, invalidations)
#              FactCommitted(guild_id, fact_id, subject_id, text,
#                            was_reinforcement) · FactSupersededEvent(old_id, new_id)
#              ExtractionFailed(job_id, attempt, error_kind)
#              BudgetWarning(guild_id, fraction_used, next_ladder_step)
#              ComponentDegraded(component, reason)      # e.g. vector index unavailable
```

Rationale for existence: moderation UX, dashboards, and alerting without forcing every
consumer to implement a `Meter`. This is the *only* callback surface — no other
inversion-of-control exists by design (AGENTS.md: no speculative abstractions;
ports already cover behavioral swaps).

---

## Part 13 — Error Taxonomy

```
DiscordMemoryError                     # base; never raised for normal operation
├── ConfigError                        # bad/unknown config, bad URLs — fail fast at start()
├── SchemaValidationError              # malformed MessageEvent/FactUpdate/etc.
│                                      #   (wraps pydantic; naive datetimes included)
├── SubjectNotAllowedError             # bot-guard / opted-out / unknown-consent target
├── FactNotFoundError                  # stale id handed to facts.*
├── IdentityAmbiguousError             # raised ONLY by explicit resolve()-style calls
│   └── candidates: ResolvedCandidate  #   prompt_context degrades gracefully instead
├── StorageUnavailableError            # start()-time or ops calls; hot paths degrade
├── BudgetExceededError                # ops/admin writes when hard-stop configured
└── WorkerNotRunningError              # flush/run_pending without started workers
```

Rules: `observe` → `SchemaValidationError` only; `recall`/`prompt_context` → never
raise operational errors (return `warnings`/`degraded`); CRUD/admin raise the specific
subclass. Nothing shadows builtins (the bot-era temptation of `MemoryError` is
explicitly avoided).

---

## Part 14 — Result & Pagination Conventions

- All result models frozen; lists are tuples (hashable, safely shared).
- Cursor pagination wherever unbounded: `Page(items: tuple[T, ...],
  next_cursor: str | None)`; cursors are opaque strings, stable within a retention
  window.
- Datetimes: tz-aware UTC everywhere in results; naive inputs rejected at boundaries.
- IDs: facts get opaque prefixed ids (`fct_<ulid>`) — sortable, greppable in logs,
  never raw DB keys.
- Every score-bearing result includes its component breakdown; every name-derived
  result includes its `Resolution`.

---

## Part 15 — Integration Extra (discord.py)

```python
# pip install discord-memory[discord]
from discord_memory.integrations import setup_discord_memory

memory = await setup_discord_memory(
    bot,                                  # wires listeners: on_message→observe,
    config,                               #   on_member_update→rename handling,
)                                         #   on_ready→alias backfill; starts client
                                          #   on setup, closes on bot teardown

@bot.command()
async def remember(ctx, *, text: str):
    await memory.facts.remember(guild_id=str(ctx.guild_id), subject_id=str(ctx.author.id),
                                text=text, category=Category.GENERAL, actor_id=str(ctx.author.id))
```

Also shipped in the extra: a `MemoryCog` providing `/memory me`, `/memory forget <n>`,
owner-only purge/export/search commands, and the `apply_citations` reply hook. ~200
lines total, no core dependencies on discord.py (core tests never import it).

---

## Part 16 — What Is Deliberately NOT Exposed

| Temptation | Rejected because |
|---|---|
| Sync client variant | discord.py is async-native; a second client doubles surface + bugs. Recipe shows `asyncio.run` bridging instead |
| Generic middleware pipeline | Ports cover swaps; events cover observation; nothing else has justified itself |
| Direct embedding/vector manipulation | Breaks scoring calibration invariants; goes through `facts`/adapters |
| Raw SQL/aggregate escape hatch | Leaks backend dialect; breaks the swap guarantee (D1) |
| Per-call prompt overrides for extraction | Prompts are versioned products tied to gate behavior (PLAN §0.2.5); forks belong upstream |
| Auto-registration of arbitrary recall channels at runtime | Channels compose via `ChannelSet` from the shipped, benchmarked set; new channels = PR + conformance tests |

---

## Part 17 — Versioning & Stability

- SemVer; the import map (Part 2) is the compatibility surface, plus wire-stable
  `MemoryExport` and cursor opacity guarantees.
- Stored-schema migrations are automatic and backward-safe within a major version;
  `storage.schema_auto_migrate=False` opts into explicit migration scripts.
- Deprecations: exported alias + warning event for one minor cycle minimum, removal
  noted in changelog.
- Public entry points carry product-quality docstrings with doctest-style examples
  (AGENTS.md: public capability surface = product).

---

## Part 18 — Recipes

### 18.1 Ten-line quickstart

```python
from discord_memory import DiscordMemory, MemoryConfig, MessageEvent

memory = DiscordMemory(MemoryConfig(storage="sqlite:///m.db", llm="openai://KEY"))
async with memory:
    await memory.observe(MessageEvent(message_id="1", guild_id="g", channel_id="c",
        author_id="u1", content="I've been learning Rust for a year",
        created_at=datetime.now(UTC)))
    ctx = await memory.prompt_context(guild_id="g", asker_id="u1",
                                      text="what am I learning?")
    print(ctx.injection_block)
```

### 18.2 Turn loop with citations

See §6.1 — `prompt_context` → generate → `ctx.apply_citations(reply)`.

### 18.3 Dedicated worker process

Process A (bot): `MemoryConfig(..., workers={"enabled": False})`.
Process B: same storage URL, `while True: await memory.ops.run_pending();
await asyncio.sleep(5)`.

### 18.4 Backfill history

`await memory.observe_many(historical_events)` — old `created_at` values preserved;
extraction anchors relative-time phrases to original timestamps (PLAN §4.4).

### 18.5 Testing your bot with memory

`DiscordMemory(config, llm=ScriptedLLM(...), clock=FakeClock(...))` — deterministic
facts, time-travel TTL tests, zero network (mirrors the library's own harness).

---

## Part 19 — API Decision Register

| # | Decision | Alternatives rejected | Rationale |
|---|---|---|---|
| DA1 | Namespaced facade (`memory.facts/graph/admin/ops/events`) with 4 hot verbs top-level | Fully flat (mem0-style `add/search/delete`) | Hot path stays tiny; growth doesn't bloat root; matches discord.py conventions consumers know |
| DA2 | `observe` returns `ObserveReceipt`, never operational raises | Raise on queue-full/storage errors | Message listeners must never crash bots; policy ignores vs actionable rejects are distinct outcomes |
| DA3 | Flexible string identifiers + attached `Resolution` everywhere | Require pre-resolved IDs only | Frictionless; transparency preserves accuracy guarantees; snowflake shortcut avoids ladder cost |
| DA4 | Manual facts default to CORE tier | Treat manual same as extracted | Proven bot behavior (`/memory remember`); user-commanded facts are intentional, high-trust |
| DA5 | `classify_command` returns typed intent, execution stays consumer-side | Library auto-mutates on "forget that" | Consent and UX policy belong to the bot owner; library provides mechanics, not policy |
| DA6 | Single `events` subscription point, typed payloads | Per-hook kwargs, callback soup, full pubsub framework | Covers dashboard/moderation/alerting needs; smallest surface that justifies itself |
| DA7 | Two-phase purge (`dry_run` report → execute) | Immediate delete | Irreversible op deserves preview; enables compliance review flows |
| DA8 | Opaque ULID-prefixed fact IDs, cursor pages | Expose DB keys, offset pagination | Log-greppable, sortable, leak-free; stable pagination under concurrent writes |
