# icelake — Complete Usage Guide

Everything a bot developer needs to run, query, and operate the memory layer.
For design rationale see [`PLAN.md`](./PLAN.md); for the normative API contract
see [`API.md`](./API.md). Runnable code lives in [`/examples`](../examples/).

---

## 1. Mental model

```
                    ┌──────────────────────────────────────────┐
   your bot         │                 icelake                  │
──────────────────►│                                          │
 observe(event)    │  queue ──► lease worker ──► extraction    │──► facts (bitemporal)
                   │  (batching)  ▲ gates      ▲ reconcile     │    embeddings
 recall(query) ◄───│  vector+keyword+links+entity channels     │    incidence links
 prompt_context ◄──│  → RRF → calibrated rerank → budget       │    relation edges
                   │                                          │    profile digests
 admin/ops/events  │  maintenance: expiry · decay · caps       │    aliases · entities
                   └──────────────────────────────────────────┘
```

**The four-method hot path**: `observe` (learn), `prompt_context` (turn context),
`recall` (explicit query), `close`. Everything else is progressive disclosure via
capability groups: `memory.facts`, `memory.identity`, `memory.graph`,
`memory.admin`, `memory.ops`, `memory.events`.

**Invariants you can rely on:**

1. One fact, one owner anchor — `subject_id` (a person) or `guild_id` (the server).
2. Cross-user references are additive link rows; they never change ownership.
3. Nothing is ever hard-deleted except explicit purge. Contradictions invalidate;
   refinements supersede; both keep history.
4. The extraction LLM only ever sees minted participant tokens (`p0`, `p1`,
   `server`) — it cannot invent attribution targets.
5. Ambiguous names resolve to *nobody*, never to a guess.

---

## 2. Setup

```bash
pip install icelake                # SQLite + hashing embeddings (zero services)
pip install "icelake[mongo]"       # MongoDB backend
pip install "icelake[discord]"     # discord.py helpers
pip install "icelake[local-embeddings]"  # real local semantic embeddings
```

```python
from icelake import MemoryConfig

config = MemoryConfig(
    # storage: sqlite file | mongodb://… ([mongo] extra)
    storage="sqlite:///bot-memory.db",

    # any OpenAI-compatible endpoint: OpenRouter, OpenAI, Ollama, vLLM…
    llm="openai://$API_KEY@openrouter.ai/api/v1?model=google/gemini-2.5-flash",
    # omit llm entirely to run with manual facts + graph APIs only

    embeddings="hashing",        # free default; or "local"; or "openai://…"

    batching={"batch_size_messages": 10, "max_age_seconds": 90},
    budgets={"guild_daily_prompt_tokens": 200_000},
    workers={"enabled": True, "count": 2},
)
```

Unknown keys raise immediately. Every knob is documented in API.md Part 4.

### Choosing embeddings

| Option | Cost | Quality | When |
|---|---|---|---|
| `hashing` (default) | free | lexical-ish | small bots, CI, deterministic tests |
| `local` extra | free, CPU | good (384-dim MiniLM-class) | single-node production |
| `openai://text-embedding-3-small` | ~$0.02/Mtok | strong | multi-host deployments |

---

## 3. Learning from conversations

### Passive observation (the normal path)

```python
receipt = await memory.observe(MessageEvent(
    message_id=str(message.id),          # idempotency key — duplicates ignored
    guild_id=str(message.guild.id),
    channel_id=str(message.channel.id),
    author_id=str(message.author.id),
    content=message.content,
    created_at=message.created_at,       # tz-aware required
    author_display_name=member.display_name,   # feeds the alias index
    mention_ids=tuple(str(m.id) for m in message.mentions),
))
# receipt.status: ACCEPTED | IGNORED (bot/opted-out/empty/duplicate) | REJECTED
```

Observation is O(1): validate → persist → enqueue. Extraction happens on workers:

- batches flush at `batch_size_messages` **or** `max_age_seconds`;
- a keyed lease (`guild_id + subject_key`) makes N workers cooperative;
- noise gate skips pure chatter without an LLM call;
- extraction emits candidate facts referencing participants by roster token;
- quality gates drop refusals, raw quotes, questions, snowflakes-in-text,
  ephemeral media shares, and low-confidence claims;
- collisions against existing memories trigger one conditional reconcile call:
  NOOP→reinforce · UPDATE→supersede · INVALIDATE→contradict.

Force-drain when you need facts now (tests, shutdown, admin flush):

```python
processed = await memory.flush(guild_id=guild_id)
```

### Explicit teaching (ChatGPT-style "remember this")

```python
fact = await memory.facts.remember(
    guild_id=guild_id,
    subject_id=user_id,
    text="prefers mechanical keyboards",
    category=FactCategory.PREFERENCES,
    actor_id=user_id,                     # who taught it
    speaker_id=None,                      # set for third-party statements
    # optional knowledge-graph participation:
    entities=(ProposedEntity(name="Keyboards"),),
    relations=(ProposedRelation(verb="likes",
                                from_token=user_id,
                                to_entity="Keyboards"),),
)
# manual facts are CORE tier: never expire, exempt from pruning & forgetting
```

### Natural-language commands

```python
command = await memory.classify_command("hey bot remember that I hate pineapple")
# UserMemoryCommand(action="remember", target_text="that I hate pineapple", …)
if command.action == "remember" and command.confidence >= 0.85:
    await memory.facts.remember(guild_id=gid, subject_id=user_id,
                                text=command.target_text, actor_id=user_id)
elif command.action == "forget":
    ...  # find matching facts via facts.search, then facts.forget(...)
```

The library classifies; **your bot decides whether to execute** (consent policy is yours).
Recommended confidence floor for mutating actions: 0.85.

### History backfill

```python
receipts = await memory.observe_many(old_events)   # original timestamps preserved
await memory.flush()
```

Relative phrases in old messages ("last week") anchor to their original timestamps.

### Synchronous extraction (tests / onboarding)

```python
receipt = await memory.extract_now(event)   # observe + flush atomically;
fact_page = await memory.facts.list_for_subject(gid, user_id)  # facts ready now
```

Production reply paths should prefer `observe` + background workers.

---

## 4. Building turns (the read path)

### One-call turn context

```python
ctx = await memory.prompt_context(
    guild_id=guild_id,
    asker_id=asker_id,                    # who's talking -> CURRENT ASKER section
    text=message_text,                    # query + entity hints
    mentioned_ids=mentioned_ids,          # REFERENCED USER sections (cap 4 subjects)
    thread_participant_ids=(),            # optional extra roster
    token_budget_tokens=800,
)

system_prompt = PERSONA + "\n\n" + ctx.injection_block
reply = await your_llm(system_prompt, history, message_text)
reply = ctx.apply_citations(reply)        # [mem:N] echoes -> markdown jump links
```

What `prompt_context` guarantees:

- every section header states whose facts they are and forbids cross-attribution;
- members known by several names get a coreference line
  (*"these names all refer to ONE person"*);
- server-wide traits render separately from personal profiles;
- the block fits `token_budget_tokens` (trimmed sets a `BUDGET_TRIMMED` warning);
- citations bind only to injected facts — the model can echo, never fabricate;
- ambiguous name resolutions degrade gracefully (`warnings`) instead of guessing.

### Memory decay (opt-in, mem0-style)

```python
MemoryConfig(retrieval={"reinforce_on_recall": True})
```

Every fact served in a turn gets its decay clock reset in one batched write —
frequently-served knowledge floats up over time while stale facts sink toward
the forgetting threshold. Off by default; one extra indexed UPDATE per turn when
enabled. Strength still only grows on real re-observation, never on reads.

### Explicit queries (power users)

```python
result = await memory.recall(RecallQuery(
    guild_id=guild_id,
    text="who plays chess?",
    subject_ids=(alice_id,),            # Q1 profile focus
    scope=Scope.SUBJECTS,               # SUBJECTS | GUILD | SERVER
    pair_ids=(alice_id, bob_id),        # Q3 relationship mode
    entity_hint="chess",                # Q5 entity aggregation seed
    top_k=8, max_per_subject=4, min_score=0.3,
    channels=CHANNELS_DEFAULT,          # or CHANNELS_DISCOVERY (adds graph-hop)
))

for sf in result.facts:
    print(sf.score, sf.components, sf.fact.text)
if result.degraded_channels:
    ...  # e.g. vector index down — other channels still served the answer
```

### Time-travel recall (bitemporal)

```python
result = await memory.recall(RecallQuery(
    guild_id=guild_id,
    text="where does alice work",
    subject_ids=(alice_id,),
    as_of=datetime(2026, 1, 1, tzinfo=UTC),   # knowledge state on Jan 1
))
```

Facts valid at that instant surface — including ones since superseded or
invalidated. Present-time recall omits them.

### Relationship / graph queries

```python
resolution = await memory.identity.resolve(gid, "klim")   # ID | @mention | username |
                                                          # display name | saved name
if resolution.ambiguous:
    ...                                                   # never guesses

edges = await memory.graph.between(gid, x_id, y_id)        # typed edges + evidence ids
stances = await memory.graph.entity_stances(gid, "movies") # positive/negative aggregation
neighbors = await memory.graph.neighbors(gid, x_id, depth=2)  # hop paths included
similar = await memory.graph.similar_users(gid, x_id)      # Jaccard over shared entities
related = await memory.graph.relations_of(gid, x_id)       # everything touching X
```

Every edge carries `evidence_fact_ids` — show receipts, not vibes.

---

## 5. Facts lifecycle

```python
fact = await memory.facts.get(gid, fact_id)
updated = await memory.facts.update(fact_id, guild_id=gid,
                                    text="refined wording", reason="correction")
await memory.facts.forget(fact_id, guild_id=gid, reason="user request")
stronger = await memory.facts.reinforce(fact_id, guild_id=gid)
lineage = await memory.facts.history(fact_id, guild_id=gid)   # full audit chain
page = await memory.facts.list_for_subject(gid, user_id, limit=50)
hits = await memory.facts.search(gid, "keyboards", subject_ids=(uid,))
```

Semantics:

- `update` rewrites in place but appends a `superseded` history entry (v1 behavior);
  pipeline-level refinements create full supersession chains.
- `forget` soft-invalidates (reversible by history reference); only `admin.purge_user`
  erases.
- Manual facts are CORE tier: no TTL, exempt from pruning and forgetting.

## 6. Identity

```python
res = await memory.identity.resolve(gid, identifier)
# res.resolved: ResolvedCandidate(user_id, matched_alias, source, weight, confidence)
# res.ambiguous: True  => candidates present, resolved None — DO NOT attribute

await memory.identity.register_alias(gid, user_id, "dr. krill",
                                     source=AliasSource.DISPLAY_NAME)
await memory.identity.handle_member_rename(gid, user_id, new_display_name)
aliases = await memory.identity.aliases_of(gid, user_id)
```

Rules: snowflake-shaped strings pass through untouched; ambiguity requires
disambiguation upstream of any write; renames keep old aliases valid-but-decaying.

## 7. Governance

```python
await memory.admin.set_opt_out(gid, user_id, opted_out=True)   # enforced everywhere, instantly

report = await memory.admin.purge_user(gid, user_id, dry_run=True)   # preview
report = await memory.admin.purge_user(gid, user_id, dry_run=False)  # erase:
# facts, aliases, summaries, vectors, incidence rows, relation endpoints —
# while OTHER members' facts merely lose their links to the purged user.

export = await memory.admin.export_guild(gid)   # wire-stable JSON export
stats = await memory.stats(gid)                 # GuildStats snapshot
```

## 8. Identity grounding (cold start)

Users who never spoke since install are invisible to the alias ladder until you
backfill from your member directory — usually once in `on_ready`:

```python
members = [
    (str(m.id), m.name, m.display_name)
    for m in guild.members
    if not m.bot
]
registered = await memory.ops.backfill_aliases(guild_id, members)
```

After this, "what does klim think?" resolves even though klim never messaged the bot.
Additionally, facts written through `facts.remember` accept `subject_username=` to
register the subject's name at write time, and `prompt_context` falls back to
searching stored name-facts ("my name is klim") when the alias ladder misses —
flagged, never auto-attributed.

## 9. Temporal queries

The data model is bitemporal; `as_of` exposes it:

```python
result = await memory.recall(RecallQuery(
    guild_id=guild_id,
    text="where does alice work",
    subject_ids=(alice_id,),
    as_of=datetime(2026, 1, 1, tzinfo=UTC),   # what did we know on Jan 1?
))
```

Facts valid at that instant surface, including ones since superseded.

## 10. Operations

```python
health = await memory.ops.health()          # storage/llm status, queues, dead letters
snapshot = memory.ops.meter_snapshot()      # tokens/calls/cost per purpose
await client.ops.retry_dead_letters()       # re-drive poison jobs after a fix
count = await memory.regenerate_summaries(gid)  # manual digest refresh
```

Worker topologies:

| Topology | Config |
|---|---|
| Single process | defaults — workers run as background tasks |
| Split bot / worker | bot: `workers={"enabled": False}`; worker: same storage + `ops.run_pending()` loop |
| Cron | workers off; call `ops.run_pending()` from your scheduler |

Multi-process is safe by construction: batch leases are atomic and expire.

Budgets: `budgets.guild_daily_prompt_tokens` triggers a degradation ladder
(warn → skip reconcile → skip extraction) *before* overspending. Retrieval is always
free and never blocked.

## 11. Events

```python
from icelake import BatchCompleted, FactCommitted

@memory.events.on(BatchCompleted)
def metrics(evt: BatchCompleted) -> None:
    statsd.increment("memory.batch", tags={"guild": evt.guild_id})

@memory.events.on(FactCommitted)
def announce(evt: FactCommitted) -> None:
    if evt.was_reinforcement:
        ...
```

Typed payloads: `BatchCompleted`, `FactCommitted`, `FactSupersededEvent`,
`ExtractionFailed`, `BudgetWarning`, `ComponentDegraded`.

## 12. Testing your integration

```python
from tests.conftest import ScriptedLLM          # pattern to copy
memory = DiscordMemory(config, llm=ScriptedLLM({"extraction": '{"operations": []}'}),
                       clock=FixedClock(start))
```

Deterministic LLM scripts + fake clock + in-memory/SQLite backends = fast, hermetic
tests of your bot's memory behavior. See `tests/conftest.py` for fakes and
`tests/integration/` for scenario inspiration.

## 13. Production checklist

- [ ] Real embedding provider configured (`local` extra or OpenAI-compatible)
- [ ] Daily budget set (`budgets.guild_daily_prompt_tokens`)
- [ ] Worker topology chosen; drain on shutdown (`close(drain=True)`)
- [ ] Opt-out + purge wired into user-facing commands (compliance)
- [ ] Dead-letter alerting (`ops.health().dead_letters` or `ExtractionFailed` event)
- [ ] `on_member_update` → rename handling (or the `[discord]` cog, which does it)
- [ ] Backfill imported with original timestamps if migrating history
