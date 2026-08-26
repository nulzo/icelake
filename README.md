# discord-memory

Accurate, scalable, cost-effective **agentic memory for Discord bots** — ChatGPT/Claude-style
memory of your users, hardened against cross-user attribution errors, working across every
member of a server.

> Priority order (non-negotiable): **Accuracy → Cost → Performance**.

- **Accurate**: facts attach to *hardened* Discord user IDs via a roster-token protocol that
  structurally prevents LLM hallucinated attribution; third-party statements ("X called Y a
  hacker") anchor on the person they're about, with the speaker kept as attribution.
- **Cost-effective**: batched extraction (~1 LLM call per ~10 messages), conditional
  reconciliation (phase-2 fires only on collisions), zero-LLM retrieval, free local embeddings.
- **Scalable**: durable lease queue safe across processes, guild-partitioned storage,
  hub-aware bounded graph traversal, per-guild budgets with graceful degradation.
- **Composable**: every external dependency sits behind a Protocol port — swap storage,
  LLM provider, embedder, or clock with one constructor argument.

## Install

```bash
pip install discord-memory                    # core (SQLite backend, hashing embedder)
pip install "discord-memory[discord]"         # + discord.py integration
pip install "discord-memory[mongo]"           # + MongoDB backend (PyMongo Async)
pip install "discord-memory[local-embeddings]"# + sentence-transformers embeddings
```

Requires Python ≥ 3.12.

## Quickstart

```python
import asyncio
from datetime import UTC, datetime

from discord_memory import DiscordMemory, MemoryConfig, MessageEvent


async def main() -> None:
    memory = DiscordMemory(MemoryConfig(
        storage="sqlite:///memory.db",
        llm="openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1"
            "?model=google/gemini-2.5-flash",
    ))
    async with memory:
        # 1. Feed it messages (fire-and-forget; extraction happens in background)
        receipt = await memory.observe(MessageEvent(
            message_id="9001", guild_id="555", channel_id="777",
            author_id="100000000000000001",
            content="I've been learning Rust for about a year now!",
            created_at=datetime.now(UTC),
            author_display_name="alice",
        ))

        # 2. Build prompt context for a reply: asker + mentioned users + server.
        ctx = await memory.prompt_context(
            guild_id="555",
            asker_id="100000000000000001",
            text="what am I learning these days?",
            mentioned_ids=("200000000000000002",),   # @mentions in this message
        )
        print(ctx.injection_block)   # labeled, budgeted, cite-tagged block

        # 3. After generation, resolve echoed [mem:N] tags into jump links.
        reply = ctx.apply_citations("You're learning Rust [mem:1]!")


asyncio.run(main())
```

Runnable, complete examples live in [`examples/`](examples/):

| File | What it demonstrates |
|---|---|
| [`examples/ping_reply_bot.py`](examples/ping_reply_bot.py) | **The classic chat bot** (like CringeDiscordBot): passive learning on every message, replies only when pinged, resolves asker + referenced-user memories per turn, applies citations to the reply, chat-native "remember/forget" commands, nickname-change tracking |
| [`examples/relationship_queries.py`](examples/relationship_queries.py) | Cross-user memory: "what does X think of Y", typed relation edges, entity stance aggregation ("who likes movies?"), shared-trait discovery. Runnable without Discord |

### The ping-reply turn, step by step

When `@Bot what happened between alice and bob?` arrives:

```python
# 1. Resolve memories for EVERYONE in the conversation in one call:
ctx = await memory.prompt_context(
    guild_id=guild_id,
    asker_id=str(message.author.id),       # who is talking -> their profile
    text=question,                          # query + entity hints
    mentioned_ids=("alice_id", "bob_id"),   # referenced users' profiles
)

# ctx.injection_block is labeled so facts never bleed across users:
#
#   [MEMORY CONTEXT]
#
#   WHAT I KNOW ABOUT THE CURRENT ASKER
#   Facts about the asker ONLY:
#   - [mem:1] alice mains support in every ranked game she plays
#
#   REFERENCED USER: bob
#   Facts about bob ONLY. Do NOT attribute these to the asker.
#   - [mem:2] bob was called a hacker by alice during the ranked match
#
#   SERVER COMMUNITY FACTS
#   Community-wide traits:
#   - [mem:3] the community bonds over late night gaming sessions
#
#   When you use a fact above in your reply, echo its [mem:N] tag ...

# 2. Generate your LLM reply using system_prompt + ctx.injection_block.

# 3. Resolve echoed tags into jump links (deleted-message safe):
reply = ctx.apply_citations(reply_text)
#   "... bob was called out [[mem:2]](https://discord.com/channels/...)"
```

### Cross-user questions ("what does X think about Y")

Names resolve through the alias ladder (mention ID / username / display name /
saved real name); ambiguity never guesses:

```python
resolution = await memory.identity.resolve(guild_id, "klim")     # or snowflake/@mention
if resolution.ambiguous:
    ...  # ask which member; never guess

edges = await memory.graph.between(guild_id, x_id, y_id)         # typed edges
stances = await memory.graph.entity_stances(guild_id, "movies")  # opposing stances co-presented
neighbors = await memory.graph.neighbors(guild_id, x_id, depth=2)  # hop discovery w/ paths
similar = await memory.graph.similar_users(guild_id, x_id)       # Jaccard over traits
```

### Chat-native commands (ChatGPT style)

```python
command = await memory.classify_command("hey bot remember that I hate pineapple")
# UserMemoryCommand(action="remember", target_text="that I hate pineapple", confidence=0.9)
if command.action == "remember":
    await memory.facts.remember(
        guild_id=guild_id, subject_id=user_id,
        text=command.target_text, actor_id=user_id,
    )
```

Manual facts accept graph participation too:

```python
await memory.facts.remember(
    guild_id=guild_id, subject_id=bob_id,
    text="carol called bob a sore loser during game night",
    actor_id=carol_id, speaker_id=carol_id,          # third-party attribution
    relations=(ProposedRelation(
        verb="called_out", from_token=carol_id, to_token=bob_id),),
)
```

### Governance every production bot should wire

```python
@bot.command()
async def forgetme(ctx: commands.Context):
    await memory.admin.purge_user(str(ctx.guild.id), str(ctx.author.id),
                                  dry_run=False)
    await ctx.reply("All memories about you have been purged.", mention_author=False)

# opt-out is enforced instantly across observe AND recall:
await memory.admin.set_opt_out(guild_id, user_id, True)
```

Full usage documentation: [`docs/USAGE.md`](docs/USAGE.md) · Complete API contract:
[`docs/API.md`](docs/API.md) · Design rationale: [`docs/PLAN.md`](docs/PLAN.md).

---

## Full bot example (omni-style)

For a production-shaped deployment — passive learning for everyone, replies only when
addressed, requester-first multi-person context, `/memory` slash group, governance — see
[`examples/omni_style_bot.py`](examples/omni_style_bot.py). It mirrors the architecture of
a memory-native production bot:

- **Composition root**: `build_memory()` is the single place config → adapters → client
  get wired; everything else receives `memory`.
- **Learn from everyone, answer the addressed**: `on_message` observes every message
  (bots included — they're registered as never-a-subject), then answers only when pinged
  **or replied to**.
- **Requester-first turn context** (capped at 4 subjects): asker + @mentions + reply-target,
  resolved in one `prompt_context` call with per-person labeled sections.
- **Coreference lines**: members known by several names get an explicit
  *"these names all refer to ONE person"* line so the model never splits them.
- **`/memory` group**: `show` (profile w/ aliases), `related` (typed edges),
  `shared` (common entities), `edit` (teach a fact), `alias` (teach a nickname).
- **Governance built in**: `/forgetme`, `/optout`, daily guild budgets, health.

```python
class OmniStyleBot(commands.Bot):
    def __init__(self, memory: DiscordMemory) -> None:
        ...
        self.memory = memory

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.guild is None:
            return
        await self.memory.observe(to_event(message))     # learn; never blocks
        if message.author.bot:
            return
        if await self._is_addressed(message):             # ping or reply-to-us
            await self._handle_turn(message)

    async def _handle_turn(self, message: discord.Message) -> None:
        question = strip_bot_mention(message.content, self.user.id).strip()
        subjects = await self._collect_subjects(message)   # mentions + reply target
        ctx = await self.memory.prompt_context(
            guild_id=guild_id,
            asker_id=str(message.author.id),
            text=question,
            mentioned_ids=tuple(subjects),
            token_budget_tokens=800,
        )
        system_prompt = PERSONA + "\n\n" + ctx.injection_block
        reply = await generate(system_prompt, history, question)
        await message.reply(ctx.apply_citations(reply)[:1900], mention_author=False)
```

The injection block a turn produces looks like:

```
[MEMORY CONTEXT]

REFERENCED USER: bob
Coreference: these names all refer to ONE person: bob, bobert, bobby.
Facts about bob ONLY. Do NOT attribute these to the asker.
- [mem:1] bob was called a hacker by alice during the ranked match

SERVER COMMUNITY FACTS
Community-wide traits:
- [mem:2] the community bonds over late night ranked gaming sessions

When you use a fact above in your reply, echo its [mem:N] tag so the user
can see the source. Do not invent tags for facts that were not listed.
```

---

## How it works

```
observe(event) ──► pending message queue ──► lease worker (per guild+author)
                                               │  noise gate (skip ~30–60% free)
                                               │  roster tokens (<p0>, <p1>, server)
                                               ▼
                                          LLM extraction ──► quality gates
                                               │
                     collision? ◄── yes ──► reconcile LLM (ADD/UPDATE/INVALIDATE/NOOP)
                          │ no                                   │
                          ▼                                      ▼
                      ADD fact ◄──────────────── SUPERSEDE / INVALIDATE (history kept)
                          │
     ┌────────────────────┼─────────────────────┐
     ▼                    ▼                     ▼
 fact store        vector index           knowledge graph
 (bitemporal,      (ANN/brute force,      incidence links + typed relation edges
  supersede chain)  scope-prefiltered)    (X—likes→movies · X—called_out→Y)
```

### Accuracy model

- **Roster-token protocol** — the extraction LLM references participants only by opaque
  tokens we mint (`p0`, `p1`, `server`). It never sees or emits snowflakes; unknown tokens
  are dropped before storage. Hallucinated attribution becomes structurally impossible.
- **Anchoring invariant** — every fact has exactly one owner anchor (subject user or the
  guild). Linking between users/entities is additive, never required.
- **Truth maintenance** — contradictions *invalidate* (bitemporal `valid_until`) or
  *supersede* (refinement chains); nothing auto-deletes. Full audit history per fact.
- **Quality gates** — refusals, LLM meta-talk, raw quotes (≥0.88 similarity), questions,
  snowflakes in text, ephemeral media shares, and low-confidence claims are all rejected
  by pure, exhaustively tested gates.
- **Identity ladder** — display names/usernames/saved names resolve through an alias index
  (source-ranked weights); ambiguity never guesses.

### Query shapes (all zero-LLM by default)

| Shape | Example | API |
|---|---|---|
| Profile | "what do you know about X" | `recall(subject_ids=(x,))` |
| Cross-linked | "did X call Y a hacker?" | facts touching both via link intersect |
| Relationship | "what does X think about Y" | `graph.between(x, y)` |
| Entity stance | "who likes movies?" | `graph.entity_stances("movies")` |
| Hop discovery | "shared connections of X" | `graph.neighbors(x, depth=2)` |

## The consumer surface

```python
memory.observe(event)                  # → ObserveReceipt (never raises operational errors)
memory.observe_many(events)            # bulk backfill
memory.flush(guild_id=...)             # force-extract pending batches now

memory.prompt_context(...)             # → PromptContext (injection block + citations)
memory.recall(RecallQuery(...))        # explicit query model

memory.facts.remember/update/forget/reinforce/history/list_for_subject/search
memory.identity.resolve/register_alias/handle_member_rename/aliases_of
memory.graph.between/entity_stances/neighbors/relations_of/similar_users*
memory.admin.set_opt_out/purge_user/export_guild/get_opt_out
memory.ops.run_pending/retry_dead_letters/meter_snapshot/health
memory.events.subscribe(BatchCompleted, handler)   # typed hook events
memory.classify_command(text)          # "remember that…" / "forget…" intent detection
memory.stats(guild_id)                 # GuildStats snapshot
```


Full contract with signatures and semantics: [`docs/API.md`](docs/API.md).
Design rationale and architecture: [`docs/PLAN.md`](docs/PLAN.md).

## discord.py integration

```python
# pip install discord-memory[discord]
import discord
from commands.bot import bot
from discord_memory import MemoryConfig
from discord_memory.integrations import setup_discord_memory, MemoryCog

config = MemoryConfig(
    storage="sqlite:///bot-memory.db",
    llm="openai://$KEY@openrouter.ai/api/v1?model=google/gemini-2.5-flash",
)

class MyBot(commands.Bot):
    async def setup_hook(self) -> None:
        memory, cog = await setup_discord_memory(self, config)
        await self.add_cog(cog)          # /memory me · /memory remember · /memory forget_me
        self.memory = memory
```

The integration wires `on_message → observe`, `on_member_update → alias refresh`, and
`on_ready → start`. Everything else stays in your control.

## Configuration

Providers are URL strings; nested typed configs also accepted:

```python
MemoryConfig(
    storage="sqlite:///memory.db",                       # or mongodb://… ([mongo] extra)
    llm="openai://$KEY@openrouter.ai/api/v1?model=…",    # OpenAI-compatible endpoints
    embeddings="hashing",                                # free default
    # embeddings="local",                                 # sentence-transformers extra
    # embeddings="openai://$KEY@api.openai.com/v1?model=text-embedding-3-small",
    batching={"batch_size_messages": 10, "max_age_seconds": 300},
    budgets={"guild_daily_prompt_tokens": 200_000},      # graceful degradation ladder
    privacy={"store_raw_messages": True},
    workers={"enabled": True, "count": 2},
)
```

Unknown keys raise immediately — typo protection by construction.

## Deployment topologies

| Topology | Config |
|---|---|
| Single process (small bots) | defaults — workers run as background tasks |
| Split bot + worker | bot: `workers={"enabled": False}`; worker process: same storage, call `await memory.ops.run_pending()` in a loop |
| Cron-style | workers disabled; invoke `ops.run_pending` from your scheduler |
| Multi-process scale-out | any number of processes share one database — keyed leases make workers cooperative |

## Extending (ports)

Every dependency is a Protocol you can replace at construction:

```python
memory = DiscordMemory(
    config,
    store=MyPostgresStore(),       # implements MemoryStore (+ optional .queue/.vectors)
    llm=MyLLM(),                   # implements ChatLLM (OpenAI-compatible shape)
    embedder=MyEmbedder(),         # implements Embedder
    clock=FakeClock(...),          # deterministic time in tests
)
```

New backends must pass the executable conformance suite
(`tests/integration/test_store_conformance.py`) — the port contract is literally a test.

## Development

```bash
uv sync --group dev
uv run pytest tests/ -q --cov=discord_memory   # ≥90% coverage enforced
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy                                    # strict mode
```

## Status & limitations (v0.1)

- Storage backends shipped: **SQLite** (default) + in-memory (tests). Postgres/pgvector
  adapter planned (M3) — the port is ready; Mongo adapter RFC open.
- Budgets meter per-process; cross-process budget accounting needs store-backed counters.
- Server-scope ("community") batches read the recent-message window; watermarking across
  restarts is best-effort.
- `similar_users` uses capped Jaccard over entity adjacency (no Louvain/PPR by design).

See [`docs/PLAN.md`](docs/PLAN.md) Part 11 for the phased roadmap and exit criteria.

## License

MIT
