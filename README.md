# discord-memory

Accurate, scalable, cost-effective **agentic memory for Discord bots** — ChatGPT/Claude-style
memory of your users, hardened against cross-user attribution errors, working across every
member of a server.

> Priority order (non-negotiable): **Accuracy → Cost → Performance**.

- **Accurate**: facts attach to *hardened* Discord user IDs via a roster-token protocol that
  structurally prevents LLM hallucinated attribution; third-party statements ("X called Y a
  hacker") anchor on the person they're about, with the speaker kept as attribution.
- **Cost-effective**: batched extraction (~1 LLM call per ~10 messages), conditional
  reconciliation (phase-2 fires only on collisions), zero-LLM retrieval, pluggable embeddings.
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
            "?model=google/gemini-3.7-flash",
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
| [`examples/omni_style_bot.py`](examples/omni_style_bot.py) | **Production-shaped bot**: passive learning, ping/reply turns, `/memory` slash group, OpenRouter chat + embeddings |
| [`examples/ping_reply_bot.py`](examples/ping_reply_bot.py) | Classic chat bot: observe every message, reply when pinged, citations, remember/forget, nickname tracking |
| [`examples/relationship_queries.py`](examples/relationship_queries.py) | Zero-LLM graph demo: 8 members, pair recall, stances, 2-hop neighbors. No Discord required |
| [`examples/e2e_simulation.py`](examples/e2e_simulation.py) | Public-API eval: 81 hard invariants + model expectations against a scripted guild |
| [`examples/bench_models.py`](examples/bench_models.py) | Parallel model matrix; writes JSON + Markdown reports |

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
from discord_memory.models.operations import ProposedRelation

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
[`docs/API.md`](docs/API.md) · Design: [`docs/PLAN.md`](docs/PLAN.md) · What ships next:
[`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Full bot example (omni-style)

For a production-shaped deployment — passive learning for everyone, replies only when
addressed, requester-first multi-person context, `/memory` slash group, governance — see
[`examples/omni_style_bot.py`](examples/omni_style_bot.py). It mirrors the architecture of
a memory-native production bot and wires **OpenRouter** for both chat (`google/gemini-3.7-flash`)
and embeddings (`openai/text-embedding-3-small`) so reconcile collisions and recall work on
paraphrases, not just exact text matches.

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

```mermaid
flowchart TB
  observe["observe(event)"] --> queue["Pending message queue"]
  queue --> worker["Lease worker<br/>one claim per guild + author"]
  worker --> noise{"Noise gate"}
  noise -->|chatter| skip["Ack - no LLM"]
  noise -->|worth extracting| roster["Mint roster tokens<br/>p0, p1, server"]
  roster --> extract["LLM extraction"]
  extract --> schema{"Valid JSON schema?"}
  schema -->|no after one repair| dead["Dead-letter the batch"]
  schema -->|yes| gates["Quality gates"]
  gates --> hit{"Near-duplicate collision?"}
  hit -->|no| add["ADD fact"]
  hit -->|yes| recon["Reconcile LLM"]
  recon --> add
  recon --> history["SUPERSEDE or INVALIDATE<br/>history kept"]
  add --> store["Fact store<br/>bitemporal, one subject anchor"]
  add --> vectors["Vector index"]
  add --> graph["Knowledge graph<br/>incidence + typed edges"]
  add --> digest["Profile digest<br/>every N new facts"]
```

`observe` is fire-and-forget: persist + enqueue, then return. Workers claim with keyed
leases so multiple processes cannot double-extract the same author. Invalid extraction
JSON is repaired once, then dead-lettered rather than silently acked empty.

### Accuracy model

- **Roster-token protocol** — identity fields (`subject_token`, `speaker_token`, relation
  endpoints) may only use tokens we minted for this batch (`p0`, `p1`, `server`). The
  model never sees Discord snowflakes; unknown tokens are dropped. Stored *prose* uses
  display names (with a detokenize pass if the model leaks `p0` into `text`). Attribution
  is the snowflake on `subject_id`, not the name in the sentence — renames add aliases;
  they do not move rows.
- **Anchoring invariant** — every fact has exactly one owner (a user or the guild).
  Links between people/entities are additive.
- **Truth maintenance** — contradictions *invalidate* (bitemporal `valid_until`) or
  *supersede* (refinement chain); nothing auto-deletes. Citations and related-user
  links survive supersede. Full audit history per fact.
- **Quality gates** — refusals, LLM meta-talk, raw quotes (≥0.88 similarity), questions,
  snowflakes in text, ephemeral media shares, and low-confidence claims are rejected by
  pure, unit-tested gates. Registered bots are never subjects and are stripped from
  mention links.
- **Identity ladder** — usernames, display names, and saved real names resolve through
  a source-ranked alias index. Ambiguity never guesses.
- **Profile digests** — a paragraph summary regenerates after
  `extraction.auto_consolidate_after_adds` new facts (default 5; not on a timer),
  stamped with the library clock.

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
memory.register_bot_id(bot_user_id)    # never a memory subject; stripped from mention links

memory.prompt_context(...)             # → PromptContext (injection block + citations)
memory.recall(RecallQuery(...))        # explicit query model

memory.facts.remember/update/forget/reinforce/history/list_for_subject/search
memory.identity.resolve/register_alias/handle_member_rename/aliases_of
memory.graph.between/entity_stances/neighbors/relations_of/similar_users
memory.admin.set_opt_out/purge_user/export_guild/get_opt_out
memory.ops.run_pending/retry_dead_letters/meter_snapshot/health
memory.events.subscribe(BatchCompleted, handler)   # typed hook events
memory.classify_command(text)          # "remember that…" / "forget…" intent detection
memory.regenerate_summaries(guild_id)  # force profile digests (else every N new facts)
memory.stats(guild_id)                 # GuildStats snapshot
```


Full contract with signatures and semantics: [`docs/API.md`](docs/API.md).
Design: [`docs/PLAN.md`](docs/PLAN.md) · Roadmap: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## discord.py integration

```python
# pip install discord-memory[discord]
from discord.ext import commands
from discord_memory import MemoryConfig
from discord_memory.integrations import setup_discord_memory

config = MemoryConfig(
    storage="sqlite:///bot-memory.db",
    llm="openai://$KEY@openrouter.ai/api/v1?model=google/gemini-3.7-flash",
)

class MyBot(commands.Bot):
    async def setup_hook(self) -> None:
        memory, helpers = await setup_discord_memory(self, config)
        self.memory = memory
        # helpers.me / helpers.remember / helpers.forget_me — bind to your own slash commands
```

The integration wires `on_message → observe`, `on_member_update → alias refresh`, and
`on_ready → start` + `register_bot_id`. The returned `MemoryCog` is a helper with
`me` / `remember` / `forget_me` methods you bind to slash commands in your bot — it is
not a `commands.Cog` subclass. For a full `/memory` slash group, see
[`examples/omni_style_bot.py`](examples/omni_style_bot.py).

## Configuration

Providers are URL strings; nested typed configs also accepted:

```python
MemoryConfig(
    storage="sqlite:///memory.db",                       # or mongodb://… ([mongo] extra)
    llm="openai://$KEY@openrouter.ai/api/v1?model=…",    # OpenAI-compatible endpoints
    embeddings="hashing",                                # free default (see below)
    # embeddings="local",                                 # sentence-transformers extra
    # embeddings="openai://$KEY@openrouter.ai/api/v1?model=openai/text-embedding-3-small",
    # embeddings="openai://$KEY@api.openai.com/v1?model=text-embedding-3-small",
    batching={"batch_size_messages": 10, "max_age_seconds": 300},
    extraction={"auto_consolidate_after_adds": 5},       # 0 disables profile digests
    budgets={"guild_daily_prompt_tokens": 200_000},      # graceful degradation ladder
    privacy={"store_raw_messages": True},
    workers={"enabled": True, "count": 2},
)
```

LLM URL query knobs that matter in production: `model`, `temperature=none` (omit sampling
on reasoning endpoints), `reasoning=low`, `max_tokens`, `max_tokens_key=max_completion_tokens`
(Azure), `structured_outputs=json_object` if the endpoint cannot enforce `json_schema`.
Capability mismatches raise `LlmCapabilityError` instead of silently degrading.

`postgresql://` is recognized and rejected with a clear error — there is no Postgres adapter
yet (and no `[postgres]` extra).

Unknown keys raise immediately — typo protection by construction.

### Embeddings (`embeddings=`)

| Provider | Spec | Best for |
|---|---|---|
| **Hashing** (default) | `"hashing"` | Tests, zero-dependency demos, deterministic CI |
| **Hosted** | `"openai://$KEY@openrouter.ai/api/v1?model=openai/text-embedding-3-small"` | Production bots already on OpenRouter/OpenAI |
| **Local** | `"local"` | Air-gapped or no embedding API cost (`pip install discord-memory[local-embeddings]`) |

Embeddings power **semantic recall**, **reconcile collision detection** (paraphrase → reinforce
instead of duplicate ADD), and consolidation sanity checks. Cosine similarity is compared against
`extraction.reconcile_collision_threshold` (default `0.85`).

#### Hashing embedder limitations

The default hashing embedder is a signed feature-hash over word/char n-grams — **not** a neural
model. It is fast, free, and reproducible, but:

- **Paraphrases do not cluster.** "Loves coding in Go" and "Enjoys programming in Go" often score
  below the reconcile threshold, so the pipeline treats them as unrelated facts and you can end up
  with many near-duplicate memories per user.
- **Recall is lexical-ish.** Vector search channels rank by token overlap more than meaning;
  semantic recall quality is noticeably weaker than with a real embedding model.
- **Reinforcement depends on collisions.** Ingest reinforce/update/noop only triggers semantic
  collision when cosine similarity clears the threshold; hashing misses most real-world re-statements.

Use **hosted** (OpenRouter/OpenAI) or **local** embeddings for any deployment where users repeat
the same preference in different words — including the omni-style example, which sets
`embeddings=EMBEDDINGS_URL` accordingly. Switching embedders invalidates existing vectors; re-embed
or start fresh on dev databases when changing provider.

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

- Storage backends shipped: **SQLite** (default), **MongoDB** (`[mongo]` extra), and in-memory
  (tests). A Postgres/pgvector adapter is planned — `postgresql://` fails loudly today.
- Default **hashing** embeddings are not suitable for production dedup/recall — see
  [Embeddings](#embeddings-embeddings) above.
- Invalid extraction JSON is repaired once, then **dead-lettered** (not silently stored as
  empty). Re-drive with `ops.retry_dead_letters`.
- Caps and TTL prune weakest-first (manual/CORE last). Budgets meter per-process; cross-process
  budget accounting needs store-backed counters.
- Server-scope ("community") batches read the recent-message window; watermarking across
  restarts is best-effort.
- `similar_users` uses capped Jaccard over entity adjacency (no Louvain/PPR by design).

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for sequenced next work and
[`docs/PLAN.md`](docs/PLAN.md) for design rationale.

## License

MIT
