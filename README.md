# icelake

Persistent memory layer specifically for discord bots. Passively consumes messages, extracts facts about users, and hands you a labeled block to put in your system prompt when you need to reply.

Facts are stored against Discord user IDs (as opposed to names), so a rename does not move someone else's memories onto a new person. Third-party claims ("alice called bob
a hacker") attach to the person they are about.

Requires Python 3.12+.

## Install

```bash
pip install icelake
pip install "icelake[discord]"           # discord.py helpers/hooks
pip install "icelake[mongo]"             # MongoDB backend
pip install "icelake[local-embeddings]"  # sentence-transformers
```

## Quickstart

```python
import asyncio
from datetime import UTC, datetime

from icelake import DiscordMemory, MemoryConfig, MessageEvent


async def main() -> None:
    memory = DiscordMemory(MemoryConfig(
        storage="sqlite:///memory.db",
        llm="openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1"
            "?model=google/gemini-3.7-flash",
    ))
    async with memory:
        await memory.observe(MessageEvent(
            message_id="9001",
            guild_id="555",
            channel_id="777",
            author_id="100000000000000001",
            content="I've been learning Rust for about a year now!",
            created_at=datetime.now(UTC),
            author_display_name="alice",
        ))

        ctx = await memory.prompt_context(
            guild_id="555",
            asker_id="100000000000000001",
            text="what am I learning these days?",
            mentioned_ids=("200000000000000002",),
        )
        print(ctx.injection_block)

        reply = ctx.apply_citations("You're learning Rust [mem:1]!")


asyncio.run(main())
```

`observe` returns immediately. Extraction runs in the background.

`prompt_context` builds a block with separate sections for the asker, anyone
they mentioned, and the server. Stick it on your system prompt, generate a
reply, then run `ctx.apply_citations` so `[mem:N]` tags become jump links.

Examples:

- [`examples/omni_style_bot.py`](examples/omni_style_bot.py) - full bot: learns from everyone, replies when pinged or replied to, `/memory` slash commands
- [`examples/ping_reply_bot.py`](examples/ping_reply_bot.py) - smaller ping-reply bot
- [`examples/relationship_queries.py`](examples/relationship_queries.py) - graph queries with no Discord or LLM
- [`examples/e2e_simulation.py`](examples/e2e_simulation.py) - scripted-guild eval
- [`examples/bench_models.py`](examples/bench_models.py) - model matrix

## A reply turn

```python
ctx = await memory.prompt_context(
    guild_id=guild_id,
    asker_id=str(message.author.id),
    text=question,
    mentioned_ids=("alice_id", "bob_id"),
)

# ctx.injection_block looks like:
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

reply = await generate(system_prompt + "\n\n" + ctx.injection_block, question)
await message.reply(ctx.apply_citations(reply), mention_author=False)
```

## Names, relationships, commands

```python
resolution = await memory.identity.resolve(guild_id, "klim")
if resolution.ambiguous:
    ...  # ask which member. do not guess

edges = await memory.graph.between(guild_id, x_id, y_id)
stances = await memory.graph.entity_stances(guild_id, "movies")
neighbors = await memory.graph.neighbors(guild_id, x_id, depth=2)
similar = await memory.graph.similar_users(guild_id, x_id)
```

Names go through mention ID, username, display name, then saved real name.
If more than one member matches, you get `ambiguous` instead of a guess.

```python
from icelake import CommandAction

command = await memory.classify_command("hey bot remember that I hate pineapple")
# UserMemoryCommand(action=CommandAction.REMEMBER, target_text="that I hate pineapple", ...)
if command.action is CommandAction.REMEMBER:
    await memory.facts.remember(
        guild_id=guild_id, subject_id=user_id,
        text=command.target_text, actor_id=user_id,
    )
```

Third-party facts can carry a relation:

```python
from icelake import ProposedRelation, RelationVerb

await memory.facts.remember(
    guild_id=guild_id, subject_id=bob_id,
    text="carol called bob a sore loser during game night",
    actor_id=carol_id, speaker_id=carol_id,
    relations=(ProposedRelation(
        verb=RelationVerb.CALLED_OUT, from_token=carol_id, to_token=bob_id),),
)
```

Opt-out and purge:

```python
await memory.admin.set_opt_out(guild_id, user_id, True)
await memory.admin.purge_user(guild_id, user_id, dry_run=False)
```

Opt-out applies to both `observe` and recall.

## Typed vocabulary

Every closed set of values is a `StrEnum` exported from the package root —
no magic strings, no guessing. Enum members are plain strings, so they
compare equal to and serialize as their values:

```python
from icelake import (
    AliasSource,       # identity.register_alias(source=...)
    AttributionType,   # facts.remember(attribution=...)
    ChannelName,       # RecallQuery.channels / channels(...)
    CommandAction,     # classify_command results
    EntityKind,        # ProposedEntity(kind=...)
    FactCategory,      # facts.remember(category=...)
    FactHistoryKind,   # facts.history() entries
    FactScope,         # FactRecord.scope: USER vs SERVER facts
    HealthStatus,      # ops.health() component states
    IgnoreReason, RejectReason, ObserveStatus,  # observe receipts
    MemoryTier,        # FactRecord.tier, GuildStats.by_tier keys
    MeterPurpose,      # the library's own LLM call purposes
    MessageRole,       # LlmMessage.role
    NodeType, Polarity, RelationVerb,           # graph edges
    RecallWarning,     # recall / prompt_context warnings
    Scope,             # RecallQuery.scope (retrieval-side only)
    SourceRole,        # citation roles
    StorageBackend,    # config.storage.backend
)
```

Two vocabularies are intentionally open and accept plain strings alongside
the enum: `RelationVerb` (extraction may produce verbs outside the known
set; unknown verbs are polarity-neutral) and meter purposes (charge your own
LLM calls under your own names). Note `Scope` (retrieval) and `FactScope`
(storage) are different sets — don't use one where the other is expected.

## How extraction works

`observe` writes the message and enqueues it. A worker claims a batch per
guild + author (leases, so two processes will not extract the same person
twice). Short chatter is skipped with no LLM call. Otherwise the pipeline
mints roster tokens (`p0`, `p1`, `server`), asks the model for JSON, runs
quality gates, and stores what survives.

The model never sees Discord snowflakes. Identity fields may only use tokens
minted for that batch, anything else is dropped. Stored text uses display
names. The owner of a fact is the snowflake on `subject_id`, so a rename
adds an alias instead of moving the row.

Invalid JSON is repaired once, then dead-lettered. Contradictions invalidate
or supersede the old fact. Nothing is deleted. Profile digests regenerate
after `extraction.auto_consolidate_after_adds` new facts (default 5).

Recall does not call the LLM. Typical queries:

| Question | Call |
|---|---|
| what do you know about X | `recall(subject_ids=(x,))` |
| what does X think about Y | `graph.between(x, y)` |
| who likes movies | `graph.entity_stances("movies")` |
| people connected to X | `graph.neighbors(x, depth=2)` |

## API

```python
memory.observe(event)
memory.observe_many(events)
memory.flush(guild_id=...)
memory.register_bot_id(bot_user_id)   # never stored as a subject

memory.prompt_context(...)
memory.recall(RecallQuery(...))

memory.facts.remember / update / forget / reinforce / history / list_for_subject / search
memory.identity.resolve / register_alias / handle_member_rename / aliases_of
memory.graph.between / entity_stances / neighbors / relations_of / similar_users
memory.admin.set_opt_out / purge_user / export_guild / get_opt_out
memory.ops.run_pending / retry_dead_letters / meter_snapshot / health
memory.events.subscribe(BatchCompleted, handler)
memory.classify_command(text)
memory.regenerate_summaries(guild_id)
memory.stats(guild_id)
```

## discord.py

```python
# pip install icelake[discord]
from discord.ext import commands
from icelake import MemoryConfig
from icelake.integrations import setup_discord_memory

config = MemoryConfig(
    storage="sqlite:///bot-memory.db",
    llm="openai://$KEY@openrouter.ai/api/v1?model=google/gemini-3.7-flash",
)

class MyBot(commands.Bot):
    async def setup_hook(self) -> None:
        self.memory = await setup_discord_memory(self, config)
```

This wires `on_message` to `observe`, refreshes aliases on
`on_member_update`, and registers the bot's user id on ready. For a full
`/memory` group see [`examples/omni_style_bot.py`](examples/omni_style_bot.py).

## Configuration

```python
MemoryConfig(
    storage="sqlite:///memory.db",          # or mongodb://... with [mongo]
    llm="openai://$KEY@openrouter.ai/api/v1?model=...",
    embeddings="hashing",                   # default. see below
    # embeddings="local",
    # embeddings="openai://$KEY@openrouter.ai/api/v1?model=openai/text-embedding-3-small",
    batching={"batch_size_messages": 10, "max_age_seconds": 300},
    extraction={"auto_consolidate_after_adds": 5},  # 0 disables digests
    budgets={"guild_daily_prompt_tokens": 200_000},
    privacy={"store_raw_messages": True},
    workers={"enabled": True, "count": 2},
)
```

Useful LLM URL query params: `model`, `temperature=none` (omit sampling on
reasoning endpoints), `reasoning=low`, `max_tokens`,
`max_tokens_key=max_completion_tokens` (Azure),
`structured_outputs=json_object` if the endpoint cannot enforce `json_schema`.
A capability mismatch raises `LlmCapabilityError`.

`postgresql://` is recognized and rejected. There is no Postgres adapter yet.
Unknown config keys raise immediately.

### Embeddings

| Provider | Spec | Notes |
|---|---|---|
| Hashing (default) | `"hashing"` | Free, deterministic, no extra deps. Fine for tests. |
| Hosted | `"openai://$KEY@openrouter.ai/api/v1?model=openai/text-embedding-3-small"` | Use this for a real bot. |
| Local | `"local"` | `pip install icelake[local-embeddings]` |

Embeddings are used for semantic recall and for catching paraphrases so the
same fact is not stored twice. The default hashing embedder is n-gram feature
hashing, not a neural model. "Loves coding in Go" and "Enjoys programming in
Go" usually do not collide, so you get duplicate memories. Switch to hosted
or local embeddings for anything you actually run.

Changing embedder invalidates existing vectors. Re-embed or start a new
database.

## Workers

| Setup | Config |
|---|---|
| One process | defaults, workers run as background tasks |
| Bot and worker split | bot: `workers={"enabled": False}`, worker: same DB, loop on `await memory.ops.run_pending()` |
| Cron | workers off, call `ops.run_pending` from your scheduler |
| Several processes | share one database, keyed leases keep workers from overlapping |

## Custom backends

```python
from icelake import ChatLLM, DiscordMemory, Embedder, MemoryStore

memory = DiscordMemory(
    config,
    store=MyPostgresStore(),   # MemoryStore (+ optional .queue / .vectors)
    llm=MyLLM(),               # ChatLLM
    embedder=MyEmbedder(),     # Embedder
    clock=FakeClock(...),
)
```

The override kwargs are typed (`MemoryOverrides`): your editor will
autocomplete `store`, `queue`, `vectors`, `embedder`, `meter`, `llm`,
`small_llm`, `clock`, and `id_gen`, all checked against the port protocols
exported from the package root. Passing `llm=None` or `embedder=None`
explicitly disables that capability (degraded mode).

A new store has to pass `tests/integration/test_store_conformance.py`.

## Development

```bash
uv sync --group dev
uv run pytest tests/ -q --cov=icelake
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
```

Coverage floor is 90%. mypy is strict.

User-facing changes need a changelog fragment in `changelog.d/` — see
[CHANGELOG.md](CHANGELOG.md). Releases are cut from the **Release** workflow;
see [docs/RELEASE.md](docs/RELEASE.md).

## Status (v0.1.x)

- Storage: SQLite (default), MongoDB (`[mongo]`), in-memory (tests). Postgres
  is not implemented, `postgresql://` fails with a clear error.
- Default hashing embeddings are not good enough for production recall. See
  [Embeddings](#embeddings).
- Bad extraction JSON is repaired once, then dead-lettered. Retry with
  `ops.retry_dead_letters`.
- Caps and TTL drop weakest facts first (manual/CORE last). Budgets are
  per-process. Cross-process accounting needs store-backed counters.
- `similar_users` is capped Jaccard over entity adjacency.

## License

MIT
