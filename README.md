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
    memory = DiscordMemory(
        MemoryConfig(
            storage="sqlite:///memory.db",
            llm="openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=google/gemini-3.7-flash",
        )
    )
    async with memory:
        await memory.observe(
            MessageEvent(
                message_id="9001",
                guild_id="555",
                channel_id="777",
                author_id="100000000000000001",
                content="I've been learning Rust for about a year now!",
                created_at=datetime.now(UTC),
                author_display_name="alice",
            )
        )

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

- [`examples/omni_style_bot.py`](examples/omni_style_bot.py) - full bot: learns from everyone, replies when pinged or replied to, `/memory` slash commands, name-in-prose lookups
- [`examples/ping_reply_bot.py`](examples/ping_reply_bot.py) - smaller ping-reply bot
- [`examples/name_lookup_tool.py`](examples/name_lookup_tool.py) - "what do you know about X?" tool handler, no Discord or LLM
- [`examples/relationship_queries.py`](examples/relationship_queries.py) - graph queries with no Discord or LLM
- [`examples/e2e_simulation.py`](examples/e2e_simulation.py) - scripted-guild eval
- [`examples/bench_models.py`](examples/bench_models.py) - model matrix

The current recommended extractor is **`z-ai/glm-5.3-flash`**. See [Model benchmark](#model-benchmark) for the ranked table and how to add a new row.

## Model benchmark

One full run of [`examples/e2e_simulation.py`](examples/e2e_simulation.py) per model (suite A drain-mode + suite B worker-mode) against OpenRouter on 2026-08-29 and 2026-08-30. Hard checks are library guarantees; expectations are model-decided extraction/reconcile/classify outcomes. Spend is the meter's provider-reported USD. List prices are OpenRouter prompt / completion per 1M tokens as of 2026-08-30.

`n=2` rows average the latest `aug30-mapped` run with the previous published fair run on comparable knobs. `n=1` is either a first fair run (correct `MODELS` knobs in [`examples/bench_models.py`](examples/bench_models.py)) or a holdover not in this matrix. Not averaged: `aug30-temp-none` GLM (wrong knobs), GPT-5 mini without `reasoning=minimal`, Luna with `reasoning=low`.

**Pick `z-ai/glm-5.3-flash` unless you have a reason not to.** It hit 47/47 this run and still sits around a cent. Gemini 3.7 Flash also hit 47/47, at ~8× the spend. Mistral Small 3.2 is the cheap quality sleeper (43/47, retired both contradictions, $0.0064) but takes ~9.5 min. GPT-5 mini is usable with `reasoning=minimal` + `temperature none`, but still leaves Omaha live. 4o-mini ranks high on cost/speed while failing reconcile.

| Rank | Model | Score | n | Exp. | Hard | Spend | Time | In $/M | Out $/M |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `z-ai/glm-5.3-flash` | **94.0** | 2 | 46.5/47 | 83/83 | $0.0079 | 3:56 | $0.075 | $0.25 |
| 2 | `openai/gpt-5.6-luna` | 89.0 | 1 | 40/46 | 83/83 | $0.0121 | 1:17 | $0.20 | $1.20 |
| 3 | `mistralai/mistral-small-3.2-24b-instruct` | 88.3 | 1 | 43/47 | 83/83 | $0.0064 | 9:32 | $0.075 | $0.20 |
| 4 | `openai/gpt-5-mini` | 85.4 | 1 | 41/46 | 83/83 | $0.0236 | 1:54 | $0.25 | $2.00 |
| 5 | `openai/gpt-4o-mini` | 84.9 | 2 | 33/46 | 83/83 | $0.0065 | 1:02 | $0.15 | $0.60 |
| 6 | `google/gemini-3.7-flash` | 82.8 | 2 | 45/47 | 83/83 | $0.0672 | 2:06 | $0.75 | $3.75 |
| 7 | `openai/gpt-4.1-mini` | 82.1 | 2 | 36/47 | 83/83 | $0.0185 | 1:13 | $0.40 | $1.60 |
| 8 | `openai/gpt-4.1-nano` | 82.0 | 1 | 30/46 | 83/83 | $0.0056 | 0:56 | $0.10 | $0.40 |
| 9 | `openai/gpt-oss-120b` | 80.6 | 1 | 33/46 | 83/83 | $0.0061 | 4:38 | $0.037 | $0.17 |
| 10 | `openai/gpt-5-nano` | 79.8 | 1 | 34/46 | 82/83 | $0.0032 | 1:06 | $0.05 | $0.40 |
| 11 | `qwen/qwen3-30b-a3b-instruct-2507` | 79.2 | 1 | 32/46 | 83/83 | $0.0054 | 5:12 | $0.048 | $0.193 |
| 12 | `inception/mercury-2` | 78.3 | 1 | 32/46 | 83/83 | $0.0234 | 0:46 | $0.25 | $0.75 |
| 13 | `qwen/qwen3-32b` | 78.1 | 2 | 34/46 | 83/83 | $0.0074 | 14:05 | $0.08 | $0.28 |
| 14 | `deepseek/deepseek-v4-flash` | 78.0 | 2 | 36/47 | 83/83 | $0.0092 | 14:47 | $0.081 | $0.16 |
| 15 | `google/gemini-3.1-flash-lite` | 77.7 | 2 | 42/47 | 82/83 | $0.0593 | 2:11 | $0.25 | $1.50 |
| 16 | `google/gemini-2.5-flash` | 76.8 | 2 | 37/47 | 83/83 | $0.0471 | 1:56 | $0.30 | $2.50 |
| 17 | `minimax/minimax-m3` | 75.9 | 1 | 36/46 | 82/83 | $0.0184 | 1:32 | $0.30 | $1.20 |
| 18 | `google/gemini-3-flash-preview` | 75.4 | 1 | 43/47 | 82/83 | $0.0612 | 1:59 | $0.50 | $3.00 |
| 19 | `deepseek/deepseek-v4-flash-0731` | 72.7 | 2 | 30/46 | 83/83 | $0.0111 | 13:05 | $0.065 | $0.18 |
| 20 | `x-ai/grok-4.3` | 66.5 | 1 | 33/46 | 83/83 | $0.1012 | 4:20 | $1.25 | $2.50 |
| 21 | `~anthropic/claude-haiku-latest` | 66.4 | 1 | 37/46 | 83/83 | $0.2160 | 5:08 | $1.00 | $5.00 |
| 22 | `google/gemma-4-31b-it` | 64.9 | 1 | 28/45 | 83/83 | $0.0224 | 39:42 | $0.09 | $0.34 |
| 23 | `z-ai/glm-4.5-air` | 57.1 | 1 | 24/45 | 83/83 | $0.0624 | 11:15 | $0.13 | $0.85 |
| 24 | `qwen/qwen3.7-flash` | 55.9 | 1 | 17/45 | 82/83 | $0.0092 | 5:07 | $0.03 | $0.13 |
| 25 | `tencent/hy4-preview` | 23.3 | 1 | 8/44 | 82/83 | $0.2871 | 23:18 | $0.83 | $2.50 |

Score is `/100`. In/out are OpenRouter list prices, not the run. Expectation denominators differ slightly when a model never produced the fixture a later check needs. Spend under the $0.007 cost floor (4o-mini, 4.1-nano, gpt-oss-120b, gpt-5-nano, Qwen3-30B) clips the cost component to 100.

### Score

Fixed anchors, so a new row does not rescale the others:

```
Exp   = 100 × (expectations met) / (expectations total)          # suites A+B
Hard  = 100 if zero hard failures, else max(0, 100 − 30 × failures)
        # n>1: use mean failure count
Cost  = 100 × (1 − log10(spend / 0.007) / log10(0.30 / 0.007))   # clamp 0–100
Speed = 100 × (1 − log10(seconds / 45)  / log10(1400 / 45))      # clamp 0–100

Score = 0.50×Exp + 0.20×Hard + 0.20×Cost + 0.10×Speed
```

`$0.007` / `$0.30` and `45s` / `1400s` are this round's observed best/worst. Leave them unless a new run is clearly outside the band.

To add a model: run `uv run python examples/bench_models.py --models <id> --out bench_runs/<date>`. If it already has a fair run on the same knobs, average hard / expectations / spend / duration; otherwise use the new run alone. Plug current OpenRouter in/out, compute Score, insert the row in rank order.

### What actually differed

The discriminating job is still **reconcile**, not first-pass extraction. Almost every model stored "lives in Omaha" and "loves Red Bull". This mapped run, only GLM 5.3 Flash, Gemini 3.7 Flash, Mistral Small 3.2, Gemini 3 Flash Preview, and Gemini 3.1 Flash Lite retired both when Alice moved to Seattle and quit the drink. GPT-5 mini and Luna retired Red Bull but left Omaha live. Gemini 2.5 Flash and GPT-4.1-mini did it on an earlier run and did not this time.

Name-binding is the other split: Bob saying "nolan's last name is gregory" must not become Bob's alias. GPT-5 nano, Gemini 3.1 Flash Lite, Gemini 3 Flash Preview, and MiniMax M3 failed that hard check.

| Model | What it did well | What it missed |
| --- | --- | --- |
| GLM 5.3 Flash | Perfect 47/47 this run (retired Omaha + Red Bull). n=2 still ~a cent | First run missed the charge-nurse merge |
| GPT-5.6 Luna | First fair with `reasoning=none`: 40/46, $0.012, 77s. Retired Red Bull | Omaha still live; nursing versions piled up; missed piano age-flush |
| Mistral Small 3.2 | First fair. Retired both contradictions at $0.0064 | ~9.5 min; leftover "prefers yellow cans" next to "does not drink"; no puppy merge / charge nurse / drums. Two reconcile JSON drops in the log |
| GPT-5 mini | First fair with `minimal` + `temp none`. 41/46, $0.024, 1:54. Retired Red Bull. Opening batch survived | Omaha still live; nursing versions piled up. Do not average the earlier $0.11 disaster (no `minimal`; dropped Alice's first batch) |
| GPT-4o-mini | n=2 cheapest-completed band, ~1 min. Mem0/Cognee's old default | Omaha and Red Bull still live both runs; no puppy / purple |
| Gemini 3.7 Flash | Perfect 47/47 this run (also retired both). Fast (~2 min) | n=2 spend ~8× GLM. First run missed a classify query |
| GPT-4.1-mini | Graphiti's default. Fast | Inconsistent: first run retired Omaha, this run left Omaha + Red Bull live |
| GPT-4.1-nano | First fair. Fastest OpenAI row (56s), $0.0056 | Omaha and Red Bull still live; thin store; no puppy / purple / piano |
| gpt-oss-120b | First fair. GLM-level spend | 4:38; Omaha and Red Bull still live; one extraction JSON drop |
| GPT-5 nano | First fair (`json_object` no longer needed). Cheapest run ($0.0032) | **Hard fail:** Bob stating Nolan's surname became Bob's alias (`gregory`). Never extracted Red Bull; added "the move went well" without retiring Omaha |
| Qwen3-30B instruct | First fair. $0.0054 | 5:12; Omaha and Red Bull still live; two extraction JSON drops |
| Mercury 2 | Fastest by far (46s). Holdover | Same reconcile misses as the pack |
| Qwen3 32B | GLM-level spend both runs | ~14 min; Omaha and Red Bull still live |
| DeepSeek V4 Flash | This run retired Omaha (not Red Bull). Near-GLM spend | n=2 time pulled to ~15 min by the 1189s mapped run. Two extraction JSON drops. `0731` still worse |
| Gemini 3.1 Flash Lite | This run retired both contradictions; 44/47 expectations | **Hard fail** this run (same Gregory-alias miss). n=2 Hard 82/83. First run did not reconcile Omaha |
| Gemini 2.5 Flash | First run retired both | This run left Omaha live (Red Bull retired). n=2 quality dropped |
| MiniMax M3 | Fast and cheap. Holdover | **Hard fail:** Gregory alias on Bob. Omaha and Red Bull still live |
| Gemini 3 Flash Preview | First fair. Retired both; 43/47 | **Hard fail:** Gregory alias on Bob. Classify JSON dropped (`remember` / `query` missed) |
| DeepSeek 0731 | Cleaner than its first published run (33/46 this time) | Still no reconcile; several JSON drops; ~13 min |
| Grok 4.3 | Holdover | Thin store; expensive; reconcile misses |
| Claude Haiku Latest | Holdover. Same expectation rate as old Luna | 28× GLM's spend; 401s and invalid JSON in the log |
| Gemma 4 31B | Holdover. Clean hard pass | 40 minutes; missed name/Go/Omaha on first pass |
| GLM 4.5 Air | First fair (`json_object`, was 404) | Dropped Alice's opening batches (invalid JSON after one repair): no name / Go / Omaha / Red Bull / nursing. Later quit-Red-Bull facts attached to the wrong speaker |
| Qwen 3.7 flash | First fair (`json_object`, was 404) | Same opening-batch drops. **Hard fail:** `BatchCompleted` never fired. 6 facts |
| Hy4 preview | Holdover | Structured output kept failing. 5 rows, all curation probes. Failed `BatchCompleted` |

### Did not finish

| Model | In $/M | Out $/M | Why |
| --- | ---: | ---: | --- |
| `openai/gpt-5.6-sol` | $2.00 | $10.00 | HTTP 404: no endpoint accepted the requested knobs |
| `tencent/hy-mt2-1.8b` | $0.044 | $0.18 | Same 404 (translation model) |
| `tencent/hy-mt2-30b-a3b` | $0.074 | $0.30 | Same 404 (translation model) |

Mapped knobs live in `MODELS` in [`examples/bench_models.py`](examples/bench_models.py): mandatory-thinking models get the cheapest allowed effort (`low` / `minimal`); GPT-5.x omits temperature; models that 404 on `json_schema` use `structured_outputs=json_object`. Holdovers are `aug29-full-models`. Challengers averaged into n=2 are `aug30-challengers`.

## Graph explorer

A self-contained HTML canvas over the public API — users, entities, typed
relations, identity links (`entity is this member`), and server facts.
Incidence links (`dm_links`) are an index, not a relationship, and are not drawn.

```bash
python -m icelake.visualizer \
  --storage "mongodb://127.0.0.1:27017/icelake" \
  --guild YOUR_GUILD_ID \
  --out icelake-graph.html \
  --serve --open
```

`--center klim --depth 2` limits the canvas to that neighborhood (`server` /
`the server` works). `--list-guilds` prints guilds in storage. Serve over HTTP
(`--serve`) so the layout worker can run; opening the file directly still works
with a one-shot layout.

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

## "What do you know about X?" — names in prose

`prompt_context` is mention-keyed: it scopes sections from `asker_id`,
`mentioned_ids`, and reply targets. It never scans the question text for
names — recall makes no LLM calls, by design. So when someone asks "what do
you know about klim?" and `klim` is typed rather than @mentioned, no klim
section exists unless you resolve the name yourself.

The library side is two calls — resolve, then a strict subject fetch:

```python
resolution = await memory.identity.resolve(guild_id, name)
if resolution.ambiguous:
    ...  # ask which member. do not guess
if resolution.resolved is not None:
    page = await memory.facts.list_for_subject(
        guild_id, resolution.resolved.user_id, include_server=False
    )
```

Getting `name` out of the conversation is your side, and no tool runtime is
required — use whatever your bot already has:

- **A slash command with a free-text option (zero LLM).** `/memory lookup
  klim` hands you the name directly. See `memory_lookup` in
  [`examples/omni_style_bot.py`](examples/omni_style_bot.py).
- **One structured-output call.** A tiny JSON router via
  `ChatRequest.response_schema` classifies the turn and extracts the name —
  no function-calling machinery. See `_route_name_lookup` in the same file.
- **Native function calling.** If your LLM client already speaks tools, hand
  it a schema and dispatch to the same handler.

Two details matter for accuracy:

- **Never guess on ambiguity.** `Resolution.ambiguous` means more than one
  member matches; ask which one. The ladder ranks mention ID > username >
  real name > display name > nicknames, and only auto-resolves a clear
  winner.
- **Fetch strict when answering "about X".** Recall channels deliberately
  surface facts that *touch* a person, not just facts they are the subject
  of: a claim like "carol and bob argued over game night" is stored on
  carol, but bob sits on its incidence links (as actor or @mention), so
  `recall(subject_ids=(bob,))` returns it under bob. That is right for
  passive context (the injection block labels who each fact is about), but
  for a direct profile answer `list_for_subject` returns only facts where X
  is the subject, so the model cannot attribute someone else's claim to X.

Full handler with mention stripping, ambiguity wording, and an unknown-name
fallback: [`examples/name_lookup_tool.py`](examples/name_lookup_tool.py).
Wired into a live reply loop and a slash command:
[`examples/omni_style_bot.py`](examples/omni_style_bot.py).

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
        guild_id=guild_id,
        subject_id=user_id,
        text=command.target_text,
        actor_id=user_id,
    )
```

Third-party facts can carry a relation:

```python
from icelake import ProposedRelation, RelationVerb

await memory.facts.remember(
    guild_id=guild_id,
    subject_id=bob_id,
    text="carol called bob a sore loser during game night",
    actor_id=carol_id,
    speaker_id=carol_id,
    relations=(
        ProposedRelation(verb=RelationVerb.CALLED_OUT, from_token=carol_id, to_token=bob_id),
    ),
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
    AliasSource,  # identity.register_alias(source=...)
    AttributionType,  # facts.remember(attribution=...)
    ChannelName,  # RecallQuery.channels / channels(...)
    CommandAction,  # classify_command results
    EntityKind,  # ProposedEntity(kind=...)
    FactCategory,  # facts.remember(category=...)
    FactHistoryKind,  # facts.history() entries
    FactScope,  # FactRecord.scope: USER vs SERVER facts
    HealthStatus,  # ops.health() component states
    IgnoreReason,
    RejectReason,
    ObserveStatus,  # observe receipts
    MemoryTier,  # FactRecord.tier, GuildStats.by_tier keys
    MeterPurpose,  # the library's own LLM call purposes
    MessageRole,  # LlmMessage.role
    NodeType,
    Polarity,
    RelationVerb,  # graph edges
    RecallWarning,  # recall / prompt_context warnings
    Scope,  # RecallQuery.scope (retrieval-side only)
    SourceRole,  # citation roles
    StorageBackend,  # config.storage.backend
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
| what do you know about X | `identity.resolve("X")` → `facts.list_for_subject(x)` |
| what does X think about Y | `graph.between(x, y)` |
| who likes movies | `graph.entity_stances("movies")` |
| people connected to X | `graph.neighbors(x, depth=2)` |

## What gets stored

Everything hangs off facts. A fact is one sentence about a user or the
server, stored in `dm_facts` against Discord IDs. The other tables exist to
find facts and connect them.

| Table | What it is | Why it exists |
|---|---|---|
| `dm_facts` | The memory itself, one row per fact | Read by everything |
| `dm_vectors` | One embedding per fact | Semantic recall: "what has he been up to creatively" finds the SoundCloud fact |
| `dm_entities` | Named things that are not Discord users (places, concepts, orgs, off-server people) | 30 facts about "the HOA" cluster on one node instead of scattered text |
| `dm_entity_aliases` | Surface name to entity slug | "Koji Sushi" and "koji-sushi" resolve to the same node |
| `dm_links` | One row per fact per node it touches | "Facts about the asker" is an index lookup, not a text scan |
| `dm_relations` | Typed edges between nodes, like `x -friend_of-> y` | Powers `graph.between`, `entity_stances`, `neighbors` |

Facts have a `subject_id` (the user the fact is about) or a null subject
with `scope="server"` for guild-wide facts. Vectors, links, and relations
all reference fact IDs, so purging a user cascades through every table.

Links vs relations is the part people mix up. A link only says "this fact
touches this user or entity". It carries no meaning by itself. A relation
says "these two nodes have a typed relationship" and carries a verb,
polarity, weight, and the fact IDs that support it. Links come from every
stored fact. Relations only exist when extraction or a manual
`facts.remember` call found an actual relationship.

Entities vs aliases: the entity is the node, the alias is how you find it.
`dm_entities` holds one row per thing (slug, display name, kind, and
`linked_user_id` when the entity turns out to be a guild member).
`dm_entity_aliases` maps names to that slug, so recall can go from a word in
the message to every fact about the thing.

Recall runs several channels over these tables (vector, keyword, links,
baseline, entity, graph hop) and fuses the results. Facts are the payload.
The rest are indexes over them.

## API

```python
memory.observe(event)
memory.observe_many(events)
memory.flush(guild_id=...)
memory.register_bot_id(bot_user_id)  # never stored as a subject

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
    storage="sqlite:///memory.db",  # or mongodb://... with [mongo]
    llm="openai://$KEY@openrouter.ai/api/v1?model=...",
    embeddings="hashing",  # default. see below
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
    store=MyPostgresStore(),  # MemoryStore (+ optional .queue / .vectors)
    llm=MyLLM(),  # ChatLLM
    embedder=MyEmbedder(),  # Embedder
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
