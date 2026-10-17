# icelake

Persistent memory for Discord bots. It passively reads messages, extracts facts about your members, and hands you a labeled block to paste into your system prompt when the bot needs to reply.

Facts are stored against Discord user IDs, not names, so a rename never moves someone's memories onto the wrong person. Third-party claims ("alice called bob a hacker") attach to the person they are about.

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

from icelake import Audience, DiscordMemory, MemoryConfig, MessageEvent


async def main() -> None:
    memory = DiscordMemory(
        MemoryConfig(
            storage="sqlite:///memory.db",
            llm="openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=z-ai/glm-5.3-flash&reasoning=low",
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
            audience=Audience(channel_id="777", requester_id="100000000000000001"),
            asker_id="100000000000000001",
            text="what am I learning these days?",
            mentioned_ids=("200000000000000002",),
            thread_participant_ids=(),  # other humans in the reply chain / channel tail
        )
        print(ctx.injection_block)

        # Exact restatements can be cited without another provider call.
        # Paraphrases remain uncited unless you supply a CitationVerifier.
        reply = "You're learning Rust!"
        cited = await memory.cite(reply, ctx)
        reply = cited.text



asyncio.run(main())
```

`observe` returns immediately. Extraction runs in the background.
`prompt_context` builds a labeled block for the asker, anyone they
mentioned, thread participants, and the server. Mentions plus thread
participants turn on graph-hop recall and pair-intersect **every** combination
of those people (not just asker-other), so shared entities surface in one
call. Stick the block on your system prompt and generate a reply. The model
sees plain facts, never citation syntax. Then `memory.cite(reply, ctx)`
links exact restatements to registered evidence. Uncertain claims stay uncited.
Pass a verifier when you want citations for paraphrases.

### Citations: provenance with explicit support checks

`Citations` registers the closed set of sources a reply may cite. Code owns
URLs and source identities; the verifier can only select candidate IDs.
`memory.cite(reply, ctx)` defaults to exact restatements, with no provider calls.
Similarity alone cannot establish support: changing a person, negation, time,
or quantity can leave an embedding almost unchanged.

For paraphrases, inject a `CitationVerifier`. The supplied `LlmCitationVerifier`
uses your existing `ChatLLM` interface, a validated output schema, a 15-second
deadline, and at most one schema-repair call. It checks at most 12 claims and
three candidate sources per claim. `retrieval.citation_min_score` controls
candidate selection, never proof of support. Supply a metered LLM to account
for calls under the `citation` purpose.

```python
from icelake import Citations, LlmCitationVerifier

citations = Citations.from_prompt_context(ctx)
citations.add_source(
    "https://example.com/docs", title="Docs", excerpt="The source passage goes here."
)
# verifier_llm is your implementation of the ChatLLM port.
verifier = LlmCitationVerifier(verifier_llm, guild_id=guild_id)
cited = await memory.cite(model_output, citations, verifier=verifier)
```

A citation establishes provenance, not independent truth. Model-based verification
is fallible; missing, invalid, or failed verification leaves uncertain claims
uncited. Third-party reports must retain their attribution. Provider-supplied
character anchors can also be registered explicitly; they are trusted provenance
annotations and do not pass through the claim verifier.

Rendering validates registered sources and verbatim span offsets before inserting
`[[N]](<url>)`. `AttributedReply.claims` and `.sources` contain used-only evidence.
Pass `weave=False` for your own renderer. Footer numbers must use `Citations.citable`
to agree with inline labels. See [`examples/citations.py`](examples/citations.py).

Examples:

- [`examples/omni_style_bot.py`](examples/omni_style_bot.py) - full bot: learns from everyone, replies when pinged or replied to, `/memory` slash commands, name-in-prose lookups
- [`examples/ping_reply_bot.py`](examples/ping_reply_bot.py) - smaller ping-reply bot
- [`examples/citations.py`](examples/citations.py) - closed-set citations, weave vs footer, used-only rendering, no Discord or LLM
- [`examples/lifecycle.py`](examples/lifecycle.py) - facts, identity, admin, ops, events, recall
- [`examples/name_lookup_tool.py`](examples/name_lookup_tool.py) - "what do you know about X?" tool handler, no Discord or LLM
- [`examples/relationship_queries.py`](examples/relationship_queries.py) - graph queries with no Discord or LLM
- [`examples/e2e_simulation.py`](examples/e2e_simulation.py) - scripted-guild eval
- [`examples/bench_models.py`](examples/bench_models.py) - model matrix

The current recommended extractor is **`z-ai/glm-5.3-flash`**. See [Model benchmark](#model-benchmark) for the ranked table and how to add a new row.

## Model benchmark

One full run of [`examples/e2e_simulation.py`](examples/e2e_simulation.py) per model (suite A drain-mode + suite B worker-mode) against OpenRouter on 2026-08-29 and 2026-08-30. Hard checks are library guarantees. Expectations are model-decided extraction/reconcile/classify outcomes. Spend is the meter's provider-reported USD. List prices are OpenRouter prompt / completion per 1M tokens as of 2026-08-30.

`n=2` rows average the latest `aug30-mapped` run with the previous published fair run on comparable knobs. `n=1` is either a first fair run (correct `MODELS` knobs in [`examples/bench_models.py`](examples/bench_models.py)) or a holdover not in this matrix. Not averaged: `aug30-temp-none` GLM (wrong knobs), GPT-5 mini without `reasoning=minimal`, Luna with `reasoning=low`.

**Pick `z-ai/glm-5.3-flash` unless you have a reason not to.** It hit 47/47 this run and still sits around a cent. Gemini 3.7 Flash also hit 47/47, at ~8x the spend. Mistral Small 3.2 is the cheap quality sleeper (43/47, retired both contradictions, $0.0064) but takes ~9.5 min. GPT-5 mini is usable with `reasoning=minimal` + `temperature none`, but still leaves Omaha live. 4o-mini ranks high on cost/speed while failing reconcile.

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
Exp   = 100 * (expectations met) / (expectations total)          # suites A+B
Hard  = 100 if zero hard failures, else max(0, 100 - 30 * failures)
        # n>1: use mean failure count
Cost  = 100 * (1 - log10(spend / 0.007) / log10(0.30 / 0.007))   # clamp 0-100
Speed = 100 * (1 - log10(seconds / 45)  / log10(1400 / 45))      # clamp 0-100

Score = 0.50*Exp + 0.20*Hard + 0.20*Cost + 0.10*Speed
```

`$0.007` / `$0.30` and `45s` / `1400s` are this round's observed best/worst. Leave them unless a new run is clearly outside the band.

To add a model: run `uv run python examples/bench_models.py --models <id> --out bench_runs/<date>`. If it already has a fair run on the same knobs, average hard / expectations / spend / duration. Otherwise use the new run alone. Plug current OpenRouter in/out, compute Score, insert the row in rank order.

### What actually differed

The discriminating job is still **reconcile**, not first-pass extraction. Almost every model stored "lives in Omaha" and "loves Red Bull". This mapped run, only GLM 5.3 Flash, Gemini 3.7 Flash, Mistral Small 3.2, Gemini 3 Flash Preview, and Gemini 3.1 Flash Lite retired both when Alice moved to Seattle and quit the drink. GPT-5 mini and Luna retired Red Bull but left Omaha live. Gemini 2.5 Flash and GPT-4.1-mini did it on an earlier run and did not this time.

Name-binding is the other split: Bob saying "nolan's last name is gregory" must not become Bob's alias. GPT-5 nano, Gemini 3.1 Flash Lite, Gemini 3 Flash Preview, and MiniMax M3 failed that hard check.

| Model | What it did well | What it missed |
| --- | --- | --- |
| GLM 5.3 Flash | Perfect 47/47 this run (retired Omaha + Red Bull). n=2 still ~a cent | First run missed the charge-nurse merge |
| GPT-5.6 Luna | First fair with `reasoning=none`: 40/46, $0.012, 77s. Retired Red Bull | Omaha still live. Nursing versions piled up. Missed piano age-flush |
| Mistral Small 3.2 | First fair. Retired both contradictions at $0.0064 | ~9.5 min. Leftover "prefers yellow cans" next to "does not drink". No puppy merge / charge nurse / drums. Two reconcile JSON drops in the log |
| GPT-5 mini | First fair with `minimal` + `temp none`. 41/46, $0.024, 1:54. Retired Red Bull. Opening batch survived | Omaha still live. Nursing versions piled up. Do not average the earlier $0.11 disaster (no `minimal`. Dropped Alice's first batch) |
| GPT-4o-mini | n=2 cheapest-completed band, ~1 min. Mem0/Cognee's old default | Omaha and Red Bull still live both runs. No puppy / purple |
| Gemini 3.7 Flash | Perfect 47/47 this run (also retired both). Fast (~2 min) | n=2 spend ~8x GLM. First run missed a classify query |
| GPT-4.1-mini | Graphiti's default. Fast | Inconsistent: first run retired Omaha, this run left Omaha + Red Bull live |
| GPT-4.1-nano | First fair. Fastest OpenAI row (56s), $0.0056 | Omaha and Red Bull still live. Thin store. No puppy / purple / piano |
| gpt-oss-120b | First fair. GLM-level spend | 4:38. Omaha and Red Bull still live. One extraction JSON drop |
| GPT-5 nano | First fair (`json_object` no longer needed). Cheapest run ($0.0032) | **Hard fail:** Bob stating Nolan's surname became Bob's alias (`gregory`). Never extracted Red Bull. Added "the move went well" without retiring Omaha |
| Qwen3-30B instruct | First fair. $0.0054 | 5:12. Omaha and Red Bull still live. Two extraction JSON drops |
| Mercury 2 | Fastest by far (46s). Holdover | Same reconcile misses as the pack |
| Qwen3 32B | GLM-level spend both runs | ~14 min. Omaha and Red Bull still live |
| DeepSeek V4 Flash | This run retired Omaha (not Red Bull). Near-GLM spend | n=2 time pulled to ~15 min by the 1189s mapped run. Two extraction JSON drops. `0731` still worse |
| Gemini 3.1 Flash Lite | This run retired both contradictions. 44/47 expectations | **Hard fail** this run (same Gregory-alias miss). n=2 Hard 82/83. First run did not reconcile Omaha |
| Gemini 2.5 Flash | First run retired both | This run left Omaha live (Red Bull retired). n=2 quality dropped |
| MiniMax M3 | Fast and cheap. Holdover | **Hard fail:** Gregory alias on Bob. Omaha and Red Bull still live |
| Gemini 3 Flash Preview | First fair. Retired both. 43/47 | **Hard fail:** Gregory alias on Bob. Classify JSON dropped (`remember` / `query` missed) |
| DeepSeek 0731 | Cleaner than its first published run (33/46 this time) | Still no reconcile. Several JSON drops. ~13 min |
| Grok 4.3 | Holdover | Thin store. Expensive. Reconcile misses |
| Claude Haiku Latest | Holdover. Same expectation rate as old Luna | 28x GLM's spend. 401s and invalid JSON in the log |
| Gemma 4 31B | Holdover. Clean hard pass | 40 minutes. Missed name/Go/Omaha on first pass |
| GLM 4.5 Air | First fair (`json_object`, was 404) | Dropped Alice's opening batches (invalid JSON after one repair): no name / Go / Omaha / Red Bull / nursing. Later quit-Red-Bull facts attached to the wrong speaker |
| Qwen 3.7 flash | First fair (`json_object`, was 404) | Same opening-batch drops. **Hard fail:** `BatchCompleted` never fired. 6 facts |
| Hy4 preview | Holdover | Structured output kept failing. 5 rows, all curation probes. Failed `BatchCompleted` |

### Did not finish

| Model | In $/M | Out $/M | Why |
| --- | ---: | ---: | --- |
| `openai/gpt-5.6-sol` | $2.00 | $10.00 | HTTP 404: no endpoint accepted the requested knobs |
| `tencent/hy-mt2-1.8b` | $0.044 | $0.18 | Same 404 (translation model) |
| `tencent/hy-mt2-30b-a3b` | $0.074 | $0.30 | Same 404 (translation model) |

Mapped knobs live in `MODELS` in [`examples/bench_models.py`](examples/bench_models.py): mandatory-thinking models get the cheapest allowed effort (`low` / `minimal`), GPT-5.x omits temperature, models that 404 on `json_schema` use `structured_outputs=json_object`. Holdovers are `aug29-full-models`. Challengers averaged into n=2 are `aug30-challengers`.

## Graph explorer

A self-contained HTML canvas over the public API: users, entities, typed
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
(`--serve`) so the layout worker can run. Opening the file directly still works
with a one-shot layout.

## A reply turn

```python
ctx = await memory.prompt_context(
    guild_id=guild_id,
    asker_id=str(message.author.id),
    text=question,
    mentioned_ids=("alice_id", "bob_id"),
    thread_participant_ids=("carol_id",),
)

# ctx.injection_block looks like:
#
#   [MEMORY CONTEXT]
#
#   WHAT I KNOW ABOUT THE CURRENT ASKER
#   Facts about the asker ONLY:
#   - alice mains support in every ranked game she plays
#
#   REFERENCED USER: bob
#   Facts about bob ONLY. Do NOT attribute these to the asker.
#   - bob was called a hacker by alice during the ranked match
#
#   SERVER COMMUNITY FACTS
#   Community-wide traits:
#   - the community bonds over late night gaming sessions
#
#   (usage guidelines follow - no citation syntax anywhere)

reply = await generate(system_prompt + "\n\n" + ctx.injection_block, question)
cited = await memory.cite(reply, ctx)
await message.reply(cited.text, mention_author=False)
# cite weaves [[N]](<url>) at attributed claim spans. Unsupported claims
# stay uncited, and sources the reply never used never render.
```

`prompt_context` pair-intersects **every** combination of the asker, mentions,
and thread participants (not just asker-other) and turns on graph-hop recall
whenever anyone else is in the turn. The pairing helper is
`discovery_pairs(asker_id, mentioned_ids, thread_participant_ids)` (exported
from `icelake`) if you need the same pairs without a recall.

Identity is always `memory.identity.resolve(guild, name_or_mention_or_id)`.
Ambiguous matches never auto-pick. `identity.display_name` is the reverse
map so replies can attribute by the name the guild actually uses.

## "What do you know about X?" - names in prose

`prompt_context` is mention-and-thread keyed: it scopes sections from
`asker_id`, `mentioned_ids`, and `thread_participant_ids`. It never scans
the question text for names. Recall makes no LLM calls, by design. So when
someone asks "what do you know about klim?" and `klim` is typed rather than
@mentioned (and isn't in the thread), no klim section exists unless you
resolve the name yourself.

Resolve against current Discord membership, then retrieve through the audience
boundary. The live example asks for a mention when a name is ambiguous. For a
strict profile, retain only facts whose subject is the resolved member:

```python
result = await memory.recall(RecallQuery(
    guild_id=guild_id,
    subject_ids=(resolved_member_id,),
    audience=Audience(channel_id=reply_channel_id, requester_id=requester_id),
))
profile = [item.fact for item in result.facts
           if item.fact.subject_id == resolved_member_id]
```

Getting `name` out of the conversation is your side, and no tool runtime is
required. Use whatever your bot already has:

- **A slash command with a free-text option (zero LLM).** `/memory lookup
  klim` hands you the name directly. See `memory_lookup` in
  [`examples/omni_style_bot.py`](examples/omni_style_bot.py).
- **One structured-output call.** A tiny JSON router via
  `ChatRequest.response_schema` classifies the turn and extracts the name,
  no function-calling machinery. See `_route_name_lookup` in the same file.
- **Native function calling.** If your LLM client already speaks tools, hand
  it a schema and dispatch to the same handler.

Two details matter for accuracy:

- **Never guess on ambiguity.** `Resolution.ambiguous` means more than one
  member matches. Ask which one. The ladder ranks mention ID > username >
  real name > display name > nicknames, and only auto-resolves a clear
  winner.
- **Fetch strict when answering "about X".** Recall channels deliberately
  surface facts that *touch* a person, not just facts they are the subject
  of: a claim like "carol and bob argued over game night" is stored on
  carol, but bob sits on its incidence links (as actor or @mention), so
  `recall(subject_ids=(bob,))` returns it under bob. That is right for
  passive context (the injection block labels who each fact is about), but
  for a direct profile answer, filter the authorized results by `subject_id`.
  `facts.list_for_subject` is available to trusted tooling, but does not enforce
  channel permissions.

Trusted offline identity/profile inspection:
[`examples/name_lookup_tool.py`](examples/name_lookup_tool.py).
Wired into a live reply loop and a slash command:
[`examples/omni_style_bot.py`](examples/omni_style_bot.py).

## Names, relationships, commands

```python
resolution = await memory.identity.resolve(guild_id, "klim")
if resolution.ambiguous:
    ...  # ask which member. do not guess

edges = await memory.graph.between(guild_id, x_id, y_id)
shared = await memory.graph.shared(guild_id, x_id, y_id)
left, right = await memory.graph.shared_attributions(guild_id, x_id, y_id)
per_user = await memory.graph.shared_n(guild_id, (x_id, y_id, z_id))
stances = await memory.graph.entity_stances(guild_id, "movies")
audience = await memory.graph.users_for_entity(guild_id, "golf")
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

Every closed set of values is a `StrEnum` exported from the package root.
No magic strings, no guessing. Enum members are plain strings, so they
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
    RerankerProvider,  # retrieval.reranker
    Scope,  # RecallQuery.scope (retrieval-side only)
    SourceRole,  # citation roles
    StorageBackend,  # config.storage.backend
)
```

Two vocabularies are intentionally open and accept plain strings alongside
the enum: `RelationVerb` (extraction may produce verbs outside the known
set, and unknown verbs are polarity-neutral) and meter purposes (charge your
own LLM calls under your own names). Note `Scope` (retrieval) and `FactScope`
(storage) are different sets. Don't use one where the other is expected.

## How extraction works

`observe` writes the message and enqueues it. A worker claims a batch per
guild + author under an exclusive attempt lease. Expired work can be retried;
processing is at least once. Short chatter is skipped with no LLM call. Otherwise the pipeline
mints roster tokens (`p0`, `p1`, `server`), asks the model for JSON, runs
quality gates, and stores what survives.

The model never sees Discord snowflakes. Identity fields may only use tokens
minted for that batch. Anything else is dropped. Stored text uses display
names. The owner of a fact is the snowflake on `subject_id`, so a rename
adds an alias instead of moving the row.

Invalid JSON is repaired once, then dead-lettered. Contradictions invalidate
or supersede the old fact. Nothing is deleted. Profile digests regenerate
after `extraction.auto_consolidate_after_adds` new facts (default 5).

Recall does not call the LLM. Typical queries:

| Question | Call |
|---|---|
| what do you know about X | `identity.resolve("X")` then `facts.list_for_subject(x)` |
| what does X think about Y | `graph.between(x, y)` |
| what do X and Y share / disagree on | `graph.shared_attributions(x, y)` |
| what do X, Y, Z all share | `graph.shared_n((x, y, z))` |
| who likes movies | `graph.users_for_entity("movies")` (stances + mentions) or `graph.entity_stances("movies")` (edges) |
| people connected to X | `graph.neighbors(x, depth=2)` |
| everything about an entity in context | `prompt_context(..., entity_hint="golf")` |

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
memory.extract_now(event)  # observe + flush in one step
memory.register_bot_id(bot_user_id)  # never stored as a subject

memory.prompt_context(...)
memory.recall(RecallQuery(...))
memory.cite(reply_text, ctx)  # weave jump links into a finished reply

memory.facts.remember / get / get_all / update / forget / reinforce / history / list_for_subject / search
memory.identity.resolve / register_alias / handle_member_rename / aliases_of / display_name
memory.graph.between / relations_of / entity_stances / users_for_entity / neighbors / similar_users / shared / shared_attributions / shared_n
memory.admin.set_opt_out / get_opt_out / purge_user / export_guild / import_guild / backup
memory.ops.run_pending / recover / backfill_aliases / retry_dead_letters / meter_snapshot / health
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
    llm="openai://$KEY@openrouter.ai/api/v1?model=z-ai/glm-5.3-flash&reasoning=low",
)


class MyBot(commands.Bot):
    async def setup_hook(self) -> None:
        self.memory = await setup_discord_memory(self, config)
```

This wires `on_message` to `observe`, refreshes aliases on
`on_member_update`, registers the bot's user id on ready, and adds a
`/memory` command group (`me`, `remember`, `forget`). For a bigger
command surface see [`examples/omni_style_bot.py`](examples/omni_style_bot.py).

## Configuration

```python
MemoryConfig(
    storage="sqlite:///memory.db",  # or mongodb://... with [mongo]
    llm="openai://$KEY@openrouter.ai/api/v1?model=...",
    embeddings="hashing",  # default. see below
    # embeddings="local",
    # embeddings="openai://$KEY@openrouter.ai/api/v1?model=openai/text-embedding-3-small",
    batching={"batch_size_messages": 10, "max_age_seconds": 300},
    retrieval={
        "default_token_budget": 2400,
        # Optional L2 after hybrid RRF. Default is "none" (zero-LLM hot path).
        # "openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=voyageai/rerank-2.5-lite"
        # "local" needs icelake[local-embeddings]. Failure degrades to hybrid order.
        "reranker": "none",
        "hop_depth": 2,  # user -> entity -> other attributions
    },
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

Embeddings select semantic recall and reconciliation candidates. They cannot
prove that two claims mean the same thing; reconciliation makes that judgment. The default hashing embedder is n-gram feature
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
autocomplete `store`, `queue`, `vectors`, `embedder`, `reranker`, `meter`, `llm`,
`small_llm`, `clock`, and `id_gen`, all checked against the port protocols
exported from the package root. Passing `llm=None`, `embedder=None`, or
`reranker=None` explicitly disables that capability (degraded mode).

A new store has to pass `tests/integration/test_store_conformance.py`.

## Development

```bash
uv sync --group dev
uv run pytest tests/ -q --cov=icelake
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
```

Coverage floor is 90%. mypy is strict.

User-facing changes need a changelog fragment in `changelog.d/`. See
[CHANGELOG.md](CHANGELOG.md). Releases are cut from the **Release** workflow.
See [docs/RELEASE.md](docs/RELEASE.md).

## Status

- Storage: SQLite (default), MongoDB (`[mongo]`), in-memory (tests). Postgres
  is not implemented and `postgresql://` fails with a clear error.
- Default hashing embeddings are not good enough for production recall. See
  [Embeddings](#embeddings).
- Bad extraction JSON is repaired once, then dead-lettered. Retry with
  `ops.retry_dead_letters`.
- Caps and TTL drop weakest facts first (manual/CORE last). Budget counters
  are store-backed, so they are correct across N worker processes.
  Observability meters stay per-process.
- `similar_users` is capped Jaccard over entity adjacency.

## License

MIT


### Production reliability and migration

Authoritative fact, graph, input-receipt and projection-job writes commit together.
SQLite uses native transactions and `synchronous=FULL`. MongoDB now requires a
**replica set or a transaction-capable cluster and PyMongo 4.17+**; standalone
MongoDB can no longer silently claim successful multi-document commits. Custom
stores must implement real `transaction()` semantics and the projection-job
methods in `MemoryStore`. In-memory storage remains an ephemeral reference adapter.
Mongo uses the driver's [bound sessions and transaction support](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/transactions/).

Input delivery is at least once. A durable receipt for each message and processing
scope makes committed batches replay-safe, including re-batched input after an
acknowledgment failure. Receipts contain identifiers/hashes, not conversation text;
they persist indefinitely to prevent old backfills from being counted again.
Plan revisions are checked before mutations, and stale plans retry from current
memory. Event hooks are best-effort notifications after commit, not a durable
external event bus.

Vectors are a rebuildable projection. A transactional outbox survives crashes and
independent index outages; old revisions cannot overwrite newer vectors. Built-in
workers drain bounded recovery work, and external schedulers call
`await memory.ops.recover()`. Failures defer individual jobs so one unavailable
fact does not block the rest. `ops.health()` reports pending indexing as degraded.
An outage can temporarily reduce semantic recall while committed facts remain
available through other retrieval channels. Injected vector adapters must honor
`VectorItem.version` with an atomic conditional upsert.

```python
config = MemoryConfig(
    storage="sqlite:///memory.db",
    recovery={
        "max_attempts": 3,
        "retry_initial_seconds": 5,
        "retry_max_seconds": 300,
        "projection_batch_size": 32,
        "projection_timeout_seconds": 30,
    },
    access={"require_audience": True, "timeout_seconds": 3},
)
```

Batch retry state and jittered backoff survive restarts. Community-window retries
also stop after the configured attempt limit; the same operator retry API resets
their budget. Community roll-ups still use the configured recent-message window
and raw-message retention, so they are not an unlimited historical backfill. Exhausted batches remain
in dead letters; repair the cause, then use `ops.retry_dead_letters(guild_id)`.
Provider retries remain separately bounded by `llm.max_retries`; the worst-case
provider attempt count is the product of both limits. Cancellation retains the
input, and crashed leases are reclaimed by `ops.recover()`.

For explicit commands, supply `facts.remember(..., idempotency_key=interaction_id)`.
The same key returns the same fact without reinforcing it; conflicting reuse is
rejected. A retry cannot recreate a fact erased after the original command.
Without a key, repeated remember/reinforce calls intentionally count as new input.

### Channel permissions

`recall` and `prompt_context` now require an `Audience` by default. Missing context
returns no facts and an `AUDIENCE_REQUIRED` warning. Build it from trusted transport
metadata, never model output or unchecked request parameters. The reply audience
means **everyone who can read the destination**, not only the person asking.

The transport-neutral default restricts disclosure to the same source channel;
the embedding application must authenticate the caller and authorize the audience.
For Discord, `setup_discord_memory` installs `DiscordChannelAuthorizer`, which reads
fresh roles, channel overwrites and membership through REST, checks private-thread
membership, and abstains if history visibility cannot be proven for all readers.
Any explicit history-denying overwrite causes conservative abstention. A custom
`ChannelAuthorizer` can support broader, proven-safe cross-channel policies.
Requests are timeout-bounded and API failures deny access. This uses discord.py's
[channel permission calculation](https://discordpy.readthedocs.io/en/stable/api.html#discord.abc.GuildChannel.permissions_for).

```python
context = await memory.prompt_context(
    guild_id=guild_id,
    asker_id=user_id,
    audience=Audience(channel_id=reply_channel_id, requester_id=user_id),
    text=question,
)
```

All evidence channels must be allowed. Unknown provenance and mixed private/public
facts are excluded from protected retrieval. Filtering happens before external
reranking and is checked again before returning results. `cite()` rechecks the
context's fact revisions, consent and audience before emitting source links.
Multi-hop relation traversal is disabled for protected retrieval because its
aggregated edges are not audience-partitioned. Direct evidence-backed retrieval
remains available. Guild-wide summaries and mined name aliases are omitted from protected prompts;
channel-partitioned extraction and reconciliation prevent private context from
being incorporated into a public-channel fact.

`facts`, `graph`, `identity` and `admin` are trusted control-plane APIs, not
end-user authorization boundaries. Do not expose their raw results in public bot
replies. `/memory me` only includes channel-authorized facts plus self-authored
manual commands in its ephemeral response. Unscoped manual memories remain
available for that private inspection; to expose a manually created fact in a
channel, attach genuine `source_refs` from authorized source messages.
`access.require_audience=False` is an explicit trusted/offline compatibility mode,
not a production authorization policy. Reauthorize immediately before sending if
your application holds a generated reply for an extended period.

The runnable bot examples install live authorization and pass audience metadata
for both generated replies and profile commands. They omit raw channel-history
injection and unscoped graph displays. Name lookups use current Discord membership;
private memory aliases are not disclosed. Graph inspection remains demonstrated
separately in the trusted `relationship_queries.py` example.

### Operational backups and restore drills

`admin.export_guild()` is a portable content export, **not a disaster-recovery
backup**. SQLite operational snapshots include facts, aliases, incidence links,
relations, history, summaries, governance generations, opt-outs, queue contents,
leases, commit receipts, budgets, cursors, cached data, vectors and pending work.
The adapter uses SQLite's [online backup API](https://docs.python.org/3/library/sqlite3.html#sqlite3.Connection.backup).

```python
from pathlib import Path
from icelake.adapters.sqlite.backup import restore_backup

await memory.admin.backup(Path("backups/snapshot-2026-10-17"))  # new directory
# Restore BEFORE opening this new database with a client:
await restore_backup(
    Path("backups/snapshot-2026-10-17"),
    Path("recovered.sqlite"),
    max_bytes=10 * 1024**3,
)
```

Restore verifies schema, size, SHA-256 and SQLite integrity, resets interrupted
claims, and atomically publishes a **new** destination. Existing databases are
never overwritten. Checksums detect corruption; they do not authenticate an
untrusted archive. Independent queue/vector overrides require coordinated native
snapshots or an explicit rebuild plan; `admin.backup()` rejects that configuration
rather than silently producing an incomplete deployment backup.

Mongo deployments can use managed service backups or `MongoReplicaBackup` from
`icelake.adapters.mongo.backup`. It wraps official `mongodump --oplog` and
`mongorestore --oplogReplay --stopOnError`, with deadlines, archive size limits,
checksums and a private credentials file. **Its scope is the entire replica set,
including all databases**, because a consistent oplog dump cannot be limited to
one guild/database. Supply deployment URIs without a database path. Restore only
to an empty, offline deployment; a failed restore must be discarded and retried
into a fresh target. Native tools must be installed separately. Follow MongoDB's
[backup requirements](https://www.mongodb.com/docs/database-tools/mongodump/) and
[restore version requirements](https://www.mongodb.com/docs/database-tools/mongorestore/).

Keep application configuration, embedding-model identity, encryption keys and
secrets in your existing deployment backup system. Restore with the matching
embedding configuration. Reconcile erasures/opt-outs newer than the snapshot
before opening the restored system to users, and enforce retention/access control
on backups themselves. A snapshot cannot reconstruct changes made after it.

### Conversation evaluations

`evals/conversations.py` replays annotated transcripts through the configured
extractor, reconciliation, storage and retrieval. It accepts JSONL
`ConversationCase` records with provenance (`consented_redacted` or `synthetic`),
a permission reference, timestamped messages, and checkpointed probes. Gold labels
include subject IDs, expected text fragments and supporting message IDs; probes
also specify forbidden content, an audience and required abstention.

```python
from pathlib import Path
from evals.conversations import evaluate_case, load_cases, save_results
from icelake import StorageConfig, WorkersConfig

cases = load_cases(Path("redacted-conversations.jsonl"))
results = []
for case in cases:
    result = await evaluate_case(
        case,
        lambda clock: DiscordMemory(
            config.model_copy(update={"storage": StorageConfig(url="sqlite://:memory:"),
                                      "workers": WorkersConfig(enabled=False)}),
            clock=clock,
        ),
    )
    results.append(result)
save_results(Path("evaluation-results.json"), tuple(results))
```

Use a fresh store per case and the supplied replay clock. The harness rejects
empty datasets, invalid/future evidence labels, duplicate message IDs, live
background workers and incomplete ingestion. Reports include matched gold facts,
relevant/returned counts, source attribution, forbidden disclosures, abstention,
retrieval latency and provider usage, without raw transcript text. Fragment
matching and provenance checks are deterministic diagnostics, not semantic truth
judgments or a substitute for human annotation review.

The existing scripted golden suite is a regression suite; its model outputs do
not establish extraction accuracy. No real-conversation corpus or proprietary
assistant parity is claimed by the bundled synthetic harness tests. Supply
consented, redacted conversations and run the chosen production model to measure
actual quality and cost.
