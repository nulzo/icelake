# External Memory Systems Review — mem0, graphiti, CringeDiscordBot, and 2026 Field Guidance

Date: 2026-08-27. Sources: local source review of `~/Github/mem0` (OSS v3),
`~/Github/graphiti` (graphiti_core), and `~/Github/CringeDiscordBot` (our production
predecessor), plus current 2026 production-memory literature.
Purpose: concrete, prioritized improvements for `discord-memory`, mapped to our files.

---

## TL;DR

- **mem0 OSS v3 went append-only**: one extraction LLM call, MD5-hash dedup, no LLM
reconciliation (their v2 ADD/UPDATE/DELETE/NOOP decide-call is dead code in the repo).
They trade contradiction-correctness for latency and let hybrid retrieval rank over
the pile. **We should not follow** — our e2e sim proved contradictions pile up
without reconciliation, and at our scale (per-user fact counts in the tens, not
thousands) reconcile is affordable.
- **graphiti is the accuracy reference**: staged resolution (deterministic → fuzzy →
LLM only for residuals), bi-temporal soft invalidation (`valid_at`/`invalid_at`/
`expired_at`), hybrid search fused with RRF. Architecturally, **we are already
graphiti-lite** — reconcile + soft invalidation + provenance. Their best ideas are
adoptable without Neo4j.
- **Vendor benchmarks are not load-bearing.** Independent reruns diverge from vendor
numbers by up to 45 points (mem0 LongMemEval: 94.4 vendor vs 49.0 independent).
The decision-relevant numbers are latency and tokens/retrieval, not accuracy
  headlines. We should build our own eval, not chase leaderboard configs.
- **CringeDiscordBot validates our core design.** Its production-hardened choices —
  batched extraction behind locks, pre-LLM quality gates, deterministic upsert
  (exact + ≥0.92 embedding → reinforce, no LLM), lifecycle tiers with TTL, hybrid
  RRF retrieval, weighted alias index with conflict scoring — are the same shapes we
  have or recommend below. Its known debt (god-service, LLM UPDATE/DELETE keyed on
  opaque IDs from partial context, hot-path lifecycle) is what our port/adapter
  split and reconcile design already avoid.

---



## 1. mem0 OSS v3 — how it actually works

Core path: `mem0/memory/main.py` (~3.9k lines). Graph memory was **removed from OSS
in v3** (commit `a488e190`); replaced by a spaCy entity vector store used only for
retrieval boosts.

### Write path (`add()`, `infer=True`) — exactly 1 LLM call

1. Pull last 10 session messages from SQLite (ring buffer for coreference).
2. Embed the flattened conversation; vector-search existing memories `top_k=10`
  (**no score threshold** — neighbors are LLM context, not a gate).
3. One call with `ADDITIVE_EXTRACTION_PROMPT` (`mem0/configs/prompts.py:468-944`):
  ADD-only, rich self-contained facts (15–80 words), temporal grounding
   ("Observation Date" vs "Current Date"), "when in doubt, extract".
4. Batch-embed extracted texts → **MD5 exact-text dedup** against neighbors and
  in-batch (`main.py:1006-1024`) → batch insert → spaCy entity linking
   (merge on exact match or cosine ≥ 0.95).

Notable: existing-memory UUIDs are remapped to integers before the prompt
(anti-hallucination, `main.py:933-953`) — but the `linked_memory_ids` the prompt
requests are **never consumed**. Shipped-unused prompt fields; we should not copy
that mistake.

### CRUD / reconciliation

None on the write path. Contradictions become new rows; retrieval is expected to
rank current over stale. Manual `update()`/`delete()` exist and write SQLite
history events (`ADD`/`UPDATE`/`DELETE`; `NONE` is never persisted). The v2
two-call pipeline (extract → batched ADD/UPDATE/DELETE/NOOP decide with
integer-remapped neighbor ids) survives only as unused prompt strings
(`prompts.py:176-460`).

### Retrieval (`search()`)

1. Lemmatize query (BM25) + spaCy entity extraction + embed.
2. **Over-fetch** `max(4·limit, 60)` semantic candidates.
3. Optional BM25 keyword search (sigmoid-normalized), entity boost
  (`similarity·0.5`, hub entities down-weighted quadratically).
4. **Semantic threshold gate first** (default 0.1), then blended score — BM25 can
  never resurrect semantically irrelevant junk.
5. Optional reranker (cohere/hf/llm/sentence-transformer) — off by default.



### Performance posture

Batch embed/insert/history with per-item fallback; entity boost capped at 4 workers;
`delete_all` in batches of 1000; session messages capped at 10; telemetry sampled at
10% on hot paths. **No LLM token metering in OSS core** (we now have this — ahead of
them here).

---



## 2. graphiti — how it actually works

Core: `graphiti_core/graphiti.py`, `node_operations.py`, `edge_operations.py`,
`search/`. Episode-centric; ingest is **sequential per partition** (their docs warn
never to parallelize episodes for one graph — matches our per-subject lease design).

### Write path (`add_episode`) — 2 LLM calls floor, then small-model residuals

1. Load previous 10 episodes as context (`RELEVANT_SCHEMA_LIMIT`).
2. **Extract nodes** (1 medium-model call, structured Pydantic output).
3. **Resolve nodes** in stages (`node_operations.py:627-708`):
  - embed name → cosine top-15 candidates, min score 0.6;
  - **deterministic**: exact normalized name → match;
  - **fuzzy**: MinHash+LSH, Jaccard ≥ 0.9 with entropy gate → match;
  - **LLM only for residuals**: one batched `dedupe_nodes` call picks
  `duplicate_candidate_id` or -1=new. Never LLM-scans the whole graph.
4. **Extract edges** (1 call): `(source, relation, target, fact, valid_at,
  invalid_at)`; names remapped to canonical UUIDs post-resolution.
5. **Resolve edges** per edge (`edge_operations.py:623-847`): exact
  endpoints+fact → append provenance episode, done (no LLM); else one
   small-model `resolve_edge` call returning `duplicate_facts` +
   `contradicted_facts`.
6. **Invalidate contradictions softly**: older overlapping facts get
  `invalid_at = new.valid_at`, `expired_at = now`. Nothing is deleted; expired
   facts remain auditable and are excluded by validity filters at read time.



### Retrieval

Parallel edge/node/episode/community searches; per-scope hybrid = BM25 ∪ cosine
(min 0.6) ∪ BFS (depth ≤ 3); candidates at `2×limit`; then one reranker: **RRF**
(default), MMR (λ=0.5), cross-encoder, node-distance, or episode-mentions
(count of supporting episodes — a direct use of provenance as a ranking signal).
Context composition emits facts with explicit validity language ("valid from X to
Y" vs "Present").

### Performance posture

`semaphore_gather` everywhere (default limit 20); SQLite LLM response cache keyed by
model+messages MD5; embedding batching; summary LLM skipped when concatenation fits
(`≤ 2×MAX_SUMMARY_CHARS`); bulk ingest path with combined extraction; small model
for dedupe/timestamps/attributes, medium for extraction.

---

## 3. CringeDiscordBot — the production predecessor

MongoDB (`memories` collection) + local sentence-transformers embeddings +
optional Atlas `$vectorSearch`. Centered on a ~3.9k-line `MemoryService`
god-object with recent partial extractions. Old and clunky in places, but every
quirk below is scar tissue from real production traffic — worth mining.

### Write path

- **Batched, gated, locked extraction** — never per-message. Fires when
  `pending ≥ batch_size` (10) **or oldest pending ≥ 300s**, rate-limited by a
  60s min-interval unless backlog ≥ 2× batch, under a distributed DB lock
  (`memory.py:418-527`). Server/culture pass every 25 guild messages.
- **Pre-LLM filters**: command/URL/short-text drops, in-batch exact dedupe,
  optional embedding clustering at 0.85 (keep longest), min-chars
  worth-extracting gate (`memory.py:553-600`).
- **One LLM call per batch** emitting `add|update|delete|noop` ops against a
  hybrid-retrieved subset of existing memories (~top-32 + core anchors), strict
  JSON schema, `json_object` retry on parse failure.
- `memory_start_at` per guild fences off historical ingest.

### Memory model & lifecycle

Fact document: text + normalized text + embedding, category, confidence,
`entity_type` (user/server/relationship), `related_users`/`tags`, **source
message snapshots** (stable citations), attribution (self/third_party/inferred/
manual), `occurrences`/`last_reinforced`, **lifecycle tier with TTL**
(short ~7d / mid ~45d / long ~180d / core = never), soft delete, and hard caps
(300 facts/user, 500/guild) (`models/memory.py:272-332`,
`memory.py:3616-3693`).

### CRUD / reconciliation

- **Deterministic upsert before any LLM judgment**: exact normalized match, or
  embedding ≥ 0.92, or ≥ 0.85 semantic → **reinforce in place** (bump
  occurrences/confidence, merge sources); can reinstate a soft-deleted exact
  match (`memory.py:1756-1855`).
- LLM UPDATE/DELETE require echoing a `memory_id` from the retrieved registry
  subset — powerful but **fragile when the right memory wasn't retrieved into
  context** (their documented debt).
- Batch maintenance: deterministic duplicate consolidation (0.88 clusters),
  optional LLM smart-merge, admin `/memory consolidate`.

### Identity

Weighted alias index (username, display name, "my name is…" regexes,
`Name (handle)` patterns, memory tags); exact-then-prefix resolution;
**conflict scoring keeps the highest `(source_rank, weight)` winner per alias**;
third-party name guards ("Klim's brother Ivan" must not alias Ivan onto Klim)
(`entity_identity.py:58-126,302-404`). This is the direct ancestor of our
`identity/aliases.py` — and confirms the design.

### Retrieval & reply-time use

- **Intent-planned RAG**: a retrieval plan (scope, focus users, recall mode)
  fans out to parallel channels (vector, text, tag, entity, graph, cross-user),
  **RRF fusion**, then hybrid rerank weighted semantic 0.55 / BM25 0.30 /
  entity 0.15 with occurrence/confidence boosts (`memory_retrieval.py:27-101`).
- **Cite-on-use `[mem:N]` aliases** mapped to real ids; post-generation expands
  citations to Discord jump links. Pool ≡ prompt symmetry: only facts shown to
  the model are citeable (`memory_context.py:111-137`).
- Load specs (none / profile / retrieval-only / hybrid) decided by an intent
  classifier, with token budgets — not full-profile dumps on every question.

### What to avoid (their documented debt)

- The ~4k-line god-service (we already split ports/adapters — keep it that way).
- LLM UPDATE/DELETE keyed on opaque Mongo ids from a **partial** retrieved
  context — if the target memory wasn't retrieved, the op silently no-ops.
- Local vector "search" over an arbitrary unsorted ~500-doc slice — a scale
  landmine they patched with Atlas. (Our SQLite brute-force is honest about the
  same limit; the pgvector swap path stays open per PLAN.md D1.)
- Lifecycle/cleanup on the retrieval hot path (they're throttling it
  retroactively); over-rich per-fact analytics schema (sentiment/tone/topics on
  every fact — token-heavy, rarely read).

---

## 4. 2026 field guidance (production literature)

Consensus across current production references:

- **Tiered memory** (working/episodic/semantic/governance) with distinct stores and
policies; don't collapse into one store.
- **Hybrid retrieval is table stakes**: dense ∪ sparse(BM25), calibrated weights;
recency decay in ranking: `final = α·relevance + (1−α)·recency`.
- **Write-time dedup by semantics, not string matching** — but pair it with an
explicit lifecycle (update/invalidate), or you get mem0's contradiction pile.
- **Reflection/consolidation on a schedule** (every 25–50 tasks or a token budget),
not per step.
- **Provenance and observability**: episode IDs on every fact, log every retrieval
decision; right-to-erasure for user data (we have `purge_user`).
- **Ledger thinking** (TARL, arXiv 2608.03699): accepted / pending / rejected
ledgers beat binary write/hold; conflicting evidence is preserved, not erased.
- Benchmark on your own traffic: vendor LoCoMo/LongMemEval numbers are
methodology-contested (the mem0↔Zep dispute is the canonical example).

---



## 5. Where we stand


| Capability            | mem0 v3               | graphiti                 | cringe (predecessor)           | **us today**           |
| --------------------- | --------------------- | ------------------------ | ------------------------------ | ---------------------- |
| Extraction            | 1 call, additive      | 2+ calls, staged         | 1 call per gated batch         | 1 call per batch ✓     |
| Reconciliation        | none (append-only)    | staged, LLM on residuals | deterministic upsert + LLM ops | LLM per collision      |
| Exact-dup handling    | MD5 hash skip         | deterministic, no LLM    | deterministic reinforce        | **LLM call** ✗         |
| High-sim reinforce    | no                    | no                       | ≥0.92, no LLM                  | no ✗                   |
| Invalidation          | manual only           | bi-temporal soft         | soft delete + TTL tiers        | soft + history ✓       |
| Provenance            | history events        | episodes[] on edges      | source snapshots + [mem:N]     | source_refs ✓          |
| Retrieval             | hybrid + entity boost | hybrid + RRF + BFS       | hybrid RRF + intent plan       | vector only ✗          |
| Recency in ranking    | no                    | via recipes              | occurrence/confidence boosts   | no ✗                   |
| Batch age trigger     | n/a                   | n/a                      | size **or** 300s age           | size only ✗            |
| LLM cost metering     | no                    | partial (cache)          | interaction metering           | **yes** ✓ (just wired) |
| Concurrency control   | worker pools          | semaphore(20)            | DB batch locks                 | per-subject leases ✓   |
| Identity/aliases      | no                    | node resolution          | weighted alias index           | alias mining ✓         |


We are graphiti-lite with mem0's single-call extraction, standing on
CringeDiscordBot's validated batching/gating/identity patterns. The gaps are
concentrated in **retrieval** (vector-only, no recency) and **reconcile
economics** (LLM per collision, including exact duplicates and near-certain
reinforces that need no judgment).

---



## 6. Recommendations

> **Status (implemented):** P0-1, P0-2, P1-1, P1-2, P1-3, P1-4 shipped — the live
> e2e sim passes 45/45 with 11 LLM calls (~$0.011) per full run. On review, three
> items were already built and needed no work: **P0-3** (FTS5 `dm_facts_fts` +
> keyword channel + RRF + hybrid rerank existed; we wired the configured-but-dead
> `weight_strength`/`weight_entity` components instead), **P0-4** (recency ships
> via the Ebbinghaus retention term in `strength_signal`, now fed into rerank),
> and **P1-5** (`due_batch_keys` already fires on `size OR oldest ≥ max_age_seconds`
> in all three queue backends). P2 items remain design-gated.

> **Field findings from the scaled e2e suite (2026-08-27).** Expanding the sim to
> ~110 public-API checks (noise/styles/pollution/governance/retrieval/workers)
> surfaced five real defects, all fixed and covered by the suite:
>
> 1. `gates.QUESTION_START` treated any leading auxiliary as a question, rejecting
>    legitimate subject-less facts ("Is allergic to shellfish."). Now requires
>    subject inversion ("is he", "does it") or a wh-word.
> 2. `FactSupersededEvent` was exported but never published — the pipeline now
>    emits it on both the UPDATE (supersede) and INVALIDATE paths.
> 3. `commit_supersede` never set `valid_until`, so superseded facts looked valid
>    forever to `as_of` time-travel queries. Now set on transition.
> 4. Category-gated collisions missed contradictions the model filed under
>    different categories ("loves Red Bull" → `preferences` vs "quit drinking Red
>    Bull" → `general`; both stayed active). Candidates with state-change phrasing
>    (`quit`, `no longer`, `moved`, `promoted`, …) now bypass the category gate and
>    face reconcile against any neighbor above the 0.35 floor.
> 5. `near_duplicate_threshold` 0.92 let deterministic reinforce swallow genuine
>    refinements ("promoted to charge nurse" reinforced "works as a nurse" without
>    applying the update). Raised to 0.96 — true paraphrases only.
>
> Two model-behavior limitations remain, by design visible in the sim as WEAK
> expectations rather than hard failures: (a) gemini-3.7-flash almost never
> re-emits facts already shown in EXISTING RELEVANT MEMORIES, so observe-path
> reinforcement rarely fires despite the prompt's explicit re-emit instruction —
> if this matters at scale, the fix is shadow reinforcement (when a batch yields
> no candidates, embed the raw messages against the subject's existing facts and
> reinforce near-identical pairs; embeddings only, no LLM), deferred as P2;
> (b) extraction recall on small batches varies run to run (a 3-message batch
> with durable content occasionally yields zero operations).
>
> **Field findings, round 2 (2026-08-27, gap-closure suite).** Nine new probes
> (multi-guild isolation, budget binding, response cache, lifecycle, strength
> ranking, token-budget trimming, channel restriction, dead letters, age-trigger
> flush) surfaced three more real defects, all fixed and covered:
>
> 6. **Budgets never bound.** `InMemoryMeter.charge_guild` had zero callers, so
>    `check_budget` always read empty guild counters and the degradation ladder
>    was dead code. `ChatRequest` now carries optional `guild_id` attribution
>    and `MeteredLLM` charges it; extraction/reconcile/summarize pass theirs.
> 7. **SQLite `claim_batch` double-processed in-flight batches.** After flipping
>    pending rows, it re-SELECTed *all* rows claimed by the same owner — so a
>    concurrent worker (same process owner) re-read a batch mid-extraction and
>    committed duplicates. Now `UPDATE … RETURNING` yields exactly the rows each
>    call flipped. (Mongo and in-memory were already correct.)
> 8. **`close(drain=True)` stranded in-flight batches** (workers cancelled
>    mid-commit, claims held until lease expiry) **and completed batches left
>    key leases to linger 120s**, stalling restarted processes. `close` now
>    gathers workers gracefully (cancel only on timeout) and `process_key`
>    releases the key lease on completion via the new `release_key` port method.
>
> Also added `GuildStats.in_flight_messages` (claimed-but-uncommitted), which
> makes "has memory caught up?" publicly observable — pending alone is blind to
> the extraction window.
>
> One design note, not a defect: the batch age trigger reads the *sender's*
> `created_at`. Discord timestamps are server-issued and trustworthy, so this is
> fine for the target deployment; a hostile/clock-skewed source could starve or
> fast-flush its own batches. If the library ever ingests untrusted timestamps,
> age on enqueue time instead.

> **Field findings, round 3 (2026-08-28, cross-model bench review).** The 9-model
> benchmark surfaced two structural defects and one harness classification leak:
>
> 9. **`require_parameters` 404s hard-failed models.** OpenRouter's constraint
>    excludes endpoints lacking full parameter support; when every healthy
>    endpoint is excluded it returns 404 at attempt 0 (never executed — safe to
>    replay). Provider quirks now live in an `OpenRouterLLM` subclass of the
>    generic OpenAI-compat client (composition root selects by host), and a 404
>    on a constrained structured request degrades once to unconstrained routing
>    while keeping json_schema. The base client carries no provider sniffing.
> 10. **Curation path published no events.** `facts.remember/update/forget/
>    reinforce` bypassed the event bus entirely — only pipeline commits fired
>    `FactCommitted`/`FactSupersededEvent`. The curation API now publishes both
>    (update = refine, old==new id; forget = retire, new=None), so bots get one
>    deterministic hook surface regardless of write path.
> 11. **Harness leak: model quality masquerading as library guarantees.** Three
>    "hard" sim checks (event-bus fired, pagination volume) implicitly required
>    strong extraction recall, so weak models exit-1'd on checks whose mechanism
>    was fine. The probes now drive the public curation API deterministically;
>    model-dependent outcomes stay in the `expect` (soft) tier.
> 12. **Repair feedback never showed the schema.** `complete_structured`'s repair
>    turn carried only the pydantic error ("extra inputs are not permitted") —
>    which says what's wrong, not what shape is expected. On endpoints *with*
>    constrained decoding this never matters (the schema is enforced wire-side);
>    on endpoints without it (what you get after a `require_parameters` 404
>    degrade), the model freestyles a plausible envelope — gpt-5.6-luna emitted
>    `{"facts": [...]}`, then "fixed" it to `{"memories": [...]}` — and every
>    batch was dropped (1 extracted fact in a full sim run). Embedding
>    `json.dumps(schema)` in the repair turn makes the same model emit perfect
>    schema-conformant JSON. This is the instructor/library-standard reask
>    pattern, and it only costs tokens when a repair actually happens.
>
> Residual latency note: after a 404 degrade, every structured call still pays
> one wasted constrained round-trip before falling back. A process-lifetime
> capability latch (stop sending `require_parameters` after the first 404,
> re-probe periodically) would halve HTTP calls on such models; deferred as a
> tuning decision, not a correctness issue.
>
> **Field findings, round 4 (2026-08-28, gpt-5.6-luna deep-dive).** The luna
> bench run (1 extracted fact) turned out to be a three-layer library defect,
> not a model deficiency — luna fully supports structured outputs:
>
> 13. **`temperature` broke `require_parameters` routing.** The constraint gates
>    on *every* parameter in the request, and no luna endpoint lists
>    `temperature` (reasoning-model family). Sending `temperature: 0.0`
>    unconditionally excluded all 7 endpoints → 404 at attempt 0, every call.
>    The 404 degrade ladder is now parameter-aware: strip incidental params
>    (temperature) while keeping the constraint, drop the constraint only if
>    the 404 persists.
> 14. **`_strict_schema` never transformed `$defs`.** Pydantic nests models via
>    `$ref`/`$defs`; the transform only rewrote the root object, so OpenAI's
>    strict validator rejected the whole schema (400: "'required' ... Missing
>    'kind'"), we silently degraded to `json_object`, and the model freestyled
>    a plausible-but-wrong envelope. Gemini's endpoints accept lenient schemas,
>    which is why this stayed invisible until a strict-validator model was
>    benched — the pigeonholing concern, confirmed in the schema layer. The
>    transform now covers `$defs`; luna returns schema-perfect output on the
>    first pass with full constrained decoding.
>
> Chain in one line: temperature → 404 → (fixed) → strict 400 → json_object
> fallback → freestyle JSON → repair without schema → drop. Every layer is now
> fixed: strip-incidentals routing, `$defs` strict transform, schema-in-repair.
>
> **Design revision (same day, post-review).** The degrade ladder above was the
> wrong philosophy: silent fallbacks let an integrator run a model at
> massively degraded quality without noticing. Replaced with explicit
> capability declaration (the mem0/graphiti/LiteLLM pattern): `LlmConfig`
> gains `temperature=None` (omit the parameter), `structured_outputs=
> "strict"|"json_object"`, and `params={}` (expert passthrough merged into the
> request body). All arbitrary fallbacks are deleted — 400/404/422 now raise
> `LlmCapabilityError` naming the model, the provider's error, and the knob
> that fixes it. This also wired up `config.temperature`, which was a dead
> knob (the adapter only ever read the per-request default).
>
> The loud-failure switch immediately paid for itself: luna's first strict
> pass surfaced a fourth schema bug the fallbacks had been hiding — pydantic
> emits `{"$ref": ..., "default": ...}` for enum fields with defaults, and
> OpenAI's strict validator rejects both `$ref` siblings *and* `allOf` in
> property position. `_strict_schema` now inlines the referenced definition
> (dropping `default`) for exactly that case; bare `$ref`s stay untouched.
> Result: luna runs the full suite at 81/81 hard checks with **zero** retries,
> 30 LLM calls total, $0.014154 — cheaper per run than gemini-3.7-flash.


### P0 — accuracy & reliability, cheap (do first)

**P0-1. Tiered deterministic reinforce — never spend an LLM call on the obvious.**
Today a normalized exact match lands in `plan.collisions` and costs a reconcile
call to reach the obvious NOOP. graphiti's fast path (identical fact → append
provenance, done) and CringeDiscordBot's production-proven upsert ladder (exact
normalized match, or embedding ≥ 0.92 → reinforce in place, no LLM;
≥ 0.85 → semantic candidate) are the references. Route `Collision.duplicates`
and ≥ `near_duplicate_threshold` neighbors straight to `commit_reinforce` in
`pipeline.py`; reserve the LLM for the ambiguous band in between.
*Files: `ingest/reconcile.py` (plan split), `ingest/pipeline.py` (commit loop).*

**P0-2. Batch reconcile decisions — one call per batch, not per collision.**
mem0 v2's one batched decide-call and graphiti's batched `dedupe_nodes` both
arbitrate many items per call; we make one call per collision (7 calls in the
latest sim run). Render all collisions into one prompt (candidate index → neighbor
ids), validate per-decision as today. Est. 30–50% reconcile token cut; fewer round
trips. *Files:* `ingest/reconcile.py`*,* `prompts/extraction.py`*.*

**P0-3. Hybrid retrieval: FTS5 BM25 ∪ cosine, RRF fusion, over-fetch.**
SQLite ships FTS5 — no new dependency. Over-fetch `max(4k, 60)` per leg, fuse by
reciprocal rank, gate on semantic score before blending (mem0's lesson: BM25 must
not resurrect junk). This is the single largest recall-quality lever available to
us. *Files:* `adapters/sqlite/` *(FTS5 table + triggers),* `recall/` *(fusion).*

**P0-4. Recency decay in recall ranking.**
`final = α·relevance + (1−α)·recency(last_seen_at)`, α≈0.8 default. Field-standard,
a few lines, prevents ancient reinforced facts from outranking current ones.
*Files:* `recall/`*,* `config.py` *(one knob).*

### P1 — cost & scale

**P1-1. Model routing: small model for reconcile/classify.**
graphiti splits medium (extraction) vs small (dedupe/attributes). Our reconcile and
classify calls are structured and shallow — a cheaper model tier is nearly free
accuracy-neutral savings. *Files:* `config.py` *(*`LlmConfig.small_model`*),
composition root.*

**P1-2. Optional SQLite LLM response cache.**
graphiti caches by MD5(model+messages). For dev loops, tests against recorded
traffic, and retry storms this is a large cost/latency win. Opt-in config knob;
bypass in production by default. *Files: new* `adapters/llm_cache.py`*, composition.*

**P1-3. Integer ID remapping in the reconcile prompt.**
We send `fct_01m…` ULIDs; mem0/graphiti remap to small ints before the prompt and
map back locally. Fewer tokens, measurably less id-hallucination. *Files:*
`ingest/reconcile.py`*.*

**P1-4. Reconcile prompt: prefer UPDATE for same-aspect refinement.**
Tonight's sim: "Nolan loves writing Go." + "…and considers it the best programming
language." committed as two rows — the model chose ADD where UPDATE (merge detail)
was right. One rule line in `RECONCILE_INSTRUCTIONS`. *Files:*
`prompts/extraction.py`*.*

**P1-5. Time-based batch flush (age trigger).**
CringeDiscordBot fires a batch on `size OR oldest-pending ≥ 300s`; we only trigger
on size, so a quiet user's last few messages can sit unextracted until the next
flush call. Add `batch_max_age_seconds` to batching config and let `observe()`
schedule the flush when the oldest pending message ages out.
*Files: `config.py`, `ingest/` queue, `api/client.py`.*

### P2 — structural bets (design before code)

**P2-1. Bi-temporal facts.** Add `valid_at`/`invalid_at`/`expired_at` to
`dm_facts`; recall emits validity language ("since March", "until June"). Enables
"what did we believe last month" and cleaner supersede semantics. Schema migration;
moderate scope. *Files:* `adapters/sqlite/connection.py`*,* `models/facts.py`*,*
`recall/`*.*

**P2-2. Pending ledger for low-confidence facts (TARL-inspired).**
Facts in the 0.55–0.7 confidence band land as `pending`, invisible to recall,
promoted on corroboration (second independent mention), expired otherwise. Directly
attacks memory pollution. *Files:* `models/facts.py`*,* `ingest/pipeline.py`*,*
`recall/`*.*

**P2-3. Entity-boost recall.** We already persist `dm_entities`; mem0's boost
(query entities → linked memories, similarity-weighted, hub down-weighted) is a
contained recall addition on top of P0-3. *Files:* `recall/`*,* `adapters/sqlite/`*.*

**P2-4. Our own eval harness.** Vendor benchmarks are untrustworthy; the e2e sim
already proved its worth by catching the reconcile bugs. Extend it into a
LoCoMo-style multi-session conversational eval tracking: recall@k, contradiction
retirement rate, duplicate rate, tokens/conversation, cost/conversation. Run it
like CI. *Files:* `examples/e2e_simulation.py` *→* `tests/eval/`*.*

**P2-5. Cite-on-use `[mem:N]` references in `prompt_context`.**
CringeDiscordBot's most-praised reliability trick: facts in the injected block
carry short `[mem:N]` aliases mapped to real fact ids, and only facts shown to
the model are citeable (pool ≡ prompt symmetry); post-generation can expand a
citation to its source messages. Gives bots grounded, auditable replies for
free. *Files: `recall/` (context assembly), `models/` (citation map).*

**P2-6. Lifecycle caps and TTL enforcement.** We have tier config
(`LifecycleConfig`) but no hard caps; cringe runs 300 facts/user, 500/guild with
short-term overflow pruning, plus TTL expiry on tiers. Prevents unbounded growth
at scale. *Files: `config.py`, `ingest/executor.py` or a maintenance pass.*

### Explicit non-goals (rejected with reasons)

- **Graph database (Neo4j), community detection, label propagation** — graphiti's
cost centers; no payoff at Discord-guild scale. Our `dm_relations` edges cover
the useful 1-hop cases.
- **mem0's append-only design** — empirically wrong for us: our sim shows
contradictions never retire without reconciliation. Keep reconcile; make it
cheaper (P0-1, P0-2, P1-1) instead.
- **Cross-encoder reranker dependency** — heavy model dependency for marginal gain
over RRF + recency at our result sizes. Revisit only if eval (P2-4) shows a
ranking bottleneck.
- **Procedural memory / vision captioning** — out of product scope.
- **CringeDiscordBot's structural debt**: the god-service monolith (our
  port/adapter split already avoids it), LLM UPDATE/DELETE keyed on opaque ids
  from a partial retrieved context (our reconcile supplies full collision
  neighborhoods instead), lifecycle maintenance on the retrieval hot path, and
  per-fact sentiment/tone analytics schemas (token-heavy, rarely read).

---



## 7. Suggested sequencing

1. P0-1 + P1-4 (reconcile correctness + cost) — small, finishes tonight's work;
  should turn the sim fully green deterministically.
2. P0-2 (batched reconcile) — biggest token win.
3. P0-3 + P0-4 (retrieval quality) — user-visible accuracy.
4. P1-1 + P1-2 (cost plumbing), P1-5 (age-triggered flush) — ops hardening.
5. P2 items behind design notes in PLAN.md, informed by P2-4 eval numbers.

Each P0/P1 item is independently shippable and covered by the existing
unit/integration suite plus the e2e sim as the acceptance gate.