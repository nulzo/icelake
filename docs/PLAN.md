# discord-memory: Library Design & Implementation Plan

A drop-in Python library that gives Discord bots accurate, scalable, cost-effective
agentic memory of their users — ChatGPT/Claude-style memory, but for every member of
a server, hardened against cross-user attribution errors.

Priority order (non-negotiable): **Accuracy → Cost → Performance → Features**.

---

## Part 0 — Ground Truth: Review of CringeDiscordBot's Memory System

The existing bot contains ~15k lines of memory machinery across `src/services/memory.py`
(~3,900-line orchestrator), 8 repositories, and ~20 memory services. It is a mature
mem0-style design with real production lessons baked in. The library should be an
*extraction and hardening* of that system, not a from-scratch invention.

### 0.1 What works and gets ported (proven machinery)

| Component | Where today | Verdict |
|---|---|---|
| Durable message batching (pending → claim → extract, DB-backed lease locks) | `memory.py:418`, `message.py`, `memory_batch_state.py` | **Port**, fix lease bugs (§0.3) |
| LLM batch extraction emitting ops with `source_message_indexes` | `memory.py:838` | **Port**, restructure prompts around participant roster tokens (§3.1) |
| Multi-strategy dedup: normalized-text exact → cosine fuzzy → reinforce-on-match | `_upsert_memory_with_dedup` (`memory.py:1756`) | **Port** as pure logic + store port |
| Quality gates: refusal/meta detection, raw-quote plagiarism check (≥0.88 similarity), snowflake bans, ephemeral-share filters, min-confidence | `memory_quality.py` | **Port nearly verbatim** — this is hard-won accuracy armor; convert to pure functions with exhaustive tests |
| Tier lifecycle: short/mid/long/core, TTLs, promotion on reinforcement, prune priority | `memory.py:3616–3897` | **Port**; add continuous decay (§4.4) |
| 8-channel recall + RRF fusion + hybrid rerank off-loop | `retrieval/recall.py`, `fusion.py`, `memory_retrieval.py` | **Port** architecture; trim channels to a curated set (§5) |
| Provenance snapshots (frozen `MemorySourceRef` w/ jump URLs captured at ingest) | `models/memory.py:205` | **Port verbatim** — citations survive message deletion |
| Attribution struct (`self / third_party / inferred / manual` + speaker ref) | `models/memory.py:234` | **Port**, make required not optional |
| Alias index w/ source ranks + weights + snowflake-poisoning guard | `entity_identity.py`, `entity_alias_repository.py` | **Port**, add decay + rename hooks (§3.2) |
| Bot guard (never remember bots) | `memory_bot_guard.py` | **Port** |
| Cite-on-use pool symmetry (only injected facts are citable) | `rag_context.py`, `citation.py` | **Port** concept for the injection block API |
| Graphiti-style edge invalidation (`valid_to` partial unique index) | `graph_edge_repository.py:33` | **Port** pattern for fact supersession (§4.3) |

### 0.2 What is right in spirit but wrong in structure

1. **God object**: `MemoryService` constructs 7 sub-services inline, takes 12 constructor
   args, and mixes ingestion, retrieval, lifecycle, identity, and graph wiring.
   The library decomposes into composable units behind ports.
2. **MongoDB-coupled everything**: repositories leak Motor collections into services;
   dedup/lifecycle logic lives at repo layer. Ports first (§6), adapters second.
3. **Synchronous CPU work on the event loop**: local embedding inference called directly
   (`recall.py:60`, `memory.py:1280/1501/2129/2643`) blocks all users. All inference
   goes through async ports executed via `asyncio.to_thread`/executors.
4. **Duplicated hot-path cleanup**: `_cleanup_memory_lifecycle` runs unthrottled from six
   call sites. Maintenance becomes scheduled background jobs only.
5. **Prompt text embedded in service code**: extraction prompts live as f-string blobs.
   They become versioned, testable prompt modules.

### 0.3 Correctness bugs the library must design away (found in review)

| # | Bug | Consequence | Library fix |
|---|---|---|---|
| B1 | **LLM-supplied `target_user_id` trusted verbatim** (`memory.py:1247`) — no membership check against batch participants | A hallucinated snowflake writes a durable third-party fact onto an arbitrary user's profile. *Worst accuracy hole.* | Roster-token protocol (§3.1): model can only reference participants by opaque tokens issued by us; IDs never enter the prompt |
| B2 | Claim protocol TOCTOU + no lease reclaim (`message.py:434–469`) — crash strands messages in `processing` forever; startup force-resets pending backlog | Silent data loss | Lease queue with heartbeats + expired-lease reclaim + idempotent acks (§4.6) |
| B3 | Server-batch trigger races (interval check without lock, `memory.py:360`) + modulo trigger on global counter | Duplicate extractions, cost waste | Keyed lease acquisition before any trigger; counter-free triggers (§4.2) |
| B4 | Extraction failure still acks messages (`finally` block) | Failed batches silently discarded | Dead-letter state + retry budget + poison-message cap (§4.6) |
| B5 | Naive `datetime.utcnow()` everywhere | TZ-mismatch corruption risk; deprecated | tz-aware UTC clock injected via `Clock` port; naive datetimes rejected at boundaries |
| B6 | Unbounded arrays: `source_messages` full-content snapshots grow on every reinforcement | Row bloat, slow reads | Cap provenance refs (e.g. 8) with primary-role preservation; overflow refs point to episode records |
| B7 | Atlas probe cache sticks off after one failure (`memory_repository.py:62`) | Permanent silent quality degradation | Health-checked backend selection with retry/backoff, explicit degraded-mode signal |
| B8 | Score-scale mismatch: flat 1.0 scores vs weighted rerank under same `min_score` gate | Threshold meaningless across modes | Single scoring contract; thresholds defined per-channel pre-fusion only (§5.2) |
| B9 | No user opt-out; "purge your memories" parses but does nothing | Compliance/UX gap | First-class consent flag + purge verb (§9.3) |
| B10 | No spend metering or budgets | Cost blowups undetectable | Token/cost metering port + per-scope budgets enforced in pipeline (§8) |
| B11 | O(n²) consolidation matrix materialized inline (5000 docs ≈ 50 MB) | Latency spikes, memory blowouts | Batched blocking computation off-loop, capped cluster size, incremental scheduling (§7) |
| B12 | Nickname drift: no rename listener; old aliases keep max weight forever | Stale identity links | Member-update hook + alias temporal validity + decay (§3.2) |

---

## Part 1 — Literature Foundations (what we adopt, and why)

Reviewed: mem0 paper + docs (arXiv 2504.19413), Zep/Graphiti (arXiv 2501.13956),
MemGPT/Letta (arXiv 2310.08560), LangMem, A-MEM (arXiv 2502.12110), MemoryBank
(arXiv 2305.10250), HippoRAG 2 (arXiv 2502.14802).

| Finding | Source | Adopted as |
|---|---|---|
| Two-phase pipeline: extract candidates → reconcile against retrieved similar memories | mem0 §Alg.1 | Conditional reconciliation: phase 2 fires **only when candidates collide** with existing memories (cost win, §4.3) |
| LLM chooses ADD/UPDATE/DELETE/NOOP via strict tool/JSON schema over ≤10 retrieved neighbors | mem0 | Reconciliation op schema (§4.3); deterministic code executes ops, LLM never touches storage |
| Production retreat from destructive updates toward append + supersession links | mem0 V3 additive pipeline | Soft-invalidate + `supersedes` link, never hard-delete on contradiction (§4.3) |
| Constrain dedup candidate space (same subject/entity pair only) | mem0 top-k=10; Graphiti same-entity-pair rule | Candidate retrieval filtered by `(guild_id, subject_id)` before reconcile (accuracy + cost) |
| Bitemporal facts: `valid_at`/`invalid_at` (world time) separate from ingest time; relative dates anchored to event time ("next Thursday") | Zep | Every fact carries `valid_from`, `valid_until`, `observed_at`; timestamps resolved at extraction time, stored absolutely (§4.4) |
| New information wins by ingestion order; history stays queryable | Zep T′ ordering | `superseded_by_id` chain; recall prefers non-invalidated, then most recent |
| Identity resolution = semantic candidate recall + LLM judge over name+summary; duplicates may have different surface forms | Graphiti reflexion pass | Alias index (cheap lexical) + embedding recall + LLM adjudication above ambiguity threshold (§3.2) |
| Atomic facts + derived summaries beat raw transcripts for multi-hop | MIRIX/LangMem profile-vs-collection | Dual representation: atomic fact collection + LLM-consolidated profile summary (§7) |
| Strength/reinforcement math: `R = e^(−t/S)`, `S += 1` and `t ← 0` on reinforcement | MemoryBank | Decay/strength module drives ranking + forgetting (§4.5) |
| Recall should combine similarity with importance and strength (recency/frequency of use) | LangMem | Ranking formula (§5.3) |
| Embedding-similarity is a cheap pre-filter; an LLM decides which links are real; older memories get *evolved* when new related ones arrive | A-MEM | Consolidation-time linking + summary evolution; never at hot path |
| Graph routing helps multi-hop but must not pollute the corpus with LLM summaries | HippoRAG 2 | Graph used for *recall channel seeds* only (tags, related_users, entity nodes); corpus stays atomic facts |
| Structured output at every boundary; small models break silently → validate + fallback ladder | mem0/Graphiti both | Pydantic schemas + parse-repair ladder + quality gates on every LLM boundary (§4.3) |
| Namespacing: every object scoped by `(org?, guild, user)` tuples | LangMem namespaces; Graphiti group_id | Scope key `(guild_id, subject_id | None)` on every record and query (§6.1) |

Non-goals informed by literature: full bitemporal community detection (Graphiti
communities — high cost, low evidence of need at guild scale), agent self-editing
memory tools (Letta — our writers are pipelines, not the chat agent), PPR graph walk
(HippoRAG 2 — revisit if multi-hop evals demand it).

---

## Part 2 — Product Definition

### 2.1 Positioning

> "Give your discord.py bot ChatGPT-grade memory of every member, in ten lines."

- **Consumer**: Python Discord bots (discord.py first-class; any framework via core API).
- **Scale target**: thousands of guilds × thousands of members × tens of thousands of
  memories per guild; multiple bot processes per deployment.
- **Cost posture**: default configuration amortizes to **≤ ~150 tokens of LLM spend per
  observed message** and zero-cost local embeddings; retrieval adds **zero** LLM calls.

### 2.2 Core capabilities (v1)

1. **Passive observation** — bot forwards message events; library handles batching,
   extraction, attribution, dedup, lifecycle. Fire-and-forget, bounded queue, backpressure.
2. **Hardened identity** — display names/nicknames/mentions resolve to Discord snowflakes;
   facts attach to subjects, never speakers-by-accident; third-party statements attributed
   correctly ("Bob says Alice loves 7-Up" lands on Alice's profile, marked third-party).
3. **Knowledge graph over memories (optional linkage, never required)** — every fact is
   self-contained with exactly one owner anchor; references and typed relation edges
   (`likes`, `dislikes`, `called_out`, …) are layered on when context warrants. Single-
   subject facts ("X likes movies"), interpersonal facts ("X called Y a hacker" — one
   record, retrievable from both endpoints), and server-wide traits all coexist. Entity
   nodes are shared junctions: X —likes→ *movies* and Y —dislikes→ *movies* aggregate on
   the same node, enabling "what does the server think about movies?" and "who shares
   X's taste?".
4. **Fact CRUD with truth maintenance** — ADD / UPDATE / DELETE / REINFORCE / SUPERSEDE;
   contradictions invalidate rather than erase; every mutation carries reason + citation.
5. **Time-aware recall** — absolute timestamps, tier TTLs, Ebbinghaus strength decay,
   "next Thursday" resolved at write time.
6. **Hybrid recall** — vector + BM25 + entity/tag + participant-link/graph channels,
   RRF fusion, calibrated rerank, token-budgeted injection block ready to paste into any
   prompt.
7. **Citations** — every injected fact carries stable jump-link provenance; optional
   `[mem:N]` tagging contract for models that cite.
8. **Governance** — per-user opt-out, purge, retention caps, export, admin search/delete,
   budgets and spend metering.

### 2.3 Explicit non-goals (v1)

- Voice/media memory, image understanding.
- Multi-agent shared-memory negotiation (Letta-style).
- Managed cloud service. The library is self-hosted BYO-infra.

---

## Part 3 — Accuracy Architecture (priority 1)

### 3.1 Roster-token protocol (kills hallucinated-ID attribution, fixes B1)

The single largest accuracy risk found in the bot: the extraction LLM echoes Discord
snowflakes it may have invented. The library removes snowflakes from the extraction
surface entirely.

- Pipeline issues each batch a **roster**: verified participants only
  (author + mention targets + thread participants, resolved through the alias service),
  rendered as:
  ```
  PARTICIPANTS (reference these EXACT tokens in target/speaker fields):
  <p0> = author of these messages
  <p1> = Alice (aliases: alice, alicia)
  <p2> = Bob (aliases: bobert)
  ```
- Model output references `<pN>` tokens for `speaker` and `subject`. Unknown people are
  named entities (`about_entities`), never user IDs.
- **Verification gate (hard)**: any op whose `subject`/`speaker` token is not in the
  roster is dropped with a logged violation. There is no path from LLM output to a
  user ID that we did not mint.
- Name→ID mapping happens deterministically post-extraction, inside the alias resolver.

### 3.2 Identity resolution ladder (name ↔ hardened ID)

Ported from the bot's alias system with fixes for B12:

1. **Alias registry** per `(guild_id, alias_norm) → [(user_id, source_rank, weight)]`.
   Sources: discord_username(100), subject_username(95), real_name(85), display_name(70),
   mention(60), backfill(50), entity_tag(40). Snowflake-like aliases rejected.
2. **Write-time capture**: every observed message registers author names; discord.py
   integration subscribes `on_member_update` to re-index renames (old aliases keep
   validity but stop accruing weight).
3. **Resolution order**: exact normalized match → prefix match (≥3 chars) → embedding
   recall over stored name-facts → LLM adjudication **only when top-2 within 10%**
   (ambiguity). Ambiguous ⇒ fact attaches to *no one* as a subject; it is stored as
   guild-scoped with `about_entities` instead. Never guess.
4. **Collision policy**: two members sharing a normalized name ⇒ both flagged; resolution
   requires context (mention presence, participant roster membership). Facts extracted
   during a session where the person spoke bind via roster tokens, bypassing name lookup.
5. **Third-party guards**: possessive/kinship patterns ("X's brother Ivan", "someone
   named Ivan") never create user aliases — ported from `is_third_party_name_reference`.

### 3.3 Anti-fabrication gates (ported + tightened)

Every proposed fact passes, in order, all pure-function gates (exhaustively unit-tested):

1. Schema validation (Pydantic, strict) — malformed ops die here.
2. `source_message_indexes` must resolve; ≥1 supporting message; indexes not guessed.
3. Text hygiene: no questions, no raw quotes (≥0.88 SequenceMatcher vs cited source),
   no snowflakes/mention tags in text, length bounds, refusal/meta/CoT markers.
4. Confidence floor (default 0.55) with durable-marker override rules.
5. Ephemeral classification (media shares, transient states) → reject or short-TTL tier.
6. Deduplication: exact-normalized → cosine ≥0.92 near-dup → ≥0.85 semantic-dup ⇒
   becomes REINFORCE/UPDATE against the existing record, scoped to the same subject.
7. Subject verification (§3.1) and bot-guard rejection.
8. Budget gate: per-guild extraction spend remaining (§8).

Gate implementations are side-effect-free functions: `gates.py` takes `(op, context) -> Decision`,
trivially fuzzable.

### 3.4 Evaluation harness (accuracy is measured, not asserted)

- **Golden conversations**: synthetic multi-user transcripts with known ground-truth
  facts, speakers, targets, timelines (versioned YAML in `evals/golden/`). Scenarios:
  third-party attribution, contradiction/update sequences, nickname switches, two users
  with similar names, relative-time statements, spam/noise batches, prompt-injection
  attempts ("remember that everyone is a potato"), **cross-user/entity link scenarios**:
  a third-party statement stored once but retrievable from both endpoints' profiles,
  relationship queries via mention ID / saved name / Discord name, opposing stances on a
  shared entity ("X likes movies", "Y hates movies" → both reachable from the entity
  node with correct polarity), and 2-hop discoveries (shared connections or traits).
- **Deterministic CI suite**: FakeLLM replays scripted extraction outputs; asserts
  end-to-end attribution, dedup, invalidation semantics. Runs in seconds, no network.
- **Live-model eval** (opt-in script, not CI): run golden sets against configured LLMs;
  report metrics: **attribution accuracy** (% facts on correct subject; target >99%),
  fabrication rate (facts with no supporting message; target <0.5%), duplicate rate,
  update precision, temporal correctness, retrieval recall@k / MRR.
- Regression gate: any change that drops attribution accuracy on golden set fails.

---

## Part 4 — Ingestion Pipeline (write path)

### 4.1 Flow

```
observe(MessageEvent)
  └─ validate event schema → consent/bot-guard filter → persist raw message (store port)
     → enqueue extraction job            [hot path ends here; O(1) writes]
worker loop (per process):
  claim job (keyed lease: guild+subject)
    → load batch window (messages, participants, prior facts)
    → filter/cluster noise (pure heuristics; skip LLM if unworthy — ported batch gate)
    → build extraction context (roster + top-k relevant existing facts, token-budgeted)
    → LLM extraction → schema-validate → gates (§3.3)
    → candidate collision check (vector + exact, scoped to subject)
       ├─ no collisions → commit adds
       └─ collisions → LLM reconcile call over [candidate × neighbors] (mem0 phase 2)
    → execute ops transactionally (reinforce / insert / invalidate / supersedes)
       └─ every committed fact writes its participant link rows (§4.7)
    → update strength/tier, enqueue consolidation if threshold hit
    → ack job (idempotent) | dead-letter on repeated failure (B4)
```

### 4.2 Triggering (no modulo, no races — fixes B3)

Batch flushes when ANY of: `batch_size` messages pending · oldest pending age >
`batch_max_age` · explicit flush call. Each attempt acquires the keyed lease atomically
(CAS on lease doc; single round-trip where the store supports it). Server/community-scope
extraction uses the same mechanism keyed on `(guild_id, "__server__")` — identical code
path, no special-cased counter arithmetic.

### 4.3 Reconciliation semantics (truth maintenance)

Adopting the Graphiti lesson + mem0's production retreat from destructive updates:

- **ADD** — new fact (no semantic match ≥ threshold among subject-scoped neighbors).
- **REINFORCE** — same meaning re-observed: `occurrences += 1`, `strength += 1`,
  `last_seen ← now`, merge citations (capped), recompute tier. *No LLM needed.*
- **UPDATE** — new info refines existing: old fact gets `superseded_by = new_id`,
  remains queryable as history; new fact carries `supersedes = old_id` + reason.
- **INVALIDATE** — contradicted: set `valid_until = now` on old fact, link contradictor.
  Nothing is hard-deleted automatically (purge/opt-out is the only erasure).
- **NOOP** — duplicate/irrelevant; counted for telemetry.

The reconcile LLM call fires **only when a candidate has ≥1 neighbor above the collision
threshold** (typically minority of candidates ⇒ big cost saving vs always-on phase 2).

### 4.4 Time model

- Every fact: `observed_at` (ingest), `valid_from`, `valid_until` (nullable),
  `last_reinforced_at`, plus tier TTL.
- Relative expressions ("tomorrow", "next week") resolved **at extraction time** against
  message timestamp (Zep lesson): stored absolutely, with the original phrase kept in
  metadata for audit.
- Expiry sweep is a scheduled maintenance job (not read-path lazy cleanup — fixes the
  duplicated-hot-path-cleanup problem); reads additionally filter `valid` defensively.

### 4.5 Strength & forgetting (MemoryBank math)

```
retention(t) = exp(-Δt_days / strength)          # Δt since last_reinforced
on observe/reinforce/use: strength += 1 ; last_reinforced ← now
forget candidates: retention < θ_forget AND tier != core AND not manual
prune order: (tier rank, retention asc, confidence asc)
```

Strength feeds ranking (§5.3); retention drives forgetting; tiers remain coarse caps
(ported short/mid/long/core semantics). Pure module, property-tested (monotonicity,
reinforcement resets curve, core/manual immunity).

### 4.6 Delivery guarantees

- Jobs are rows in a work queue table/collection with `state ∈ {pending, leased, done,
  dead}`, `lease_expires_at`, `attempts`. Lease renewal via heartbeat; expired leases
  reclaimed by any worker (fixes orphaned `processing` messages, B2).
- Op application is idempotent per `(job_id, op_index)`; acks recorded per-op so retries
  skip completed work.
- Poison jobs (≥ N failures) → `dead` + metric + optional webhook. Raw messages are never
  lost: dead-letter preserves ability to reprocess manually.

### 4.7 Memory scopes & the knowledge graph (write side)

**Anchoring invariant — one fact, one owner.** Every fact carries exactly ONE primary
anchor, which decides ownership, lifecycle, pruning, and purge semantics:

| Scope | Anchor | Example | Frequency |
|---|---|---|---|
| Personal | `(subject_user_id)` | "X likes going to watch movies" | The default by far |
| Interpersonal | `(subject_user_id)` = the person the statement is **about**; speaker stored as attribution | "X called Y a hacker" → anchored on Y, `speaker=X` | Context-dependent |
| Server-wide | `(guild_id)`, no subject | "The community loves sci-fi movies" | Steady trickle |

Linking is **optional and additive**: a fact with zero references beyond its anchor is a
complete, first-class memory. When context warrants (mentions, third-party statements,
named entities), references and derived edges are layered on without changing ownership.
Directionality rule for interpersonal facts: about-target wins as anchor; if no target is
resolvable (mutual banter), anchor = speaker and the other party becomes `mentioned_with`.

**Three-layer graph model** (evidence stays in the fact store; topology is materialized
around it):

```
Layer 1  FACTS          self-contained records w/ provenance, validity, strength.
                        Exactly one anchor each. Never duplicated per participant.

Layer 2  INCIDENCE      memory_links(guild_id, memory_id, node_type ∈ {user, entity},
   (fact↔node)             node_id, edge_kind ∈ {subject_of, speaker_of, about_user,
                           mentioned_with, about_entity}, valid_from/until)
                        → answers "which FACTS touch node X?" (exact citations)
        Indexes: (guild_id, node_id, valid_until)              # forward lookup
                 (guild_id, memory_id)                         # reverse lookup

Layer 3  RELATIONS      relations(guild_id, src_type, src_id, dst_type, dst_id,
   (node↔node)            verb ∈ {likes, dislikes, member_of, called_out, brother_of,
                              teammate_of, …}, polarity, weight, occurrences,
                              confidence, evidence_ids[≤8], valid_from/until)
                        → answers "what is X's relationship to Y / to 'movies'?"
                        Aggregated across many observations; unique partial index on
                        (guild, src, dst, verb) where valid_until IS NULL; contradiction
                        supersedes the old edge bitemporally (Graphiti pattern).
```

Write rules:
- A personal fact with an extracted stance writes Layer-2 rows for anchor + entity
  (`about_entity→movies`) and upserts ONE Layer-3 edge (`X —likes→ movies`,
  weight = ln(1+occurrences) × confidence × recency-decay). Another user's opposing fact
  hits the SAME entity node and creates their own independent edge (`Y —dislikes→
  movies`). Entity nodes are shared junctions that accumulate everyone's stances.
- Third-party statements ("X called Y a hacker"): fact anchored on Y; incidence rows
  `subject_of→Y`, `speaker_of→X`; relation edge `X —called_out→ Y` with the fact as
  evidence. Retrievable from both profiles via incidence; owned unambiguously by Y.
- LLM-proposed verbs/names resolve through the alias ladder before ANY row is written;
  no LLM-minted IDs or slugs ever reach storage.
- REINFORCE increments edge weight/occurrences; INVALIDATE/SUPERSEDE soft-invalidates
  fact rows and re-derives affected edges transactionally (bitemporal consistency).
- Entity aliasing: slug normalization at write + consolidation-time embedding merge
  ("movies"/"cinema"/"film" → canonical node with alias set), so stances aggregate on
  one junction instead of fragmenting across near-duplicate nodes.

Scale mechanics (DSA):
- All adjacency access is index-only lookups — no array scans, no regex-over-text
  (fixes the bot's mention-search hazard). Hop cost = O(Σ degree of visited nodes),
  bounded by fan-out caps.
- **Hub mitigation**: popular entities ("games", "movies") become high-degree hubs.
  Every expansion fetches top-k edges ranked by `weight` (which decays with staleness),
  never full adjacency. Per-node degree cap (default 512 active): maintenance job sheds
  lowest-weight edges to cold state rather than blocking writes.
- **Hot-row contention**: synchronous counter increments on popular nodes serialize
  writers at scale. Degree counts are therefore approximate — refreshed by a periodic
  rollup job per guild; exact counts remain available via index queries when needed.
- Shared-trait discovery ("users similar to X") = intersection of adjacency sets via
  sorted merge-join O(d₁+d₂), capped; "who else likes movies" = single forward index
  lookup on the entity node, ranked by edge weight. No Louvain/PPR machinery in v1 —
  these aggregate queries are provably served by indexed lookups.

---

## Part 5 — Retrieval (read path)

### 5.1 Query model

```python
@dataclass(frozen=True)
class RecallQuery:
    guild_id: str
    text: str | None                  # natural language query / recent context
    subject_ids: tuple[str, ...]      # hardened IDs to focus on
    pair_ids: tuple[str, str] | None = None   # relationship mode: resolved (a, b)
    scope: Literal["subjects", "guild", "server"] = "subjects"
    exclude_ids: tuple[str, ...]      # e.g. asking bot itself
    top_k: int = 8
    max_per_subject: int = 4
    token_budget: int = 600           # injection budget
    channels: ChannelSet = ChannelSet.DEFAULT
```

Scope discipline (ported): `subjects` restricts candidate space to those users' facts +
facts about them; `server` restricts to community facts; `guild` is the fallback union.
Subject restriction is enforced in the store query, not post-filtering.

### 5.2 Channel architecture (curated from the bot's 8)

| Channel | Mechanism | Keep? |
|---|---|---|
| vector | ANN over fact embeddings, subject/scope prefilter | Yes (core) |
| keyword | BM25/FTS over fact text | Yes |
| entity/tag | exact tag hits (`entity:{slug}`, `alias:{slug}`, `about_entity:{slug}`) | Yes |
| related-users | `memory_links` adjacency (§4.7): facts joined to X via any edge kind — subject, speaker, about, mention — regardless of which profile they surfaced from | Yes (core) |
| recency/strength | top-strength recent facts as baseline (profile anchor) | Yes (cheap, replaces bot's `subject_self` dump) |
| graph-hop | bounded 2-hop expansion over the link graph: shared partners, shared traits, relationship edges (§5.4) | Yes |
| mention-regex scan | unindexed regex over text | **Drop** (B-class perf hazard; replaced by indexed `memory_links`/tags) |

All channels run concurrently; per-channel failure degrades gracefully with a warning
metric (never raises into caller).

### 5.3 Fusion & ranking (single scoring contract — fixes B8)

1. Per-channel ranked lists → **RRF** (`k=60`, ported — robust to heterogeneous scores).
2. Candidate pool (top ~100) rescored once, off-loop, with calibrated components:
   ```
   final = w_sem·semantic        # cosine, query-embedding cached
         + w_lex·bm25_norm
         + w_ent·entity_overlap
         + w_str·strength_signal # log(strength)·recency-decay (LangMem principle)
         + mode_boosts           # small additive boosts per recall mode
   ```
   Weights fixed defaults, tunable via config; **all** components produce comparable
   [0,1]-normalized values or are omitted — no mixed scales, no fabricated minimum
   scores (the bot smuggled below-threshold chunks past gates; forbidden here).
3. Hard filters after ranking: consent, bot-guard, validity window, per-subject cap.
4. Injection builder packs selected facts into a labeled block with per-subject headers
   ("Facts about ALICE ONLY — do not attribute to the asker"), token-budget aware,
   citation refs included. Returns structured `RecallResult` (facts + scores + prompt
   block + citation map) — transport-free.

Optional LLM query understanding (scope hints, query rewrite) is a **pluggable
pre-ranker** behind a port; disabled by default (zero-LLM retrieval), enabled by config
for consumers who want the bot's intent-routing behavior.

### 5.4 Relationship, entity & multi-hop recall (first-class, not a flag)

Five canonical query shapes, all answered without an LLM in the loop unless configured:

| Shape | Example | Mechanism |
|---|---|---|
| **Q1 profile** | "what do you know about X" | `subjects=(X,)` → vector+keyword+links channels scoped to X |
| **Q2 cross-linked fact** | "did X call Y a hacker?" / fact retrievable from either endpoint | `memory_links` intersect: facts joined to both X and Y (single indexed self-join on `(guild_id, node_id)`); union with per-subject recall |
| **Q3 relationship** | "what does X think about Y" (mention ID *or* saved name *or* Discord name) | resolve both identifiers via alias ladder (§3.2) → `pair_ids` mode: `relations` edges between them first, then bidirectional text/vector search with both resolved names; inject with explicit directionality ("X said about Y…", "Y said about X…") so attribution stays crisp |
| **Q4 discovery (hops)** | "who else does Y argue with?", "shared connections of X and Z" | 1–2 hop BFS over incidence + relation adjacency: node→facts/nodes→co-linked nodes, ranked by summed edge weight × fact strength; fan-out capped (top-N per hop), depth hard-limited to 2 in v1 |
| **Q5 entity aggregation** | "what does the server think about movies?", "who shares X's taste in games?" (entity by name → slug via alias ladder) | resolve entity node → forward index lookup of incident `relations`, grouped by polarity/verb and user → top facts as evidence per stance. Opposing stances co-presented honestly ("3 members love it; Y dislikes it") |

Design constraints that keep Q4/Q5 honest at scale:
- All traversal is index-only lookups with hub-aware top-k expansion — never regex
  scans or unbounded adjacency fetches.
- Every hop/aggregation result is still a *fact* (with citations), never an inferred
  summary — the graph routes to evidence, it doesn't generate claims (HippoRAG 2
  lesson).
- Hop results carry their path (`Y —called_out→ X ←called_out— W`) in metadata so the
  injection builder can phrase provenance truthfully instead of overclaiming.
- Identifier resolution for Q3/Q4/Q5 accepts `{mention ID, Discord username, display
  name, saved real name}` through the same §3.2 ladder as extraction-time resolution —
  one resolver, one ambiguity policy (ambiguous ⇒ ask/disambiguate, never guess).
- Server-scope traits participate too: community facts referencing entities/users
  surface in every participant's Q1/Q4 results via their link rows.

---

## Part 6 — Architecture: Ports, Adapters, Composition

### 6.1 Layering (one concern per module; deps point inward)

```
┌────────────────────────────────────────────────────────────┐
│ integrations/discord_py (optional extra; thin listeners)   │
├────────────────────────────────────────────────────────────┤
│ api: DiscordMemory facade (the ONLY class consumers touch) │
├──────────────┬──────────────────────┬──────────────────────┤
│ ingest/      │ retrieval/           │ consolidation/       │
│ pipeline,    │ channels, fusion,    │ summarizer, linker   │
│ gates,       │ rerank, injection    │ (background jobs)    │
│ reconcile    │                      │                      │
├──────────────┴──────────────────────┴──────────────────────┤
│ domain: models (frozen dataclasses/pydantic), identity/,   │
│ lifecycle/ (tiers, strength), scoring — pure logic         │
├────────────────────────────────────────────────────────────┤
│ ports/: MemoryStore, VectorIndex, WorkQueue, Embedder,     │
│ ChatLLM, Clock, IdGen, Meter                               │
├────────────────────────────────────────────────────────────┤
│ adapters/: sqlite, postgres(pgvector), in_memory,          │
│ openai_compat llm, local/api embedders, noop meter         │
└────────────────────────────────────────────────────────────┘
```

Rules enforced by import-linter (CI):
- `domain/` imports nothing above it.
- Only `adapters/` imports vendor SDKs.
- Only `api/` is exported from the package root.
- Every public function fully type-annotated; mypy strict clean.

### 6.2 Ports (Protocol classes, runtime-checkable)

| Port | Surface (abridged) | Adapters shipped |
|---|---|---|
| `MemoryStore` | facts CRUD w/ optimistic versioning, scoped queries, supersede/invalidate, **`memory_links` incidence + `relations` typed-edge tables (bidirectional-indexed; intersect, entity-aggregation, 1–2 hop queries)**, alias + entity tables, episodes | SQLite (default), Postgres, InMemory (tests) |
| `VectorIndex` | upsert/search/delete, metadata filters | sqlite-vec/numpy (embedded), pgvector, in-memory brute |
| `WorkQueue` | enqueue/claim(lease)/heartbeat/ack/dead-letter, keyed leases | shared table inside MemoryStore backends |
| `Embedder` | `embed(texts) -> list[Embedding]` (batch, async) | sentence-transformers (thread-offloaded, free default), OpenAI-compatible API |
| `ChatLLM` | `complete(messages, response_schema, tier) -> ValidatedOutput` | OpenAI-compatible (OpenRouter/OpenAI/Ollama/vLLM) |
| `Meter` | tokens/cost/latency counters, budget checks | logging, Prometheus-friendly dict, Noop |
| `Clock`, `IdGen` | tz-aware now(); ids | system defaults; fakes for tests |

Storage choice rationale: SQLite gives frictionless drop-in for small bots (single file,
zero services); Postgres/pgvector is the scale-out path (ANN + FTS + row-level
`(guild_id, subject_id)` partitioning in one engine); MongoDB adapter is *contributable
later* because the bot's Mongo logic maps cleanly onto `MemoryStore` — proving the port
isn't Mongo-shaped. The swap test: adding a backend = one adapter package + config; zero
domain changes.

### 6.3 Consumer surface (frictionless)

> **Normative reference:** the full consumer contract — lifecycle, every method
> signature, result/error models, events, and recipes — is specified in
> [`API.md`](./API.md). The sketch below is illustrative only.

```python
from discord_memory import DiscordMemory, MemoryConfig

config = MemoryConfig(
    storage="sqlite:///memory.db",          # or postgres://...
    llm="openrouter://...?model=google/gemini-2.5-flash",
    embeddings="local",                     # or "openai://text-embedding-3-small"
)
memory = DiscordMemory(config)              # builds adapters internally

await memory.observe(MessageEvent(...))     # fire-and-forget, non-blocking
result = await memory.recall(RecallQuery(guild_id=..., text=..., subject_ids=(uid,)))
prompt.system += result.injection_block     # paste-ready, budgeted, citable

# admin / governance
await memory.forget_user(guild_id, user_id)
await memory.set_opt_out(guild_id, user_id, True)
stats = await memory.stats(guild_id)
```

Plus `integrations/discord_py.py`: a cog/listener mix-in (~100 lines) that wires
`on_message` → `observe`, `on_member_update` → alias refresh, and exposes `/memory me`,
`/memory forget`, owner purge/export commands. Kept thin; core stays transport-free and
testable without discord.py installed (optional extra `[discord]`).

Background workers: `asyncio.TaskGroup` workers started via `await memory.start()`;
library also exposes `run_pending(now)` for consumers preferring cron/external schedulers
(N-process safe by construction via leases).

---

## Part 7 — Derived Representations & Consolidation

- **Profile summaries** (LangMem profile pattern): one consolidated per-subject summary
  doc, regenerated asynchronously after N adds (default 5, ported) with LLM, validated
  against source facts (embedding sanity ≥0.55 — ported anti-drift check). Stored with
  TTL cache semantics; injection prefers summary + top atomic facts (MIRIX finding:
  consolidated events drive multi-hop wins).
- **Entity nodes**: `(guild, slug, kind, linked_user_id?, fact_count, summary≤480)`
  maintained from `about_entities`; feed tag channel + graph seeds. Caps enforced
  (defining memories ≤ 20/node).
- **Community/guild digest**: rolling server-culture facts (community scope), regenerated
  like profile summaries.
- Consolidation runs exclusively as background jobs with keyed leases; never on read or
  reply paths. Batched similarity computations off-loop with hard caps.

---

## Part 8 — Cost Engineering (priority 2)

Budgets and metering are structural, not aspirational:

1. **Model tiering**: extraction/reconcile on a cheap configurable model
   (default suggestion: a fast/cheap flash-tier model); consolidation may use a stronger
   model; retrieval = zero LLM calls by default.
2. **Amortized batching**: extraction fires once per ~10 messages (or 5-min age flush);
   noise gate skips ~30–60% of batches without LLM (ported `batch_worth_extracting`);
   conditional reconcile avoids always-on phase 2. Target: **≤150 amortized tokens/msg**
   including overhead.
3. **Context budgets**: extraction context = roster + ≤k=24 relevant facts (token-
   estimated, truncated deterministically); injection budget default 600 tokens; every
   prompt builder takes explicit budgets and reports actual usage to `Meter`.
4. **Embedding economics**: local model default (free); API mode batches (32/call) and
   caches by content-hash (persistent, not 64-entry LRU).
5. **Enforced budgets** (`Meter` port): per-guild daily/monthly token ceilings with
   graceful degradation ladder: skip reconcile → skip extraction → skip consolidation →
   alert. Retrieval never blocked (it's free).
6. **Observability**: counters for calls/tokens/cost by purpose (extract/reconcile/
   consolidate/summarize), cache hit rates, batch skip rates, dead-letter depth —
   exported via `Meter` so consumers ship their own dashboards.

---

## Part 9 — Performance & Scale (priority 3)

### 9.1 Hot-path SLOs (no LLM involved)

- `observe()` ack: p99 < 5 ms (two inserts, fire-and-forget).
- `recall()` warm: p95 < 120 ms at 50k facts/guild (ANN + FTS + fusion; rerank off-loop).
- Worker throughput: ≥ 30 msgs/sec/process with local embeddings; linear scaling across
  processes via lease queue.

### 9.2 Techniques

- Async everywhere; CPU-bound work (embeddings, BM25, fusion math) in executors
  (`anyio.to_thread`); no synchronous vendor calls on the loop (bot's recurring sin).
- Persistent query-embedding cache; content-hash fact-embedding reuse.
- Composite indexes leading with `(guild_id, ...)`; partial indexes for active+embedded
  rows (ported); pagination everywhere; no unbounded arrays (B6: provenance capped,
  overflow → episode records).
- Backpressure: bounded internal queues; `observe()` sheds to persistent queue rather
  than growing RAM; consumer-visible gauge.
- Load simulation harness (`evals/load/`): synthesizes K guilds × U users × M messages,
  runs pipeline against SQLite/Postgres, asserts SLOs and cost ceilings — runs in CI
  nightly at reduced scale, full scale locally.

### 9.3 Governance & safety

- Consent: `opt_out(guild,user)` persisted; observe/recall filters enforce; purge removes
  the subject's facts + summaries + aliases + vector entries, every `memory_links` row
  pointing at them, and every `relations` edge they endpoint — while *other* users' facts
  that merely mention them keep their text (their link/edge rows to the purged user are
  removed; the subject is forgotten without rewriting other people's history). Entity
  nodes are never purged by user deletion — only orphaned ones (zero remaining edges)
  get garbage-collected by maintenance.
- Injection labels prevent cross-user attribution in prompts (ported CURRENT_ASKER /
  REFERENCED_USER separation).
- Prompt-injection resistance: extraction treats message content as data, never
  instructions; gates strip instruction-like content; golden-set adversarial cases.
- PII minimization: snowflakes/mentions banned from fact text; provenance holds the IDs.

---

## Part 10 — Package Layout & Testing

### 10.1 Layout (module ≤ ~300 lines; function ≤ ~40; justification comments where exceeded)

```
discord_memory/
  __init__.py            # exports DiscordMemory, config, models only
  config.py              # MemoryConfig (validated pydantic-settings)
  api.py                 # DiscordMemory facade (~200 lines max; composition only)
  models/                # frozen, schema-validated boundary types
  identity/{resolver,aliases,guards}.py
  graph/{links,relations,traversal}.py   # incidence rows, typed edges, hop BFS (pure)
  ingest/{queue_worker,pipeline,context_builder,extraction,reconcile,gates}.py
  prompts/               # versioned prompt templates + fixtures
  lifecycle/{tiers,strength,maintenance}.py
  retrieval/{service,channels,fusion,rerank,injection}.py
  consolidation/service.py
  ports/*.py             # protocols
  adapters/{in_memory,sqlite,postgres,llm_openai_compat,embedders}/
  integrations/discord_py.py
tests/unit/...           # pure logic, fakes at seams, no IO
tests/integration/...    # real sqlite/pg containers, fake LLM scripts
evals/{golden,live,load}/...
```

### 10.2 Testing doctrine

- Business logic tested without transport/network/live services: FakeLLM (scripted +
  property-mutating), FakeClock (deterministic time travel for decay/TTL tests),
  InMemory adapters shared with prod code paths (adapter conformance suite runs every
  backend through identical scenarios — the port contract is literally executable).
- Exhaustive coverage mandates: gates (§3.3), reconcile state machine, identity ladder,
  RRF/rank math, strength/decay, lease queue concurrency (hypothesis + asyncio
  race replay).
- Contract tests: every adapter passes conformance suite; schema round-trips; migration
  tests for store schema versions.
- CI: ruff + format + mypy strict + import-linter + pytest (unit+integration) +
  golden-set deterministic evals; nightly: load sim + optional live-model eval (flagged,
  metered).

---

## Part 11 — Roadmap (phases with exit criteria)

**M0 — Foundation (pure core)**
Ports, models, config; identity ladder; gates; reconcile state machine; tiers/strength;
RRF/fusion; **knowledge-graph domain model (incidence links, typed relations, stance
edges) + traversal (hop BFS, entity aggregation, path metadata)**; in-memory
adapters; FakeLLM harness. *Exit*: unit suite green incl. golden-attribution tests with
scripted LLM; cross-user/entity link semantics proven on the InMemory adapter; mypy
strict clean.

**M1 — Drop-in MVP**
SQLite adapter (+ sqlite-vec/numpy vector), lease worker, extraction pipeline
end-to-end incl. incidence/relation writes, recall service with Q1/Q2/Q5 shapes,
injection builder,
discord.py integration extra, opt-out/purge/stats API. *Exit*: example bot runs on a
real server for a week; a third-party statement is retrievable from both endpoints;
entity stances aggregate per-node; amortized cost ≤ target; recall p95 within SLO on
seeded 50k-fact DB.

**M2 — Accuracy hardening**
Roster-token verification end-to-end; live-model eval harness with scorecard;
contradiction/supersession flows; alias rename hooks; adversarial golden cases;
consolidation service w/ validation; **Q3 relationship + Q4 hop-discovery modes with
mixed-identifier resolution**. *Exit*: attribution ≥99%, fabrication <0.5% on golden
sets across 2 model families; relationship-query golden cases pass from every identifier
type; contradiction tests prove history preserved.

**M3 — Scale-out**
Postgres/pgvector adapter passing conformance suite; budget enforcement + Meter
exports; load harness at 1k guilds × 50 users × 40 msgs/day equivalent; multi-process
soak test (lease correctness under contention). *Exit*: linear worker scaling ×8
processes; no lost/duplicated facts under kill -9 chaos tests.

**M4 — Polish & release**
Docs/tutorials, packaging (extras: `[postgres] [discord] [local-embeddings]`),
semver + changelog, benchmark publication (LOCOMO-derived guild variant), MongoDB
adapter RFC. *Exit*: fresh-consumer install-to-first-recall < 5 minutes.

---

## Part 12 — Decisions Register (ADR seeds)

| # | Decision | Alternatives rejected | Rationale |
|---|---|---|---|
| D1 | Ports + adapters, storage pluggable (SQLite default → PG at scale) | Mongo-only (bot heritage) | Swap test; Mongo adapter remains contributable; AGENTS.md replaceability mandate |
| D2 | Roster tokens, not snowflakes, in extraction I/O | Trust-but-verify IDs | Removes hallucination class entirely; cheaper prompt than full ID lists |
| D3 | Conditional reconcile (LLM phase 2 only on collisions) | Always-two-phase (mem0) / single-phase | ~half the reconcile calls at equal accuracy; collisions minority |
| D4 | Supersede/invalidate, never auto-hard-delete | mem0-paper DELETE | History queryable; reversible; matches industry retreat from destructive updates |
| D5 | Ebbinghaus strength for rank + forget | Flat TTL-only (bot) | Continuous ranking signal + principled forgetting; trivially testable |
| D6 | Zero-LLM retrieval default | Intent-LLM routing always-on (bot) | Cost priority; pluggable pre-ranker keeps the capability |
| D7 | Background-only maintenance | Lazy read-path cleanup (bot) | Predictable latency; kills duplicated throttling bugs |
| D8 | Atomic facts + profile summaries dual-layer | Single layer | MIRIX/LangMem evidence for multi-hop; summaries are derived/cacheable |
| D9 | Heterogeneous knowledge graph as v1: facts stay self-contained with ONE anchor (user or guild); optional `memory_links` incidence + `relations` typed edges (users ↔ entities, users ↔ users) materialized around them; entity nodes are shared junctions; bounded 2-hop BFS for discovery | Mandatory multi-participant facts; denormalized `related_users` arrays only (bot); full property-graph DB (Neo4j); PPR walks (HippoRAG 2) | Single-subject facts remain the simple default case; linking is additive so ownership/lifecycle/purge stay unambiguous; indexed adjacency serves relationship, entity-aggregation ("server thinks…"), and shared-trait queries without array scans or a second database; hops return evidence-backed facts with paths, never inferred claims |
| D10 | One typed structured-output path: every JSON-producing LLM call goes through `complete_structured(llm, model=...)`; the Pydantic model is both wire schema (strict `json_schema` on the first pass) and validator; one feedback-repair retry, then `None` | Per-call-site parsers/repair blocks; prompt-only schema instructions; shape-coercion shims for off-spec model output | Constrained decoding beats instruction-following (OpenAI/OpenRouter docs); a single helper removes three divergent hand-rolled contracts; off-shape output is a schema violation → retry → drop, never silently coerced |
