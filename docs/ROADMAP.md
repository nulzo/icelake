# Roadmap — from working library to professional, composable memory layer

Date: 2026-08-28. Status: living document; items link to evidence in
`RESEARCH.md` (field findings 1–14) and exit criteria in `PLAN.md` Part 11.
Audience: maintainers and contributors deciding **what to build next and why**.

---

## 1. Where we are (measured, not asserted)

The P0/P1 program from `RESEARCH.md` §6 is shipped and verified by the public-API
e2e suite (`examples/e2e_simulation.py`, 81 hard checks + 43 model expectations):

| Evidence | Value |
|---|---|
| Hard invariants (both models benched) | 81/81, exit 0 |
| LLM calls per full sim run | 30–40 (was 61+ before reconcile batching) |
| Cost per run | $0.014 (gpt-5.6-luna) / $0.058 (gemini-3.7-flash) |
| Cross-model benchmark harness | `examples/bench_models.py` (parallel, JSON+MD reports) |
| Structured-output path | strict `json_schema` + `require_parameters`; `LlmCapabilityError` on HTTP mismatch; `StructuredOutputError` dead-letters the batch |
| Concurrency | keyed leases, atomic `UPDATE…RETURNING` claims, graceful drain |

Architecture is already graphiti-lite with mem0's single-call extraction:
ports/adapters, staged reconcile (deterministic → LLM on residuals only),
bi-temporal soft invalidation, hybrid FTS5+cosine retrieval with RRF, event bus,
cost metering with provider-reported spend.

## 2. Design rules distilled from the field findings

Every bug we fixed generalizes to a rule. Future work is reviewed against these:

1. **Configured means wired.** Three separate findings (budgets, `config.temperature`,
   `weight_strength`/`weight_entity`) were dead knobs. Every new config field must
   ship with a test that observes its effect on the wire or in behavior.
2. **Fail loudly, degrade never.** Silent fallback ladders hid four schema bugs.
   Capability mismatches raise; the integrator declares capabilities
   (`temperature=None`, `structured_outputs`, `params`).
3. **Deterministic before LLM.** Exact/near-duplicate reinforcement, state-change
   collision detection, and event publication are deterministic; the LLM judges
   only genuine ambiguity. Each LLM call must justify its existence.
4. **Model quality is not a library guarantee.** The sim's two-tier check system
   (hard `check` vs soft `expect`) exists because hard checks that secretly
   required strong extraction recall produced false failures. Probes for library
   mechanisms must be deterministic (curation-API-driven).
5. **Concurrency bugs only surface under concurrency.** `claim_batch`
   double-processing, stranded leases, and drain orphans were invisible to the
   sequential sim. Concurrency claims require a load harness (W2).
6. **The wire layer is a boundary; test it like one.** Golden evals script LLM
   *outputs* and therefore never exercised schema transform or routing — where
   four bugs lived. Schema transforms get unit tests against real pydantic
   output; routing gets live probes per model family.

## 3. Gap inventory (everything discovered, triaged)

| # | Gap | Evidence | Severity | Status |
|---|---|---|---|---|
| G1 | Alias mining is regex-on-text; phrasing variance binds third-party names to the wrong subject (GLM bench: "gregory" aliased onto Bob) | bench aug27 | **High** (attribution correctness) | open → W1 |
| G2 | Models rarely re-emit known facts, so observe-path reinforcement under-fires | RESEARCH §6 finding (a) | Medium | open → W1.2 (prompt now live; shadow reinforce still needed) |
| G3 | Extraction recall varies on small batches (3-msg batch occasionally yields zero ops) | RESEARCH §6 finding (b) | Medium | known limitation → W1.4 few-shot (instructions now wired) |
| G4 | Reconcile decision quality varies by model (luna missed Omaha→Seattle supersede) | bench aug28-luna2 | Medium | open → W1 (deterministic slot contradiction) |
| G5 | Golden evals thin: `contradiction_supersede` tests no supersede; no capability-error scenario; wire layer uncovered | evals/golden review | Medium | open → W2 |
| G6 | `evals/load/` empty — no concurrency/throughput harness | PLAN §624 | **High** (rule 5) | open → W2 |
| G7 | SQLite vector scan is brute-force; recall cliffs at `candidate_cap` (default 500), not 50k | PLAN D1 | Medium | open → W3 (pgvector / sqlite-vec) |
| G8 | Azure-only endpoints want `max_completion_tokens`, not `max_tokens` | luna probe | Low | **closed** — `LlmConfig.max_tokens_key` |
| G9 | Many providers ignore the `reasoning.effort` dial | bench aug28-lowreasoning | Low | documented in API.md |
| G10 | Caps/TTL were unwired | RESEARCH P2-6 | Medium | **closed** (enforcement existed; prune victim order fixed 2026-08-28). Residual: CORE not fully exempt → W4.1 |
| G11 | No pending ledger for 0.55–0.7 confidence facts (pollution defense) | RESEARCH P2-2 | Medium | open → W4 |
| G12 | Cite-on-use `[mem:N]` in `prompt_context` | RESEARCH P2-5 | Medium | **closed** — `InjectionBuilder` + `PromptContext.apply_citations` |
| G13 | Postgres/pgvector adapter absent (D1 swap path unproven at scale) | PLAN M3 | **High** for scale-out | open → W3 (`postgresql://` fails with a clear error; no phantom extra) |
| G14 | Mongo adapter tested only against live instances (skipped in CI) | tests/integration/test_mongo_live.py | Low | open → W3 (conformance via container or documented manual gate) |
| G15 | Changelog / CI / semver policy / migration framework | PLAN M4 | Medium (adoption) | open → W6 (extras exist: mongo, discord, local-embeddings) |
| G16 | No metrics-export hook (event bus + meter exist; no OTel/Prometheus bridge) | — | Low | open → W6 |

Closed this cycle (for the record): dead budget wiring, `claim_batch`
double-processing, lease/drain lifecycle, `require_parameters` routing,
`$defs`/`$ref` schema transforms, dead `temperature` knob, silent degrade
ladders, curation-path event publication, bi-temporal `valid_until` on
supersede, gate false-positives, category-gated contradiction blindness.

Reliability pass 2026-08-28: `EXTRACTION_INSTRUCTIONS` actually sent;
`StructuredOutputError` dead-letters invalid JSON (no silent ack);
shared prune selection (weakest first, manuals exempt) on SQLite/Mongo/in-memory;
dead knobs removed (`pool_size`, `schema_auto_migrate`); `short_term_days` and
`llm.max_tokens` wired; `max_tokens_key` for Azure; phantom `[postgres]` extra
dropped; `LlmCapabilityError` exported; `ensure_started` serialized.

## 4. Workstreams

Ordered by dependency and value. Each task lists files and its acceptance gate.
The standing gates: unit+integration suite, mypy strict, ruff, e2e sim
(two models), bench matrix — plus new gates introduced by W2.

### W1 — Accuracy residuals (attacks G1–G4)

**W1.1 Move alias declaration into structured extraction output.**
Regex mining on prose is phrasing-sensitive by construction (G1). The extraction
LLM call already emits entities; extend `ExtractionOutput` with an `aliases`
field (`{subject_token, alias, kind: real_name|nickname}`) so name binding is
LLM-intent with roster-token attribution — the same mechanism that killed
hallucinated-ID attribution (D2). Regexes demote to a deterministic pre-filter
for first-person declarations only. Third-party name guards become schema-level
(alias must bind to a token present in the batch).
*Files: `models/operations.py`, `prompts/extraction.py`, `ingest/extraction.py`,
`ingest/pipeline.py` (alias writes), `identity/aliases.py` (delete mined-regex
paths).* Gate: new golden scenario for third-party name statements; GLM
re-bench passes the surname check.

**W1.2 Shadow reinforcement (G2).** When a batch yields zero/few candidates,
embed the raw messages and reinforce existing facts with cosine ≥
`near_duplicate_threshold`. Embeddings only, no LLM — converts a model
limitation into a deterministic signal.
*Files: `ingest/pipeline.py` (post-extraction hook), `config.py` (one knob,
default on).* Gate: sim "go knowledge reinforced" expectation promotes from
WEAK to hard check.

**W1.3 Deterministic slot contradiction (G4).** Candidates with state-change
phrasing already bypass the category gate; extend to a deterministic supersede
when subject + aspect (category + entity overlap) match and polarity flips
("loves X" / "quit X"). LLM reconcile remains for the genuinely ambiguous band.
*Files: `ingest/reconcile.py`.* Gate: luna's Omaha→Seattle and Red Bull
expectations pass without improve-the-model luck.

**W1.4 Small-batch recall floor (G3).** Prompt-side only: `EXTRACTION_INSTRUCTIONS`
is now in the extraction system message. Strengthen the "durable content ⇒ at
least one operation" instruction with one few-shot pair if variance persists.
No machinery. If variance persists across two model families, accept and
document — do not build retry-storm machinery for it.
*Files: `prompts/extraction.py`.* Gate: extraction-recall expectations stable
across 3 consecutive bench runs.

### W2 — Evaluation & load infrastructure (attacks G5, G6; enables M3 exits)

**W2.1 Load harness (`evals/load/`, currently empty).** ScriptedLLM (free,
deterministic) driving N workers × M subjects × K guilds through the public
API. Measures: commit throughput, queue depth over time, duplicate-commit count
(must be zero), drain cleanliness on `close()`, p95 observe latency. Includes a
multi-process mode (×8 workers on one SQLite/PG file) and a `kill -9` chaos
mode (no lost or duplicated commits after restart).
*Files: `evals/load/runner.py`, `tests/integration/test_load.py` (scaled-down
CI variant).* Gate: PLAN M3 exit — linear worker scaling, zero lost/duplicated
facts under chaos.

**W2.2 Golden set refresh (G5).** Rewrite `contradiction_supersede` to actually
script a contradiction + reconcile decision; add a capability-error scenario
(`LlmCapabilityError` → batch dead-letters, `ExtractionFailed` fires); add the
W1.1 third-party alias scenario. Keep the set deterministic (ScriptedLLM).
*Files: `evals/golden/*.yaml`.* Gate: CI gate stays green; new scenarios fail
when their mechanism is deliberately broken (mutation check).

**W2.3 Bench matrix as a scheduled gate.** `bench_models.py` already produces
machine-readable reports; add a nightly GitHub Action running the matrix
(2 model families minimum per PLAN M2 exit) and diffing hard-check regressions.
*Files: `.github/workflows/bench.yml`.* Gate: attribution ≥99%, fabrication
<0.5% on golden sets across 2 families (M2 exit, finally enforced).

### W3 — Storage scale-out (attacks G7, G13, G14; PLAN M3 core)

**W3.1 Postgres + pgvector adapter.** The conformance suite
(`tests/integration/test_store_conformance.py`) already defines the contract —
the adapter must pass it unchanged. pgvector replaces brute-force scan;
`UPDATE…RETURNING` claim semantics port directly. This is the D1 swap test made
real: if more than the adapter + config changes, the abstraction has leaked.
*Files: `adapters/postgres/` (new), `config.py` (URL scheme), docker-compose
for CI.* Gate: conformance suite + full e2e sim on Postgres; load harness (W2.1)
at 1k guilds × 50 users × 40 msgs/day equivalent.

**W3.2 sqlite-vec evaluation (G7 middle rung).** One spike: does sqlite-vec
extend SQLite's honest limit enough to matter for single-process deployments?
If yes, adopt behind the existing vector port; if marginal, document the limit
and point to W3.1. Timeboxed; do not gold-plate.

**W3.3 Mongo conformance in CI (G14).** Either a mongo container in CI or an
explicitly documented manual release gate. No silent skips.

### W4 — Lifecycle & governance (attacks G10, G11)

**W4.1 CORE exemption on cap prune (G10 residual).** Caps, TTL, and weakest-first
prune already run from `lifecycle/maintenance.py` (off the hot path). Optional:
skip CORE entirely when overflowing, instead of pruning it last.
*Files: `lifecycle/prune.py`.* Gate: over-cap subject with mixed tiers keeps every
CORE fact.

**W4.2 Pending ledger (G11).** Facts in the 0.55–0.7 confidence band land
`pending` (invisible to recall), promote on independent corroboration, expire
otherwise. Directly attacks context pollution — the sim's pollution phases
become its acceptance test.
*Files: `models/facts.py`, `ingest/pipeline.py`, `retrieval/` (status filter).*
Gate: new sim phase — single low-confidence mention invisible, second mention
promotes.

### W5 — Retrieval surface (G12 closed)

Cite-on-use `[mem:N]` ships in `retrieval/injection.py`. No open W5 work.

### W6 — Library professionalism (attacks G15, G16; PLAN M4)

**W6.1 Packaging & release discipline.** Extras already: `[mongo]`, `[discord]`,
`[local-embeddings]`. Remaining: CHANGELOG, CI, deprecation policy (two minor
releases), schema-migration framework. Do not re-add `[postgres]` until W3.1.
Gate: fresh-consumer install-to-first-recall < 5 minutes (M4 exit), verified
by a clean-env script.

**W6.2 Observability bridge (G16).** Thin optional adapter mapping the event
bus + meter to OpenTelemetry spans/metrics. No new core concepts; the event
bus is the seam. Gate: example emits spans; zero overhead when uninstalled.

**W6.3 Docs.** API.md is current; add a "choosing a model" guide (capability
knobs per provider family — the luna lesson), a concurrency/operations guide
(leases, drain, multi-process), and a migration guide stub. Docs are product
surface per AGENTS.md.

## 5. Sequencing

```
Phase 0 (reliability, shipped 2026-08-28): extraction prompt live, loud structured
  failure, prune victim contract, dead knobs, Azure token key, public error types
Phase 1 (correctness, 1–2 weeks):   W1.1 → W1.2 → W1.3 → W1.4
Phase 2 (proof, parallel with 1):   W2.1 → W2.2 → W2.3
Phase 3 (scale, 2–3 weeks):         W3.1 → (W3.2 spike) → W3.3
Phase 4 (governance):               W4.1 (CORE exempt, optional) → W4.2
Phase 5 (surface):                  G12 closed
Phase 6 (release):                  W6.1 → W6.2 → W6.3
```

Phase 1+2 close every correctness gap found since the bench program started
and give us the harnesses that would have caught them earlier. Phase 3 is the
M3 gate. Phases 4–6 are independently shippable; order by user demand.

## 6. Non-goals (carried from RESEARCH.md §6, still rejected)

- Graph database / community detection — cost without payoff at guild scale.
- Append-only memory (mem0 v3) — empirically wrong here; contradictions never
  retire without reconcile.
- Cross-encoder reranker dependency — revisit only if W2.3 numbers show a
  ranking bottleneck.
- Capability latches / silent retry degradation — replaced by declared
  capabilities + loud failure (design rule 2).
- Procedural memory, vision captioning — out of product scope.

## 7. How this document is used

- New work references a W-item and its gate; work without a gate doesn't merge.
- When a field finding generalizes, add the rule to §2 and the gap to §3.
- When an item ships, move it to the §3 closed list with its evidence link.
- Re-plan triggers: a failed gate, a new model family in the bench matrix, or
  a production deployment shape we haven't load-tested.
