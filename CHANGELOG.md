# Changelog

All notable changes to icelake are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[PEP 440](https://peps.python.org/pep-0440/).

Release notes are assembled from fragments in `changelog.d/` with
[towncrier](https://towncrier.readthedocs.io/). To add an entry, drop a
`<issue-or-name>.<type>.md` fragment into `changelog.d/` where `<type>` is one
of `security`, `removed`, `deprecated`, `added`, `changed`, `fixed`.

<!-- towncrier release notes start -->

## [0.4.3](https://github.com/nulzo/icelake/tree/v0.4.3) - 2026-09-07

No significant changes.


## [0.4.2](https://github.com/nulzo/icelake/tree/v0.4.2) - 2026-09-07

### Added

- Added `icelake.citations`: a reply-side citation system built on a closed source set with deny-by-default parsing, mirroring how production assistants (ChatGPT, Claude, Perplexity) keep citations reliable. `Citations` seeds from `PromptContext`, registers link sources (`add_source`) and Discord messages (`add_message`) with library-minted `[src:N]` refs, exposes the model-facing `instructions` / `source_list`, and validates with one call — `parse(text, markers=...)` splices provider-anchored offsets, resolves echoed refs, canonicalizes mangled-but-known references, and deletes every citation-shaped token outside the set (invented `[src:9]`, self-written `[1](url)` links, `【…】` artifacts). `parse` returns a structured `ParsedReply` (cleaned `text` + resolved `citations`) so presentation — inline links, footers, or stripped markers — is the consumer's; an `apply(text)` convenience weaves Discord jump links for the common bot path. Also added `MarkerMode`, `CitationSource`, `UsedSource`, `render_fact_list()`, and top-level exports of `Citations`, `MarkerMode`, `ParsedReply`, `UsedSource`, `CitationSource`, `message_url`, and `render_fact_list`. ([#citation-registry](https://github.com/nulzo/icelake/issues/citation-registry))

### Changed

- `PromptContext.apply_citations` now delegates to `Citations.apply` — a single parsing implementation for all citation handling. Behavior tightens deny-by-default: model-written `[N](url)` links whose target is not a registered source and `【…】` artifacts are now removed, while word-attached prose like `array[0]` is preserved. ([#apply-citations-registry](https://github.com/nulzo/icelake/issues/apply-citations-registry))


## [0.4.1](https://github.com/nulzo/icelake/tree/v0.4.1) - 2026-09-07

No significant changes.


## [0.4.0](https://github.com/nulzo/icelake/tree/v0.4.0) - 2026-09-07

### Added

- Closed-ID citation objects plus an optional OpenRouter L2 reranker after RRF.
  The generator may only echo advertised `[mem:N]` tags (or structured `fact_id`
  claims); consumers call `PromptContext.resolve_used` and weave the returned
  objects. Banter emits no IDs, so `used` is empty and no links are inserted.
  Rerank degrades to hybrid order on failure, timeout, or score-arity mismatch.
  `prompt_context` now pair-intersects every combination of mentions and thread
  participants and enables graph-hop recall whenever anyone else is in the turn.
  `graph.shared_attributions` returns both members' stances over shared entities
  so consumers can show agreement and disagreement without a second fetch. ([#citation-objects-reranker](https://github.com/nulzo/icelake/issues/citation-objects-reranker))


## [0.3.4](https://github.com/nulzo/icelake/tree/v0.3.4) - 2026-09-06

### Fixed

- Relation endpoints named after a guild member now collapse to that member's user
  node at write time (no more person-entity twins), and every person-facing graph
  query (`relations_of`, `between`, `shared`, `similar_users`, `entity_stances`,
  `neighbors`) follows `entity.linked_user_id` so pre-existing twins still resolve
  to the member. Reads go through a single seam (`GraphApi._incident`) that unions
  the member's user node with every entity twin and collapses the result, so edges
  stored against a twin (a member mentioned by name before they spoke) surface on
  that member's star. Added `graph.shared(a, b)`: the identity-collapsed,
  weight-ranked intersection of two members' entity edges, with polarity
  preserved. Bot IDs are no longer written to `FactRecord.related_user_ids`. ([#identity-collapse](https://github.com/nulzo/icelake/issues/identity-collapse))


## [0.3.3](https://github.com/nulzo/icelake/tree/v0.3.3) - 2026-08-31

### Added

- Documented the extraction-model benchmark in the README: ranked OpenRouter
  models on the e2e sim (quality, spend, latency) with a fixed weighted score
  so later runs can be appended without rescaling. Includes the 2026-08-30
  challenger round (GPT-4o-mini, GPT-4.1-mini, Gemini 2.5 Flash, Qwen3 32B). ([#model-bench-readme](https://github.com/nulzo/icelake/issues/model-bench-readme))
- Documented the name-in-prose lookup pattern ("what do you know about X?" with no
  @mention): a new README section explains that `prompt_context` is mention-keyed by
  design and shows the resolve-then-strict-fetch flow (`identity.resolve` →
  `facts.list_for_subject`, never guessing on ambiguity) across three entry points —
  slash command (zero LLM), a structured-output router, and native function calling.
  Added `examples/name_lookup_tool.py`, a runnable LLM-free demo of the shared
  handler, and wired the pattern into `examples/omni_style_bot.py` (router +
  `/memory lookup`). ([#name-lookup-docs](https://github.com/nulzo/icelake/issues/name-lookup-docs))
- `RetrievalConfig.top_k` and `RetrievalConfig.max_per_subject` now bound the
  `prompt_context` hot path (previously hardcoded to the `RecallQuery` defaults
  of 8 total / 4 per subject), so consumers can widen per-turn memory injection
  from configuration. ([#prompt-context-caps](https://github.com/nulzo/icelake/issues/prompt-context-caps))

### Changed

- Bench matrix is a per-model OpenRouter param map (reasoning, temperature,
  structured_outputs) instead of one global flag. LlmConfig accepts
  `reasoning=none` to disable thinking on models that allow it. ([#bench-model-params](https://github.com/nulzo/icelake/issues/bench-model-params))


## [0.3.2](https://github.com/nulzo/icelake/tree/v0.3.2) - 2026-08-30

### Added

- Added `lifecycle.decay_stability_days` (default 7.0), the Ebbinghaus timescale:
  retention is now exp(-days / (strength x stability)), so a one-off fact survives
  ~3 weeks instead of ~3 days. Forgetting policy moved into a single pure selector
  (`lifecycle.select_forgotten_facts`) shared by all store adapters — mongo/sqlite
  no longer carry private copies of the decay math, and the sweep is one bulk
  UPDATE instead of per-fact writes. ([#decay-stability](https://github.com/nulzo/icelake/issues/decay-stability))


## [0.3.1](https://github.com/nulzo/icelake/tree/v0.3.1) - 2026-08-29

No significant changes.


## [0.3.0](https://github.com/nulzo/icelake/tree/v0.3.0) - 2026-08-29

No significant changes.


## [0.2.0](https://github.com/nulzo/icelake/tree/v0.2.0) - 2026-08-29

No significant changes.
