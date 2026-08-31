# Changelog

All notable changes to icelake are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[PEP 440](https://peps.python.org/pep-0440/).

Release notes are assembled from fragments in `changelog.d/` with
[towncrier](https://towncrier.readthedocs.io/). To add an entry, drop a
`<issue-or-name>.<type>.md` fragment into `changelog.d/` where `<type>` is one
of `security`, `removed`, `deprecated`, `added`, `changed`, `fixed`.

<!-- towncrier release notes start -->

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
