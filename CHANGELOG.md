# Changelog

All notable changes to icelake are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[PEP 440](https://peps.python.org/pep-0440/).

Release notes are assembled from fragments in `changelog.d/` with
[towncrier](https://towncrier.readthedocs.io/). To add an entry, drop a
`<issue-or-name>.<type>.md` fragment into `changelog.d/` where `<type>` is one
of `security`, `removed`, `deprecated`, `added`, `changed`, `fixed`.

<!-- towncrier release notes start -->

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
