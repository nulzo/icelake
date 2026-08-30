Added `lifecycle.decay_stability_days` (default 7.0), the Ebbinghaus timescale:
retention is now exp(-days / (strength x stability)), so a one-off fact survives
~3 weeks instead of ~3 days. Forgetting policy moved into a single pure selector
(`lifecycle.select_forgotten_facts`) shared by all store adapters — mongo/sqlite
no longer carry private copies of the decay math, and the sweep is one bulk
UPDATE instead of per-fact writes.
