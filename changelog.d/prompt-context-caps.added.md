`RetrievalConfig.top_k` and `RetrievalConfig.max_per_subject` now bound the
`prompt_context` hot path (previously hardcoded to the `RecallQuery` defaults
of 8 total / 4 per subject), so consumers can widen per-turn memory injection
from configuration.
