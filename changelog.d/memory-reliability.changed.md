Make citation support explicit: exact restatements are cited without provider calls;
paraphrases require an injected CitationVerifier. Add a bounded structured LLM verifier,
validate rendered spans and URLs, and remove unsupported claims about citation accuracy.

Harden SQLite cancellation, mutation transactions, queue ownership and reclaim across
adapters, consent at extraction/reconciliation boundaries, Mongo bulk generation fences,
purge/export handling of third-party evidence, and temporal recall. Similarity alone no
longer reinforces memory; unresolved decisions retain their input in dead letters.

Bound the full injection estimate, validate embedding responses and URL config values,
reduce the default embedding cache to 2,048 entries, deduplicate batched embeddings,
serialize local model calls, and read each graph-pair participant once.

Migration: remove extraction.near_duplicate_threshold, accept get_summary(..., now=None)
in custom stores, and acknowledge successful work before release_key(). See README for
transaction boundaries and remaining production limitations.
