# Migrating to icelake 0.7

The working release candidate is 0.7.2. This document does not imply a PyPI
publication or Git tag.

## Storage and runtime

Mongo production deployments require a replica set for atomic ingestion,
corrections, source invalidation and budget reservations. A single-node replica
set supplies transactions, not high availability. Back up and restore-test before
converting a standalone deployment. SQLite upgrades to schema 5 on open, adding
source identity, revision and original creation time to the observation queue.
Back up the database before opening it with the new version; do not downgrade a
migrated database in place.

Install `icelake[mongo]` for the NumPy-backed Mongo vector adapter. Matching scans
all vectors in the guild/subject scope in bounded batches. This is exact search,
not ANN: cost still grows with the number of candidate vectors. The candidate
batch setting bounds working memory; it is no longer an age cutoff. Retrieval
channels have deadlines and report degraded results rather than pretending an
unavailable search found nothing.

## Evidence and access

Construct `Audience(channel_id=..., requester_id=...)` from trusted transport
metadata. Use a live `ChannelAuthorizer` for cross-channel disclosure. Unknown
permissions fail closed. Use `graph.for_audience(...)` for user-facing graph reads;
control-plane store/fact APIs remain privileged APIs.

Use `visible_facts` and `visible_sources` for transport-owned evidence. Preserve
`PromptContext` when combining memory and web/message sources in `Citations`.
`cite` revalidates fact versions and audience both before and after verification.
Pass `citation_verifier(guild_id=...)` to support paraphrase attribution. Without
one, attribution is conservative. A citation proves provenance, not whether a
speaker's statement was true. Never treat a missing citation as permission to
assert an unsupported claim.

Unbound summaries are intentionally excluded from personal factual context.
Records without original sources cannot become trustworthy merely by upgrading.
Audit them separately; preserve historical data and never manufacture excerpts.

## Observation, corrections and deletion

Pass the actual original message excerpt and author in `SourceRef`; do not put an
extracted fact into `content_snippet`. `MessageEvent.edited_at` must be aware and
represent the Discord edit time. Edits receive distinct observation revisions
while citations retain the original message ID and creation time.

Call `delete_source` for deleted source messages. Out-of-order edits cannot
resurrect a deletion; pending stale observations cannot recommit old facts.
Persist rejected `ObserveReceipt`s in the application outbox for retry. Retrying
an accepted observation is idempotent.

`facts.update` now supersedes the old record and returns the replacement with a
new ID and lineage. Pass correction evidence and an idempotency key, and retain
the returned ID. Remembering an existing fact reinforces it and retains its new
source. Source invalidation conservatively retires a fact if any supporting
source changes; a later extraction can establish a replacement.

## Worker maintenance

Retention uses a transaction and a persisted per-guild completion timestamp, so
restarts and additional workers share the throttle. Workers within one client are
serialized. A conflicting transaction from another client/process rolls back and
remains eligible for its next worker poll; monitor recurring conflicts, not only
queue depth. Explicit `force=True` bypasses the interval, not atomicity.

## Budget ports

Custom `Meter` implementations must implement `reserve` and `settle` using
`BudgetReservation`. Built-in adapters atomically reserve a conservative prompt
allowance before a provider request and settle its actual prompt usage afterward.
Failed or cancelled requests retain the reservation because billing is unknown.
Counters are process-local observability; guild enforcement is store-backed.
The configured limit is prompt tokens, not a universal dollar/output-token cap.
