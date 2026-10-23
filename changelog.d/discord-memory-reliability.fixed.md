Preserve original source excerpts, authors, timestamps and every supporting
message through retrieval and citation weaving. Mixed citation registries retain
memory context and revalidate versions, consent and channel access after support
verification. Unbound summaries no longer enter factual context. Related words
and vector similarity are retrieval signals, not proof of a statement.

Source edits and deletions invalidate derived facts and fence stale ingestion.
Corrections atomically supersede facts with new evidence and retry receipts.
Merged facts preserve both inputs' provenance; reinforcement retains new sources.

Mongo lexical search filters before limiting; Mongo and SQLite vectors scan the
complete requested scope in bounded batches without dropping older candidates.
Mongo uses NumPy off the event loop. Prompt-budget reservations are transactional
across processes, and meter snapshots expose operational counters.

Remove edit-distance rejection of faithful third-person facts. A concise personal
fact may legitimately closely match its source; extracted first-person statements
with an unclear subject frame remain rejected.

Run retention sweeps in a transaction with the existing guild commit fence and a
durable completion timestamp. Competing workers cannot retire the same version
twice or discard a concurrent reinforcement; failed sweeps roll back and remain
eligible for retry. Reference-store retirement now advances versions consistently.
