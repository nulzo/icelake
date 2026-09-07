Closed-ID citation objects plus an optional OpenRouter L2 reranker after RRF.
The generator may only echo advertised `[mem:N]` tags (or structured `fact_id`
claims); consumers call `PromptContext.resolve_used` and weave the returned
objects. Banter emits no IDs, so `used` is empty and no links are inserted.
Rerank degrades to hybrid order on failure, timeout, or score-arity mismatch.
`prompt_context` now pair-intersects every combination of mentions and thread
participants and enables graph-hop recall whenever anyone else is in the turn.
`graph.shared_attributions` returns both members' stances over shared entities
so consumers can show agreement and disagreement without a second fetch.
