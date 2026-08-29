"""Extraction context assembly: anchors + relevant prior facts, budgeted."""

from __future__ import annotations

from icelake.ports.store import MemoryStore
from icelake.ports.vectors import VectorIndex

MAX_ANCHOR_FACTS = 6
MAX_RELEVANT_FACTS = 12


def render_memory_lines(records: tuple[object, ...]) -> str:
    """Format fact records as compact context lines (deduped by identity)."""
    seen: set[str] = set()
    lines: list[str] = []
    for record in records:
        fact_id = getattr(record, "id", "")
        text = getattr(record, "text", "")
        if not fact_id or fact_id in seen or not text:
            continue
        seen.add(fact_id)
        lines.append(f"- [{fact_id}] {text}")
    return "\n".join(lines)


async def build_extraction_context(
    *,
    store: MemoryStore,
    vectors: VectorIndex | None,
    guild_id: str,
    subject_id: str | None,
    batch_text: str,
    batch_embedding: tuple[float, ...] | None,
) -> str:
    """Anchor core facts plus semantically relevant prior memories for the LLM."""
    anchors = await store.top_strength_facts(
        guild_id,
        subject_ids=(subject_id,) if subject_id else None,
        server_only=subject_id is None,
        limit=MAX_ANCHOR_FACTS,
    )
    relevant: tuple[object, ...] = ()
    if vectors is not None and batch_embedding is not None:
        scope_ids = (subject_id,) if subject_id else None
        hits = await vectors.search(
            batch_embedding,
            guild_id=guild_id,
            subject_ids=scope_ids,
            server_only=subject_id is None,
            limit=MAX_RELEVANT_FACTS,
            candidate_cap=100,
        )
        records = await store.get_facts(guild_id, tuple(h.id for h in hits))
        # get_facts does not guarantee order (Mongo $in) and may drop missing
        # ids; pair scores to records by id, never by position.
        by_id = {record.id: record for record in records}
        relevant = tuple(
            by_id[hit.id]
            for hit in hits
            if hit.id in by_id and by_id[hit.id].is_active and hit.score >= 0.5
        )
    combined = list(anchors) + list(relevant)
    return render_memory_lines(tuple(combined))
