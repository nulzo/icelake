"""Extraction context assembly: id-safe pairing of vector hits to facts."""

from __future__ import annotations

from datetime import UTC, datetime

from icelake.adapters.in_memory.store import InMemoryStore
from icelake.ingest.context_builder import build_extraction_context
from icelake.models.facts import FactRecord
from icelake.ports.vectors import VectorHit


class _FakeVectors:
    """Returns hits in a fixed order regardless of what the store has."""

    def __init__(self, hits: tuple[VectorHit, ...]) -> None:
        self._hits = hits

    async def search(self, *args, **kwargs) -> tuple[VectorHit, ...]:
        return self._hits


def _fact(fact_id: str, text: str) -> FactRecord:
    now = datetime.now(UTC)
    return FactRecord(
        id=fact_id,
        guild_id="g1",
        subject_id="u1",
        text=text,
        created_at=now,
        updated_at=now,
        observed_at=now,
        valid_from=now,
    )


async def test_missing_fact_id_does_not_shift_scores() -> None:
    """The store may drop a missing id; the remaining records must not slide
    into the wrong hit's score slot."""
    store = InMemoryStore()
    await store.insert_fact(_fact("fct_high", "high score fact"))
    await store.insert_fact(_fact("fct_low", "low score fact"))
    vectors = _FakeVectors(
        (
            VectorHit(id="fct_high", score=0.9),
            VectorHit(id="fct_gone", score=0.8),  # not in the store
            VectorHit(id="fct_low", score=0.6),
        )
    )
    block = await build_extraction_context(
        store=store,
        vectors=vectors,
        guild_id="g1",
        subject_id="u1",
        batch_text="batch",
        batch_embedding=(0.1, 0.2),
    )
    # Both surviving facts render; the missing one is skipped, not mis-paired.
    assert "high score fact" in block
    assert "low score fact" in block
    assert "fct_gone" not in block


async def test_inactive_fact_is_dropped() -> None:
    store = InMemoryStore()
    await store.insert_fact(_fact("fct_dead", "invalidated fact"))
    record = await store.get_fact("g1", "fct_dead")
    assert record is not None
    await store.transition_fact(
        "g1", "fct_dead", valid_until=datetime.now(UTC), updated_at=datetime.now(UTC)
    )
    vectors = _FakeVectors((VectorHit(id="fct_dead", score=0.9),))
    block = await build_extraction_context(
        store=store,
        vectors=vectors,
        guild_id="g1",
        subject_id="u1",
        batch_text="batch",
        batch_embedding=(0.1, 0.2),
    )
    assert "invalidated fact" not in block
