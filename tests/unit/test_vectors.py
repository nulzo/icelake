"""Vector index conformance: scope filtering happens before the candidate cap."""

from __future__ import annotations

import pytest

from icelake.adapters.in_memory.vectors import InMemoryVectorIndex
from icelake.adapters.sqlite.connection import SqliteConnection
from icelake.adapters.sqlite.vectors import SqliteVectorIndex
from icelake.ports.vectors import VectorItem


@pytest.fixture(params=["in_memory", "sqlite"])
async def vectors(request):
    if request.param == "in_memory":
        index = InMemoryVectorIndex()
    else:
        conn = SqliteConnection("sqlite://:memory:")
        await conn.connect()
        index = SqliteVectorIndex(conn)
    await index.setup()
    yield index


def _item(fact_id: str, subject_id: str | None, embedding: tuple[float, ...]) -> VectorItem:
    return VectorItem(
        id=fact_id,
        guild_id="g1",
        subject_id=subject_id,
        embedding=embedding,
    )


async def test_subject_filter_applies_before_cap(vectors) -> None:
    """A busy guild must not starve a quiet member's semantic hits. The cap
    applies to the already-filtered candidate set, not the whole guild."""
    query = (1.0, 0.0)
    # Fill the cap with OTHER subjects' vectors, all orthogonal to the query.
    filler = tuple(_item(f"fct_fill_{i}", f"u-other-{i}", (0.0, 1.0)) for i in range(10))
    # The target fact matches the query perfectly but is the oldest row.
    target = _item("fct_target", "u-quiet", (1.0, 0.0))
    await vectors.upsert((target, *filler))

    hits = await vectors.search(
        query,
        guild_id="g1",
        subject_ids=("u-quiet",),
        limit=5,
        candidate_cap=5,  # smaller than the filler set
    )
    assert any(hit.id == "fct_target" for hit in hits)


async def test_server_only_excludes_subject_vectors(vectors) -> None:
    await vectors.upsert(
        (
            _item("fct_user", "u1", (1.0, 0.0)),
            _item("fct_server", None, (1.0, 0.0)),
        )
    )
    hits = await vectors.search((1.0, 0.0), guild_id="g1", server_only=True, limit=10)
    assert tuple(hit.id for hit in hits) == ("fct_server",)
