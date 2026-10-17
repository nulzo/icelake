"""Mongo vector index + remaining new-feature coverage (JSON repair, maintenance)."""

from __future__ import annotations

import pytest

from icelake.adapters.mongo import MongoStore
from icelake.adapters.mongo.vectors import MongoVectorIndex
from icelake.ports.vectors import VectorItem


@pytest.fixture()
def vectors(mongo_store: MongoStore) -> MongoVectorIndex:
    return mongo_store.vectors


def item(item_id: str, vec: tuple[float, ...], subject: str | None = "u1") -> VectorItem:
    return VectorItem(id=item_id, guild_id="g1", subject_id=subject, embedding=vec)


class TestMongoVectors:
    async def test_upsert_search_scoping(self, vectors) -> None:
        index = vectors
        await index.upsert(
            (
                item("v1", (1.0, 0.0)),
                item("v2", (0.9, 0.1)),
                item("srv", (1.0, 0.0), subject=None),
            )
        )
        hits = await index.search((1.0, 0.0), guild_id="g1", limit=5)
        assert {h.id for h in hits} >= {"v1", "srv"}
        scoped = await index.search((1.0, 0.0), guild_id="g1", subject_ids=("u1",), limit=5)
        assert "srv" in {h.id for h in scoped} and "v1" in {h.id for h in scoped}
        server_only = await index.search((1.0, 0.0), guild_id="g1", server_only=True, limit=5)
        assert [h.id for h in server_only] == ["srv"]

    async def test_delete_count_and_empty_query(self, vectors) -> None:
        index = vectors
        await index.upsert((item("d1", (0.5, 0.5)),))
        assert await index.count("g1") == 1
        removed = await index.delete(("d1",))
        assert removed == 1
        assert await index.count("g1") == 0
        assert await index.search((), guild_id="g1") == ()

    async def test_other_guild_excluded(self, vectors) -> None:
        index = vectors
        await index.upsert((item("x1", (1.0,)),))
        assert await index.search((1.0,), guild_id="other") == ()
