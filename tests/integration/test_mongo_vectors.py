"""Mongo vector index + remaining new-feature coverage (JSON repair, maintenance)."""

from __future__ import annotations

import socket

import pytest

pytest.importorskip("pymongo")


def _mongo_available() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 27017), timeout=0.3):
            return True
    except OSError:
        return False


if not _mongo_available():
    pytest.skip("no MongoDB at localhost:27017", allow_module_level=True)

from icelake.adapters.mongo.vectors import MongoVectorIndex  # noqa: E402
from icelake.ports.vectors import VectorItem  # noqa: E402


@pytest.fixture()
async def vectors():
    from pymongo import AsyncMongoClient

    client = AsyncMongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=3000)
    db = client["icelake_vec_test"]
    await db["dm_vectors"].delete_many({})
    index = MongoVectorIndex(db)
    await index.setup()
    yield index, client
    await db["dm_vectors"].delete_many({})
    await client.close()


def item(item_id: str, vec: tuple[float, ...], subject: str | None = "u1") -> VectorItem:
    return VectorItem(id=item_id, guild_id="g1", subject_id=subject, embedding=vec)


class TestMongoVectors:
    async def test_upsert_search_scoping(self, vectors) -> None:
        index, _client = vectors
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
        index, _client = vectors
        await index.upsert((item("d1", (0.5, 0.5)),))
        assert await index.count("g1") == 1
        removed = await index.delete(("d1",))
        assert removed == 1
        assert await index.count("g1") == 0
        assert await index.search((), guild_id="g1") == ()

    async def test_other_guild_excluded(self, vectors) -> None:
        index, _client = vectors
        await index.upsert((item("x1", (1.0,)),))
        assert await index.search((1.0,), guild_id="other") == ()
