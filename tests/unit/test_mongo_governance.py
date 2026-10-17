"""Mongo adapter governance behavior against an in-process fake database.

Standalone Mongo has no cross-document transactions, so the fence must come
from durable generation stamps checked on every read. These tests pin that
behavior without any live server.
"""

from __future__ import annotations

from typing import Any

from icelake.adapters.mongo import MongoStore
from icelake.models.facts import FactRecord


class FakeCollection:
    """Minimal find_one/find/insert/delete stand-in over a dict of docs."""

    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self._rows = rows

    async def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        if "_id" in query and "guild_id" in query:
            row = self._rows.get(query["_id"])
            if row is not None and row["guild_id"] == query["guild_id"]:
                return row
            return None
        raise AssertionError(f"unsupported fake query: {query}")

    def find(self, query: dict[str, Any], projection: dict[str, Any] | None = None):
        guild = query["guild_id"]
        return FakeCursor([row for row in self._rows.values() if row["guild_id"] == guild])

    async def insert_one(self, doc: dict[str, Any]) -> None:
        self._rows[doc["_id"]] = doc


class FakeCursor:
    """Chained find() cursor stub: to_list only."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def sort(self, *_args: Any, **_kwargs: Any) -> FakeCursor:
        return self

    def limit(self, _count: int) -> FakeCursor:
        return self

    def __aiter__(self) -> FakeCursor:
        return self

    async def __anext__(self) -> dict[str, Any]:
        if not self._rows:
            raise StopAsyncIteration
        return self._rows.pop(0)

    async def to_list(self, length: int | None = None) -> list[dict[str, Any]]:
        return self._rows if length is None else self._rows[:length]


class FakeDatabase(dict[str, Any]):
    def __init__(self) -> None:
        super().__init__()
        self["dm_facts"] = FakeCollection({})
        self["dm_governance"] = FakeCollection({})

    def __getitem__(self, name: str) -> Any:
        return super().__getitem__(name)


class FencedMongoStore:
    """Exposes only the fenced fact reads over a fake database."""

    def __init__(self, db: FakeDatabase) -> None:
        self.db = db
        self._ACTIVE = {"valid_until": None, "superseded_by_id": None}

    governance_generations = MongoStore.governance_generations
    get_facts = MongoStore.get_facts
    get_fact = MongoStore.get_fact
    list_facts = MongoStore.list_facts
    top_strength_facts = MongoStore.top_strength_facts
    consent_state = MongoStore.consent_state


def test_mongo_fact_reads_fence_pre_purge_rows() -> None:
    db = FakeDatabase()
    store = FencedMongoStore(db)
    db["dm_governance"]._rows["u"] = {
        "_id": "g:u",
        "guild_id": "g",
        "user_id": "u",
        "opted_out": False,
        "generation": 2,
    }
    stale = FactRecord(id="f-old", guild_id="g", subject_id="u", text="stale", governance={"u": 1})
    fresh = FactRecord(id="f-new", guild_id="g", subject_id="u", text="fresh", governance={"u": 2})
    for record in (stale, fresh):
        db["dm_facts"]._rows[record.id] = {
            **{
                "_id": record.id,
                "guild_id": record.guild_id,
                "subject_id": record.subject_id,
                "text": record.text,
                "text_normalized": "",
                "category": "general",
                "confidence": 1.0,
                "tier": "short_term",
                "scope": "user",
                "attribution": {},
                "occurrences": 1,
                "strength": 1.0,
                "citations": [],
                "related_user_ids": [],
                "entity_slugs": [],
                "tags": [],
                "version": 1,
                "governance": record.governance,
            },
            "valid_until": None,
            "superseded_by_id": None,
        }
    import asyncio

    async def run() -> None:
        assert [r.id for r in await store.get_facts("g", ("f-old", "f-new"))] == ["f-new"]
        assert await store.get_fact("g", "f-old") is None
        assert (await store.get_fact("g", "f-new")) is not None
        page = await store.list_facts("g", subject_id="u")
        assert [r.id for r in page.items] == ["f-new"]
        tops = await store.top_strength_facts("g", subject_ids=("u",))
        assert [r.id for r in tops] == ["f-new"]

    asyncio.run(run())
