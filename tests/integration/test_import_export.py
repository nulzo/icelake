"""Export → import round-trip: data integrity across all backends."""

from __future__ import annotations

import pytest

from discord_memory.adapters.in_memory.store import InMemoryStore
from discord_memory.adapters.sqlite.store import SqliteStore
from tests.integration.test_store_conformance import make_fact


@pytest.fixture(params=["in_memory", "sqlite"])
async def source_store(request):
    if request.param == "in_memory":
        s = InMemoryStore()
    else:
        s = SqliteStore("sqlite://:memory:")
    await s.setup()
    yield s
    await s.close()


@pytest.fixture(params=["in_memory", "sqlite"])
async def target_store(request):
    if request.param == "in_memory":
        s = InMemoryStore()
    else:
        s = SqliteStore("sqlite://:memory:")
    await s.setup()
    yield s
    await s.close()


class TestExportImportRoundTrip:
    async def test_facts_survive_round_trip(self, source_store, target_store) -> None:

        fact = make_fact(id="fct_rt1", text="plays bass guitar in a band")
        await source_store.insert_fact(fact)
        await source_store.upsert_entity(
            "g1",
            "bass-guitar",
            "Bass Guitar",
            "concept",
            ("bass",),
        )

        facts, entities, relations = await source_store.export_guild("g1")

        # Import into a different store
        inserted = await target_store.import_guild(facts, entities, relations)
        assert inserted == 1

        loaded = await target_store.get_fact("g1", "fct_rt1")
        assert loaded is not None
        assert loaded.text == "plays bass guitar in a band"
        entity = await target_store.get_entity("g1", "bass-guitar")
        assert entity is not None and entity.name == "Bass Guitar"
