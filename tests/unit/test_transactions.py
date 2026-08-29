"""SQLite transaction semantics: atomicity, rollback, and scope serialization."""

from __future__ import annotations

import asyncio

import pytest

from icelake.adapters.sqlite.store import SqliteStore


@pytest.fixture()
async def store():
    s = SqliteStore("sqlite://:memory:")
    await s.setup()
    yield s
    await s.close()


class TestTransactionScope:
    async def test_inner_writes_commit_once_at_scope_exit(self, store) -> None:
        async with store.transaction():
            await store._db.execute(
                "INSERT INTO dm_meta (key, value) VALUES ('a', '1') ON CONFLICT(key) DO NOTHING"
            )
            await store._db.execute(
                "INSERT INTO dm_meta (key, value) VALUES ('b', '2') ON CONFLICT(key) DO NOTHING"
            )
        rows = await store._db.query("SELECT key FROM dm_meta WHERE key IN ('a', 'b')")
        assert {row["key"] for row in rows} == {"a", "b"}

    async def test_failure_rolls_back_all_inner_writes(self, store) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            async with store.transaction():
                await store._db.execute(
                    "INSERT INTO dm_meta (key, value) VALUES ('x', '1') ON CONFLICT(key) DO NOTHING"
                )
                raise RuntimeError("boom")
        row = await store._db.query_one("SELECT key FROM dm_meta WHERE key = 'x'")
        assert row is None  # nothing leaked out of the failed unit of work

    async def test_concurrent_scopes_serialize(self, store) -> None:
        """Two tasks opening units of work must not interleave on one connection."""

        async def write_keys(prefix: str) -> None:
            async with store.transaction():
                for i in range(5):
                    await store._db.execute(
                        "INSERT INTO dm_meta (key, value) VALUES (?, '1') "
                        "ON CONFLICT(key) DO NOTHING",
                        (f"{prefix}-{i}",),
                    )
                    await asyncio.sleep(0)  # yield point: interleaving would deadlock/fail

        await asyncio.gather(write_keys("a"), write_keys("b"))
        rows = await store._db.query(
            "SELECT key FROM dm_meta WHERE key LIKE 'a-%' OR key LIKE 'b-%'"
        )
        assert len(rows) == 10

    async def test_nested_scope_joins_outer_transaction(self, store) -> None:
        async with store.transaction():
            await store._db.execute(
                "INSERT INTO dm_meta (key, value) VALUES ('outer', '1') ON CONFLICT(key) DO NOTHING"
            )
            async with store.transaction():  # joins; no nested BEGIN
                await store._db.execute(
                    "INSERT INTO dm_meta (key, value) VALUES ('inner', '1') "
                    "ON CONFLICT(key) DO NOTHING"
                )
        rows = await store._db.query("SELECT key FROM dm_meta WHERE key IN ('outer', 'inner')")
        assert len(rows) == 2


class TestCommitterAtomicity:
    async def test_failed_graph_write_leaves_no_fact(self) -> None:
        """commit_add is one unit of work: a mid-commit failure rolls the fact back."""
        from datetime import UTC, datetime

        from icelake.config import MemoryConfig
        from icelake.ingest.executor import FactCommitter
        from icelake.ingest.roster import Roster
        from icelake.models.operations import ProposedFact
        from icelake.ports.clock import FixedClock, UlidIdGen

        store = SqliteStore("sqlite://:memory:")
        await store.setup()
        commit = FactCommitter(
            store=store,
            vectors=None,
            embedder=None,
            clock=FixedClock(datetime(2026, 8, 29, tzinfo=UTC)),
            id_gen=UlidIdGen(),
            config=MemoryConfig(),
        )

        original_add_links = store.add_links

        async def failing_add_links(rows):
            await original_add_links(rows)
            raise RuntimeError("simulated graph-write failure")

        store.add_links = failing_add_links  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="simulated"):
            await commit.commit_add(
                proposal=ProposedFact(
                    subject_token="p0",
                    text="alice plays violin in the city orchestra",
                    category="interests",
                    confidence=0.9,
                    source_message_indexes=[1],
                ),
                subject_id="u-alice",
                speaker_id=None,
                guild_id="g1",
                roster=Roster(),
            )
        page = await store.list_facts("g1", subject_id="u-alice")
        assert page.items == ()  # fact insert rolled back with the graph write
        history = await store.get_history("g1", "fct_nonexistent")
        assert history == ()
        await store.close()
