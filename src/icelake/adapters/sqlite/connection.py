"""SQLite connection management, PRAGMAs, and schema migration."""

from __future__ import annotations

import asyncio
import contextlib
import json
import sqlite3
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from icelake.concurrency import TaskLock as _TaskLock
from icelake.errors import StorageUnavailableError

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS dm_projection_jobs (
    guild_id TEXT NOT NULL,
    fact_id TEXT PRIMARY KEY,
    version INTEGER NOT NULL,
    embedding TEXT,
    next_attempt_at TEXT
);

CREATE TABLE IF NOT EXISTS dm_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dm_messages (
    message_id TEXT PRIMARY KEY,
    guild_id TEXT NOT NULL,
    channel_id TEXT NOT NULL DEFAULT '',
    author_id TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    content TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    author_username TEXT NOT NULL DEFAULT '',
    author_display_name TEXT NOT NULL DEFAULT '',
    author_is_bot INTEGER NOT NULL DEFAULT 0,
    mention_ids TEXT NOT NULL DEFAULT '[]',
    governance TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    lease_owner TEXT,
    lease_until TEXT
);
CREATE INDEX IF NOT EXISTS ix_dm_messages_due
    ON dm_messages (guild_id, subject_key, status, created_at);
CREATE TABLE IF NOT EXISTS dm_facts (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    guild_id TEXT NOT NULL,
    subject_id TEXT,
    text TEXT NOT NULL,
    text_normalized TEXT NOT NULL,
    category TEXT NOT NULL,
    confidence REAL NOT NULL,
    tier TEXT NOT NULL,
    scope TEXT NOT NULL,
    attribution TEXT NOT NULL DEFAULT '{}',
    occurrences INTEGER NOT NULL DEFAULT 1,
    strength REAL NOT NULL DEFAULT 1.0,
    last_reinforced_at TEXT,
    created_at TEXT,
    updated_at TEXT,
    observed_at TEXT,
    valid_from TEXT,
    valid_until TEXT,
    supersedes_id TEXT,
    superseded_by_id TEXT,
    citations TEXT NOT NULL DEFAULT '[]',
    related_user_ids TEXT NOT NULL DEFAULT '[]',
    entity_slugs TEXT NOT NULL DEFAULT '[]',
    tags TEXT NOT NULL DEFAULT '[]',
    expires_at TEXT,
    version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_dm_facts_subject ON dm_facts (guild_id, subject_id);
CREATE INDEX IF NOT EXISTS ix_dm_facts_norm
    ON dm_facts (guild_id, subject_id, text_normalized);
CREATE INDEX IF NOT EXISTS ix_dm_facts_strength ON dm_facts (guild_id, strength DESC);
CREATE VIRTUAL TABLE IF NOT EXISTS dm_facts_fts USING fts5(text);
CREATE TABLE IF NOT EXISTS dm_llm_cache (
    key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS dm_history (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    at TEXT NOT NULL,
    kind TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    fact_version INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS ix_dm_history ON dm_history (fact_id, seq);
CREATE TABLE IF NOT EXISTS dm_aliases (
    guild_id TEXT NOT NULL,
    alias_norm TEXT NOT NULL,
    user_id TEXT NOT NULL,
    source TEXT NOT NULL,
    weight REAL NOT NULL,
    updated_at TEXT,
    PRIMARY KEY (guild_id, alias_norm, user_id)
);
CREATE INDEX IF NOT EXISTS ix_dm_aliases_weight ON dm_aliases (guild_id, alias_norm, weight DESC);
CREATE TABLE IF NOT EXISTS dm_links (
    memory_id TEXT NOT NULL,
    guild_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    node_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    created_at TEXT,
    PRIMARY KEY (memory_id, node_type, node_id, kind)
);
CREATE INDEX IF NOT EXISTS ix_dm_links_node ON dm_links (guild_id, node_type, node_id);
CREATE TABLE IF NOT EXISTS dm_relations (
    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id TEXT NOT NULL,
    src_type TEXT NOT NULL,
    src_id TEXT NOT NULL,
    dst_type TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    verb TEXT NOT NULL,
    polarity TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0,
    occurrences INTEGER NOT NULL DEFAULT 1,
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_ids TEXT NOT NULL DEFAULT '[]',
    valid_from TEXT,
    valid_until TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_dm_relations_active
    ON dm_relations (guild_id, src_type, src_id, dst_type, dst_id, verb)
    WHERE valid_until IS NULL;
CREATE INDEX IF NOT EXISTS ix_dm_relations_dst ON dm_relations (guild_id, dst_type, dst_id);
CREATE TABLE IF NOT EXISTS dm_entities (
    guild_id TEXT NOT NULL,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'concept',
    aliases_json TEXT NOT NULL DEFAULT '[]',
    fact_count INTEGER NOT NULL DEFAULT 0,
    linked_user_id TEXT,
    summary TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (guild_id, slug)
);
CREATE TABLE IF NOT EXISTS dm_entity_aliases (
    guild_id TEXT NOT NULL,
    alias_norm TEXT NOT NULL,
    slug TEXT NOT NULL,
    PRIMARY KEY (guild_id, alias_norm)
);
CREATE TABLE IF NOT EXISTS dm_summaries (
    guild_id TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    text TEXT NOT NULL,
    generated_at TEXT,
    source_fact_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, subject_key)
);
CREATE TABLE IF NOT EXISTS dm_optouts (
    guild_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    PRIMARY KEY (guild_id, user_id)
);
CREATE TABLE IF NOT EXISTS dm_governance (
    guild_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    opted_out INTEGER NOT NULL DEFAULT 0,
    generation INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, user_id)
);
CREATE TABLE IF NOT EXISTS dm_cursors (
    guild_id TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (guild_id, key)
);
CREATE TABLE IF NOT EXISTS dm_budgets (
    guild_id TEXT NOT NULL,
    period_key TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (guild_id, period_key)
);
CREATE TABLE IF NOT EXISTS dm_batch_leases (
    guild_id TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    owner TEXT NOT NULL,
    lease_until TEXT NOT NULL,
    PRIMARY KEY (guild_id, subject_key)
);
CREATE INDEX IF NOT EXISTS ix_dm_messages_lease
    ON dm_messages (lease_until) WHERE status = 'claimed';
CREATE INDEX IF NOT EXISTS ix_dm_messages_guild_created
    ON dm_messages (guild_id, created_at DESC);
"""


def dumps(value: object) -> str:
    return json.dumps(value, default=str)


def iso(moment: Any) -> str | None:
    """Serialize datetimes as ISO-8601 UTC strings."""
    if moment is None:
        return None
    return str(moment.isoformat()) if hasattr(moment, "isoformat") else str(moment)


Params = tuple[object, ...]


class SqliteConnection:
    """Async facade over a single-writer SQLite connection run off-loop."""

    def __init__(self, url: str) -> None:
        path = url.removeprefix("sqlite:///").removeprefix("sqlite://")
        if not path or path == ":memory:":
            path = ":memory:"
        else:
            parent = Path(path).parent
            if str(parent) not in {"", "."}:
                Path(parent).mkdir(parents=True, exist_ok=True)
        self._path = path
        self._conn: sqlite3.Connection | None = None
        self._lock = _TaskLock()
        # While a transaction_scope is open, per-statement auto-commit must be
        # suppressed or every inner write would commit the unit of work early.
        self._txn_open = False

    async def connect(self) -> None:
        if self._conn is not None:
            return

        def _open() -> sqlite3.Connection:
            conn = sqlite3.connect(
                self._path,
                timeout=0.0,
                isolation_level=None,
                check_same_thread=False,
            )
            conn.row_factory = sqlite3.Row
            return conn

        self._conn = await asyncio.to_thread(_open)
        await self._execute("PRAGMA journal_mode=WAL")
        await self._execute("PRAGMA synchronous=FULL")
        await self._execute("PRAGMA foreign_keys=ON")
        await self._execute("PRAGMA busy_timeout=5000")

    async def ensure_schema(self) -> None:
        meta = await self.query_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dm_meta'"
        )
        if meta is not None:
            version = await self.query_one("SELECT value FROM dm_meta WHERE key='schema_version'")
            if version is not None and int(version["value"]) > SCHEMA_VERSION:
                raise StorageUnavailableError("database schema is newer than this library")
        await self._executescript(_SCHEMA)
        async with self.transaction_scope():
            job_columns = await self.query("PRAGMA table_info(dm_projection_jobs)")
            if not any(row["name"] == "next_attempt_at" for row in job_columns):
                await self.execute("ALTER TABLE dm_projection_jobs ADD COLUMN next_attempt_at TEXT")
            if not any(row["name"] == "embedding" for row in job_columns):
                await self.execute("ALTER TABLE dm_projection_jobs ADD COLUMN embedding TEXT")
            await self.execute(
                "CREATE INDEX IF NOT EXISTS ix_dm_projection_due "
                "ON dm_projection_jobs(next_attempt_at)"
            )
            summary_columns = await self.query("PRAGMA table_info(dm_summaries)")
            if not any(row["name"] == "provenance" for row in summary_columns):
                await self.execute(
                    "ALTER TABLE dm_summaries ADD COLUMN provenance TEXT NOT NULL DEFAULT '{}'"
                )
            fact_columns = await self.query("PRAGMA table_info(dm_facts)")
            if not any(row["name"] == "governance" for row in fact_columns):
                await self.execute(
                    "ALTER TABLE dm_facts ADD COLUMN governance TEXT NOT NULL DEFAULT '{}'"
                )
            message_columns = await self.query("PRAGMA table_info(dm_messages)")
            if not any(row["name"] == "governance" for row in message_columns):
                await self.execute(
                    "ALTER TABLE dm_messages ADD COLUMN governance TEXT NOT NULL DEFAULT '{}'"
                )
            await self.execute(
                "INSERT INTO dm_meta (key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    async def backup_to(self, destination: Path) -> None:
        """Native online snapshot, serialized with writes and settled on cancellation."""
        await self._lock.acquire()
        try:
            if self._txn_open:
                raise RuntimeError("cannot back up inside a transaction")
            source = self._require_conn()

            def copy() -> None:
                with destination.open("xb"):
                    pass
                destination.chmod(0o600)
                with contextlib.closing(sqlite3.connect(destination)) as target:
                    source.backup(target)
                    if target.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                        raise StorageUnavailableError("snapshot failed SQLite integrity check")

            await self._run(copy)
        finally:
            self._lock.release()

    async def close(self) -> None:
        await self._lock.acquire()
        try:
            if self._conn is not None:
                connection = self._conn
                try:
                    await self._run(connection.close)
                finally:
                    self._conn = None
        finally:
            self._lock.release()

    async def execute(self, sql: str, params: Params = ()) -> None:
        await self._execute(sql, params)

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageUnavailableError(
                "SQLite connection not open — call setup()/start() first",
            )
        return self._conn

    async def _run[T](self, operation: Callable[[], T]) -> T:
        """Settle off-thread work before releasing the connection on cancellation."""
        task = asyncio.create_task(asyncio.to_thread(operation))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Cancelling an await cannot stop SQLite's worker thread. Keep the
            # lock until it finishes, including when shutdown cancels us twice.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not task.cancelled():
                task.exception()
            raise

    async def query(self, sql: str, params: Params = ()) -> list[sqlite3.Row]:
        conn = self._require_conn()

        def execute() -> list[sqlite3.Row]:
            with contextlib.closing(conn.execute(sql, tuple(params))) as cursor:
                return list(cursor.fetchall())

        await self._lock.acquire()
        try:
            return await self._run(execute)
        finally:
            self._lock.release()

    async def query_one(self, sql: str, params: Params = ()) -> sqlite3.Row | None:
        rows = await self.query(sql, params)
        return rows[0] if rows else None

    async def execute_returning(self, sql: str, params: Params = ()) -> list[sqlite3.Row]:
        """Execute and fetch under one lock; SQLite autocommits outside a transaction."""
        return await self.query(sql, params)

    async def _execute(self, sql: str, params: Params = ()) -> None:
        await self.query(sql, params)

    async def _executescript(self, script: str) -> None:
        conn = self._require_conn()

        def execute() -> None:
            with contextlib.closing(conn.executescript(script)):
                pass

        await self._lock.acquire()
        try:
            await self._run(execute)
        finally:
            self._lock.release()

    @contextlib.asynccontextmanager
    async def transaction_scope(self) -> AsyncIterator[SqliteConnection]:
        """Async CM wrapping BEGIN IMMEDIATE / COMMIT / ROLLBACK.

        Holds the connection lock for the whole unit of work (single-writer
        serialization) and suppresses per-statement commits inside the scope.
        A nested scope joins the open transaction — savepoint semantics are
        deliberately not offered; keep units of work flat.
        """
        conn = self._require_conn()
        if self._txn_open and self._lock.held_by_current_task():
            yield self
            return
        await self._lock.acquire()
        self._txn_open = True
        try:
            await self._execute("BEGIN IMMEDIATE")
            yield self
            await self._execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                await self._execute("ROLLBACK")
            raise
        finally:
            self._txn_open = False
            self._lock.release()

    async def transaction(self, statements: list[tuple[str, Params]]) -> None:
        """Apply a group of writes atomically, joining an existing unit of work."""
        async with self.transaction_scope():
            for sql, params in statements:
                await self.execute(sql, params)
