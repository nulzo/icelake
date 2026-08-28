"""SQLite IngestQueue: atomic claims, leases, dead-letters (B2/B3/B4 fixes)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

from discord_memory.adapters.sqlite.connection import SqliteConnection, iso
from discord_memory.models.common import ensure_aware
from discord_memory.ports.queue import BatchKey, ClaimOutcome, MessageStatus, StoredMessage

SERVER_SUBJECT_KEY = "__server__"


def _message_from_row(row: sqlite3.Row) -> StoredMessage:
    return StoredMessage(
        message_id=row["message_id"],
        guild_id=row["guild_id"],
        channel_id=row["channel_id"] or "",
        author_id=row["author_id"],
        subject_key=row["subject_key"],
        content=row["content"] or "",
        created_at=datetime.fromisoformat(row["created_at"]),
        author_username=row["author_username"] or "",
        author_display_name=row["author_display_name"] or "",
        author_is_bot=bool(row["author_is_bot"]),
        mention_ids=tuple(json.loads(row["mention_ids"] or "[]")),
        status=MessageStatus(row["status"]),
    )


class SqliteIngestQueue:
    """Durable pending-message queue with keyed leases."""

    def __init__(self, connection: SqliteConnection) -> None:
        self._db = connection

    async def put_message(
        self,
        message: StoredMessage,
        *,
        max_depth: int | None = None,
    ) -> bool:
        if max_depth is not None:
            count = await self.pending_count(message.guild_id)
            if count >= max_depth:
                return False
        ensure_aware(message.created_at)
        try:
            await self._db.execute(
                """INSERT INTO dm_messages
                   (message_id, guild_id, channel_id, author_id, subject_key, content,
                    created_at, author_username, author_display_name, author_is_bot,
                    mention_ids, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (
                    message.message_id,
                    message.guild_id,
                    message.channel_id,
                    message.author_id,
                    message.subject_key,
                    message.content,
                    iso(message.created_at),
                    message.author_username,
                    message.author_display_name,
                    int(message.author_is_bot),
                    json.dumps(list(message.mention_ids)),
                ),
            )
            return True
        except Exception as exc:
            if "UNIQUE constraint failed" in str(exc):
                return False
            raise

    async def due_batch_keys(
        self,
        *,
        now: datetime,
        batch_size: int,
        max_age_seconds: float,
        limit: int,
    ) -> tuple[BatchKey, ...]:
        now = ensure_aware(now)
        cutoff = iso(now - timedelta(seconds=max_age_seconds))
        rows = await self._db.query(
            """SELECT guild_id, subject_key,
                      COUNT(*) AS pending_count,
                      MIN(created_at) AS oldest_at
               FROM dm_messages
               WHERE status = 'pending'
               GROUP BY guild_id, subject_key
               HAVING pending_count >= ? OR oldest_at <= ?
               ORDER BY oldest_at ASC
               LIMIT ?""",
            (batch_size, cutoff, limit),
        )
        return tuple(BatchKey(guild_id=r["guild_id"], subject_key=r["subject_key"]) for r in rows)

    async def _acquire_key_lease(
        self,
        key: BatchKey,
        *,
        now: datetime,
        lease_seconds: float,
        owner: str,
    ) -> bool:
        """CAS the durable lease row. True iff we own the key until expiry."""
        await self._db.execute(
            """INSERT INTO dm_batch_leases (guild_id, subject_key, owner, lease_until)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(guild_id, subject_key) DO UPDATE SET
                 owner=excluded.owner, lease_until=excluded.lease_until
               WHERE dm_batch_leases.owner = excluded.owner
                  OR dm_batch_leases.lease_until <= ?""",
            (
                key.guild_id,
                key.subject_key,
                owner,
                iso(now + timedelta(seconds=lease_seconds)),
                iso(now),
            ),
        )
        row = await self._db.query_one(
            "SELECT owner FROM dm_batch_leases WHERE guild_id=? AND subject_key=?",
            (key.guild_id, key.subject_key),
        )
        return row is not None and row["owner"] == owner

    async def renew_lease(
        self,
        key: BatchKey,
        *,
        owner: str,
        now: datetime,
        lease_seconds: float,
    ) -> bool:
        result = await self._db.query_one(
            "SELECT owner FROM dm_batch_leases WHERE guild_id=? AND subject_key=?",
            (key.guild_id, key.subject_key),
        )
        if result is None or result["owner"] != owner:
            return False
        await self._db.execute(
            "UPDATE dm_batch_leases SET lease_until=? "
            "WHERE guild_id=? AND subject_key=? AND owner=?",
            (
                iso(now + timedelta(seconds=lease_seconds)),
                key.guild_id,
                key.subject_key,
                owner,
            ),
        )
        return True

    async def release_key(self, key: BatchKey, *, owner: str) -> None:
        await self._db.execute(
            "DELETE FROM dm_batch_leases WHERE guild_id=? AND subject_key=? AND owner=?",
            (key.guild_id, key.subject_key, owner),
        )

    async def claim_batch(
        self,
        key: BatchKey,
        *,
        now: datetime,
        lease_seconds: float,
        owner: str,
        limit: int,
    ) -> ClaimOutcome:
        now = ensure_aware(now)
        if not await self._acquire_key_lease(
            key, now=now, lease_seconds=lease_seconds, owner=owner
        ):
            return ClaimOutcome(key=key, locked_by_other=True)

        rows = await self._db.execute_returning(
            """UPDATE dm_messages SET status='claimed', lease_owner=?, lease_until=?
               WHERE message_id IN (
                   SELECT message_id FROM dm_messages
                   WHERE guild_id=? AND subject_key=? AND status='pending'
                   ORDER BY created_at ASC LIMIT ?
               )
               RETURNING *""",
            (
                owner,
                iso(now + timedelta(seconds=lease_seconds)),
                key.guild_id,
                key.subject_key,
                limit,
            ),
        )
        # RETURNING yields exactly the rows this call flipped — a concurrent
        # same-owner claim sees zero rows instead of re-reading an in-flight batch.
        return ClaimOutcome(key=key, messages=tuple(_message_from_row(r) for r in rows))

    async def complete_messages(self, message_ids: tuple[str, ...], owner: str) -> int:
        count = 0
        for message_id in message_ids:
            cursor_info = await self._db.query_one(
                "SELECT status, lease_owner FROM dm_messages WHERE message_id=?",
                (message_id,),
            )
            if (
                cursor_info is None
                or cursor_info["status"] != "claimed"
                or cursor_info["lease_owner"] != owner
            ):
                continue
            await self._db.execute(
                """UPDATE dm_messages SET status='processed', lease_owner=NULL,
                       lease_until=NULL WHERE message_id=?""",
                (message_id,),
            )
            count += 1
        return count

    async def release_expired_leases(self, now: datetime) -> int:
        now = ensure_aware(now)
        expired_keys = await self._db.query(
            """SELECT DISTINCT guild_id, subject_key FROM dm_batch_leases
               WHERE lease_until <= ?""",
            (iso(now),),
        )
        released = 0
        for lease_row in expired_keys:
            rows = await self._db.query(
                """SELECT message_id FROM dm_messages
                   WHERE guild_id=? AND subject_key=? AND status='claimed'""",
                (lease_row["guild_id"], lease_row["subject_key"]),
            )
            ids = [r["message_id"] for r in rows]
            if ids:
                placeholders = ",".join("?" * len(ids))
                await self._db.execute(
                    f"""UPDATE dm_messages SET status='pending', lease_owner=NULL,
                          lease_until=NULL WHERE message_id IN ({placeholders})""",
                    tuple(ids),
                )
                released += len(ids)
            await self._db.execute(
                "DELETE FROM dm_batch_leases WHERE guild_id=? AND subject_key=?",
                (lease_row["guild_id"], lease_row["subject_key"]),
            )
        return released

    async def dead_letter_messages(
        self,
        message_ids: tuple[str, ...],
        owner: str,
    ) -> int:
        count = 0
        for message_id in message_ids:
            row = await self._db.query_one(
                "SELECT status, lease_owner FROM dm_messages WHERE message_id=?",
                (message_id,),
            )
            if row is None or row["status"] != "claimed" or row["lease_owner"] != owner:
                continue
            await self._db.execute(
                "UPDATE dm_messages SET status='dead', lease_owner=NULL WHERE message_id=?",
                (message_id,),
            )
            count += 1
        return count

    async def requeue_dead_letters(self, guild_id: str | None = None) -> int:
        rows = await self._db.query(
            "SELECT message_id FROM dm_messages WHERE status='dead'"
            + (" AND guild_id=?" if guild_id else ""),
            (guild_id,) if guild_id else (),
        )
        for row in rows:
            await self._db.execute(
                "UPDATE dm_messages SET status='pending' WHERE message_id=?",
                (row["message_id"],),
            )
        return len(rows)

    async def prune_processed(self, *, older_than: datetime) -> int:
        result = await self._db.query(
            """SELECT message_id FROM dm_messages
               WHERE status='processed' AND created_at < ?""",
            (iso(ensure_aware(older_than)),),
        )
        ids = [r["message_id"] for r in result]
        for message_id in ids:
            await self._db.execute(
                "DELETE FROM dm_messages WHERE message_id=?",
                (message_id,),
            )
        return len(ids)

    async def pending_count(self, guild_id: str) -> int:
        row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM dm_messages WHERE guild_id=? AND status='pending'",
            (guild_id,),
        )
        return int(row["n"]) if row is not None else 0

    async def recent_messages(self, guild_id: str, limit: int) -> tuple[StoredMessage, ...]:
        rows = await self._db.query(
            """SELECT * FROM dm_messages WHERE guild_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (guild_id, limit),
        )
        return tuple(_message_from_row(r) for r in reversed(rows))

    async def dead_letter_count(self, guild_id: str) -> int:
        row = await self._db.query_one(
            "SELECT COUNT(*) AS n FROM dm_messages WHERE guild_id=? AND status='dead'",
            (guild_id,),
        )
        return int(row["n"]) if row is not None else 0
