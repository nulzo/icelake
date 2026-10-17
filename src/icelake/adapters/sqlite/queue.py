"""SQLite IngestQueue: atomic claims, leases, dead-letters (B2/B3/B4 fixes)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

from icelake.adapters.sqlite.connection import SqliteConnection, iso
from icelake.models.common import ensure_aware
from icelake.ports.queue import BatchKey, ClaimOutcome, MessageStatus, StoredMessage

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
        governance=dict(json.loads(row["governance"] or "{}")),
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
        async with self._db.transaction_scope():
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
                        mention_ids, status, governance)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)""",
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
                        json.dumps(dict(message.governance)),
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
        rows = await self._db.execute_returning(
            """INSERT INTO dm_batch_leases (guild_id, subject_key, owner, lease_until)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(guild_id, subject_key) DO UPDATE SET
                 owner=excluded.owner, lease_until=excluded.lease_until
               WHERE dm_batch_leases.lease_until <= ? RETURNING owner""",
            (
                key.guild_id,
                key.subject_key,
                owner,
                iso(now + timedelta(seconds=lease_seconds)),
                iso(now),
            ),
        )
        return bool(rows)

    async def renew_lease(
        self,
        key: BatchKey,
        *,
        owner: str,
        now: datetime,
        lease_seconds: float,
    ) -> bool:
        async with self._db.transaction_scope():
            rows = await self._db.execute_returning(
                """UPDATE dm_batch_leases SET lease_until=?
                   WHERE guild_id=? AND subject_key=? AND owner=? AND lease_until>?
                   RETURNING owner""",
                (
                    iso(now + timedelta(seconds=lease_seconds)),
                    key.guild_id,
                    key.subject_key,
                    owner,
                    iso(now),
                ),
            )
            if not rows:
                return False
            await self._db.execute(
                """UPDATE dm_messages SET lease_until=? WHERE guild_id=?
                   AND subject_key=? AND status='claimed' AND lease_owner=?""",
                (iso(now + timedelta(seconds=lease_seconds)), key.guild_id, key.subject_key, owner),
            )
            return True

    async def release_key(self, key: BatchKey, *, owner: str) -> None:
        async with self._db.transaction_scope():
            await self._db.execute(
                """UPDATE dm_messages SET status='pending', lease_owner=NULL, lease_until=NULL
                   WHERE guild_id=? AND subject_key=? AND status='claimed' AND lease_owner=?""",
                (key.guild_id, key.subject_key, owner),
            )
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
        async with self._db.transaction_scope():
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
            # The key lease and message transition share the same transaction.
            return ClaimOutcome(key=key, messages=tuple(_message_from_row(r) for r in rows))

    async def _finish(self, message_ids: tuple[str, ...], owner: str, status: MessageStatus) -> int:
        count = 0
        for message_id in message_ids:
            rows = await self._db.execute_returning(
                """UPDATE dm_messages SET status=?, lease_owner=NULL, lease_until=NULL
                   WHERE message_id=? AND status='claimed' AND lease_owner=?
                   RETURNING message_id""",
                (status.value, message_id, owner),
            )
            count += len(rows)
        return count

    async def complete_messages(self, message_ids: tuple[str, ...], owner: str) -> int:
        return await self._finish(message_ids, owner, MessageStatus.PROCESSED)

    async def release_expired_leases(self, now: datetime) -> int:
        async with self._db.transaction_scope():
            rows = await self._db.execute_returning(
                """UPDATE dm_messages SET status='pending', lease_owner=NULL, lease_until=NULL
                   WHERE status='claimed' AND lease_until<=? RETURNING message_id""",
                (iso(ensure_aware(now)),),
            )
            await self._db.execute("DELETE FROM dm_batch_leases WHERE lease_until<=?", (iso(now),))
            return len(rows)

    async def dead_letter_messages(self, message_ids: tuple[str, ...], owner: str) -> int:
        return await self._finish(message_ids, owner, MessageStatus.DEAD)

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

    async def drop_governance(self, guild_id: str, user_id: str) -> int:
        rows = await self._db.query(
            """SELECT message_id FROM dm_messages
               WHERE guild_id=? AND (author_id=? OR mention_ids LIKE ?)""",
            (guild_id, user_id, f'%"{user_id}"%'),
        )
        for row in rows:
            await self._db.execute(
                "DELETE FROM dm_messages WHERE message_id=?",
                (row["message_id"],),
            )
        return len(rows)

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
