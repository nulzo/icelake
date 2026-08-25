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

    async def put_message(self, message: StoredMessage) -> bool:
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
        lease_until = now + timedelta(seconds=lease_seconds)

        held = await self._db.query_one(
            """SELECT lease_owner, lease_until FROM dm_messages
               WHERE guild_id = ? AND subject_key = ?
                 AND status = 'claimed' AND lease_until IS NOT NULL
               LIMIT 1""",
            (key.guild_id, key.subject_key),
        )
        if held is not None:
            active_until = datetime.fromisoformat(held["lease_until"])
            if active_until > now and held["lease_owner"] != owner:
                return ClaimOutcome(key=key, locked_by_other=True)

        await self._db.execute(
            """UPDATE dm_messages SET status='claimed', lease_owner=?, lease_until=?,
                  message_id=message_id
               WHERE message_id IN (
                   SELECT message_id FROM dm_messages
                   WHERE guild_id=? AND subject_key=? AND status='pending'
                   ORDER BY created_at ASC LIMIT ?
               )""",
            (owner, iso(lease_until), key.guild_id, key.subject_key, limit),
        )
        rows = await self._db.query(
            """SELECT * FROM dm_messages
               WHERE guild_id=? AND subject_key=? AND status='claimed' AND lease_owner=?
               ORDER BY created_at ASC""",
            (key.guild_id, key.subject_key, owner),
        )
        messages = tuple(_message_from_row(r) for r in rows)
        return ClaimOutcome(key=key, messages=messages)

    async def complete_messages(self, message_ids: tuple[str, ...], owner: str) -> int:
        count = 0
        for message_id in message_ids:
            cursor_info = await self._db.query_one(
                "SELECT status FROM dm_messages WHERE message_id=?",
                (message_id,),
            )
            if cursor_info is None or cursor_info["status"] != "claimed":
                continue
            await self._db.execute(
                """UPDATE dm_messages SET status='processed', lease_owner=NULL,
                       lease_until=NULL WHERE message_id=?""",
                (message_id,),
            )
            count += 1
        del owner
        return count

    async def release_expired_leases(self, now: datetime) -> int:
        now = ensure_aware(now)
        result = await self._db.query(
            "SELECT message_id FROM dm_messages WHERE status='claimed' AND lease_until<=?",
            (iso(now),),
        )
        ids = [r["message_id"] for r in result]
        if not ids:
            return 0
        placeholders = ",".join("?" * len(ids))
        await self._db.execute(
            f"""UPDATE dm_messages SET status='pending', lease_owner=NULL, lease_until=NULL
                WHERE message_id IN ({placeholders})""",
            tuple(ids),
        )
        return len(ids)

    async def dead_letter_messages(
        self,
        message_ids: tuple[str, ...],
        owner: str,
    ) -> int:
        count = 0
        for message_id in message_ids:
            row = await self._db.query_one(
                "SELECT status FROM dm_messages WHERE message_id=?",
                (message_id,),
            )
            if row is None or row["status"] != "claimed":
                continue
            await self._db.execute(
                "UPDATE dm_messages SET status='dead', lease_owner=NULL WHERE message_id=?",
                (message_id,),
            )
            count += 1
        del owner
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
