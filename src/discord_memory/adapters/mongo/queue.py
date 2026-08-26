"""Mongo IngestQueue: keyed leases via atomic find_one_and_update claims."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from discord_memory.models.common import ensure_aware
from discord_memory.ports.queue import BatchKey, ClaimOutcome, MessageStatus, StoredMessage

PENDING = "pending"
CLAIMED = "claimed"
PROCESSED = "processed"
DEAD = "dead"


def message_to_doc(message: StoredMessage) -> dict[str, Any]:
    return {
        "_id": message.message_id,
        "guild_id": message.guild_id,
        "channel_id": message.channel_id,
        "author_id": message.author_id,
        "subject_key": message.subject_key,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
        "author_username": message.author_username,
        "author_display_name": message.author_display_name,
        "author_is_bot": int(message.author_is_bot),
        "mention_ids": json.dumps(list(message.mention_ids)),
        "status": PENDING,
        "lease_owner": None,
        "lease_until": None,
    }


def doc_to_message(doc: dict[str, Any]) -> StoredMessage:
    return StoredMessage(
        message_id=doc["_id"],
        guild_id=doc["guild_id"],
        channel_id=doc.get("channel_id", ""),
        author_id=doc["author_id"],
        subject_key=doc["subject_key"],
        content=doc.get("content", ""),
        created_at=datetime.fromisoformat(doc["created_at"]),
        author_username=doc.get("author_username", ""),
        author_display_name=doc.get("author_display_name", ""),
        author_is_bot=bool(doc.get("author_is_bot")),
        mention_ids=tuple(json.loads(doc.get("mention_ids") or "[]")),
        status=MessageStatus(doc.get("status", "pending")),
    )


class MongoIngestQueue:
    """Durable pending-message queue with keyed leases (B2/B3/B4 fixes)."""

    def __init__(self, db: Any) -> None:
        self.col = db["dm_messages"]

    async def setup(self) -> None:
        await self.col.create_index(
            [("guild_id", 1), ("subject_key", 1), ("status", 1), ("created_at", 1)],
        )

    async def put_message(
        self,
        message: StoredMessage,
        *,
        max_depth: int | None = None,
    ) -> bool:
        if max_depth is not None:
            count = await self.col.count_documents(
                {"guild_id": message.guild_id, "status": "pending"},
            )
            if count >= max_depth:
                return False
        ensure_aware(message.created_at)
        try:
            await self.col.insert_one(message_to_doc(message))
            return True
        except Exception as exc:
            if "duplicate key" in str(exc).lower() or "E11000" in str(exc):
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
        cutoff = (now - timedelta(seconds=max_age_seconds)).isoformat()
        pipeline: list[dict[str, Any]] = [
            {"$match": {"status": PENDING}},
            {
                "$group": {
                    "_id": {"guild_id": "$guild_id", "subject_key": "$subject_key"},
                    "count": {"$sum": 1},
                    "oldest": {"$min": "$created_at"},
                }
            },
            {
                "$match": {
                    "$or": [
                        {"count": {"$gte": batch_size}},
                        {"oldest": {"$lte": cutoff}},
                    ],
                }
            },
            {"$sort": {"oldest": 1}},
            {"$limit": limit},
        ]
        keys: list[BatchKey] = []
        agg_cursor = await self.col.aggregate(pipeline)
        docs = await agg_cursor.to_list(length=limit)
        for doc in docs:
            keys.append(
                BatchKey(guild_id=doc["_id"]["guild_id"], subject_key=doc["_id"]["subject_key"])
            )
        return tuple(keys)

    async def _acquire_key_lease(
        self,
        key: BatchKey,
        *,
        now: datetime,
        lease_seconds: float,
        owner: str,
    ) -> bool:
        """Atomic CAS on a durable lease doc; True iff we own it until expiry."""
        from datetime import timedelta

        from pymongo.errors import DuplicateKeyError

        lease_until = now + timedelta(seconds=lease_seconds)
        try:
            result = await self.col.database["dm_batch_leases"].update_one(
                {
                    "_id": f"{key.guild_id}:{key.subject_key}",
                    "$or": [
                        {"owner": owner},
                        {"lease_until": {"$lte": now.isoformat()}},
                    ],
                },
                {
                    "$set": {
                        "guild_id": key.guild_id,
                        "subject_key": key.subject_key,
                        "owner": owner,
                        "lease_until": lease_until.isoformat(),
                    }
                },
                upsert=True,
            )
        except DuplicateKeyError:
            # A live lease held by another worker rejected the CAS.
            return False
        if result.upserted_id or result.modified_count:
            return True
        held = await self.col.database["dm_batch_leases"].find_one(
            {"_id": f"{key.guild_id}:{key.subject_key}", "owner": owner},
        )
        return held is not None

    async def renew_lease(
        self,
        key: BatchKey,
        *,
        owner: str,
        now: datetime,
        lease_seconds: float,
    ) -> bool:
        from datetime import timedelta

        result = await self.col.database["dm_batch_leases"].update_one(
            {"_id": f"{key.guild_id}:{key.subject_key}", "owner": owner},
            {
                "$set": {
                    "lease_until": (now + timedelta(seconds=lease_seconds)).isoformat(),
                }
            },
        )
        return bool(result.modified_count > 0)

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
            key,
            now=now,
            lease_seconds=lease_seconds,
            owner=owner,
        ):
            return ClaimOutcome(key=key, locked_by_other=True)

        claimed: list[StoredMessage] = []
        for _ in range(limit):
            doc = await self.col.find_one_and_update(
                {
                    "guild_id": key.guild_id,
                    "subject_key": key.subject_key,
                    "status": PENDING,
                },
                {
                    "$set": {
                        "status": CLAIMED,
                        "lease_owner": owner,
                        "lease_until": (now + timedelta(seconds=lease_seconds)).isoformat(),
                    }
                },
                sort=[("created_at", 1)],
            )
            if doc is None:
                break
            claimed.append(doc_to_message(doc))
        return ClaimOutcome(key=key, messages=tuple(claimed))

    async def complete_messages(self, message_ids: tuple[str, ...], owner: str) -> int:
        result = await self.col.update_many(
            {"_id": {"$in": list(message_ids)}, "status": CLAIMED, "lease_owner": owner},
            {"$set": {"status": PROCESSED, "lease_owner": None, "lease_until": None}},
        )
        return int(result.modified_count)

    async def release_expired_leases(self, now: datetime) -> int:
        expired = [
            doc
            async for doc in self.col.database["dm_batch_leases"].find(
                {"lease_until": {"$lte": now.isoformat()}},
            )
        ]
        released = 0
        for lease in expired:
            guild_id = lease.get("guild_id", "")
            subject_key = lease.get("subject_key", "")
            result = await self.col.update_many(
                {"guild_id": guild_id, "subject_key": subject_key, "status": CLAIMED},
                {"$set": {"status": PENDING, "lease_owner": None, "lease_until": None}},
            )
            released += int(result.modified_count)
            await self.col.database["dm_batch_leases"].delete_one({"_id": lease["_id"]})
        return released

    async def dead_letter_messages(
        self,
        message_ids: tuple[str, ...],
        owner: str,
    ) -> int:
        result = await self.col.update_many(
            {"_id": {"$in": list(message_ids)}, "status": CLAIMED},
            {"$set": {"status": DEAD, "lease_owner": None}},
        )
        del owner
        return int(result.modified_count)

    async def requeue_dead_letters(self, guild_id: str | None = None) -> int:
        query_filter: dict[str, Any] = {"status": DEAD}
        if guild_id is not None:
            query_filter["guild_id"] = guild_id
        result = await self.col.update_many(query_filter, {"$set": {"status": PENDING}})
        return int(result.modified_count)

    async def pending_count(self, guild_id: str) -> int:
        count = await self.col.count_documents(
            {"guild_id": guild_id, "status": PENDING},
        )
        return int(count)

    async def dead_letter_count(self, guild_id: str) -> int:
        count = await self.col.count_documents(
            {"guild_id": guild_id, "status": DEAD},
        )
        return int(count)

    async def recent_messages(self, guild_id: str, limit: int) -> tuple[StoredMessage, ...]:
        cursor = self.col.find({"guild_id": guild_id}).sort("created_at", -1).limit(limit)
        docs = [doc async for doc in cursor]
        return tuple(doc_to_message(doc) for doc in reversed(docs))

    async def prune_processed(self, *, older_than: datetime) -> int:
        cutoff = ensure_aware(older_than).isoformat()
        result = await self.col.delete_many(
            {"status": PROCESSED, "created_at": {"$lt": cutoff}},
        )
        return int(result.deleted_count)
