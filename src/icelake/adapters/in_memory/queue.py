"""In-memory IngestQueue with keyed leases — reference semantics for all backends."""

from __future__ import annotations

import asyncio
from datetime import datetime

from icelake.models.common import ensure_aware
from icelake.ports.queue import (
    BatchKey,
    ClaimOutcome,
    MessageStatus,
    StoredMessage,
)


class InMemoryIngestQueue:
    """Dict-backed queue. Lease/claim behavior mirrors the SQLite adapter exactly."""

    def __init__(self, *, max_pending: int = 10_000) -> None:
        self._lock = asyncio.Lock()
        self._messages: dict[str, StoredMessage] = {}
        self._order: list[str] = []
        self._lease_owner: dict[tuple[str, str], str] = {}
        self._lease_until: dict[tuple[str, str], datetime] = {}
        self.max_pending = max_pending

    async def put_message(
        self,
        message: StoredMessage,
        *,
        max_depth: int | None = None,
    ) -> bool:
        ensure_aware(message.created_at)
        async with self._lock:
            if max_depth is not None:
                guild_pending = sum(
                    1
                    for m in self._messages.values()
                    if m.guild_id == message.guild_id and m.status is MessageStatus.PENDING
                )
                if guild_pending >= max_depth:
                    return False
            if message.message_id in self._messages:
                return False
            if len(self._messages) >= self.max_pending:
                return False
            self._messages[message.message_id] = message
            self._order.append(message.message_id)
            return True

    async def due_batch_keys(
        self,
        *,
        now: datetime,
        batch_size: int,
        max_age_seconds: float,
        limit: int,
    ) -> tuple[BatchKey, ...]:
        now = ensure_aware(now)
        pending: dict[tuple[str, str], list[StoredMessage]] = {}
        for message_id in self._order:
            message = self._messages.get(message_id)
            if message is None or message.status is not MessageStatus.PENDING:
                continue
            pending.setdefault((message.guild_id, message.subject_key), []).append(message)
        due: list[BatchKey] = []
        for (guild_id, subject_key), messages in pending.items():
            if len(messages) >= batch_size:
                due.append(BatchKey(guild_id=guild_id, subject_key=subject_key))
                continue
            oldest = min(messages, key=lambda m: m.created_at)
            age = (now - oldest.created_at).total_seconds()
            if age >= max_age_seconds:
                due.append(BatchKey(guild_id=guild_id, subject_key=subject_key))
        return tuple(due[:limit])

    async def claim_batch(
        self,
        key: BatchKey,
        *,
        now: datetime,
        lease_seconds: float,
        owner: str,
        limit: int,
    ) -> ClaimOutcome:
        from datetime import timedelta

        now = ensure_aware(now)
        lease_key = key.as_tuple
        async with self._lock:
            deadline = self._lease_until.get(lease_key)
            holder = self._lease_owner.get(lease_key)
            if deadline is not None and deadline > now and holder != owner:
                return ClaimOutcome(key=key, locked_by_other=True)
            candidates = [
                message
                for message_id in self._order
                if (message := self._messages.get(message_id)) is not None
                and message.guild_id == key.guild_id
                and message.subject_key == key.subject_key
                and message.status is MessageStatus.PENDING
            ]
            if not candidates:
                return ClaimOutcome(key=key)
            window = sorted(candidates, key=lambda m: m.created_at)[:limit]
            claimed: list[StoredMessage] = []
            for message in window:
                updated = message.model_copy(update={"status": MessageStatus.CLAIMED})
                self._messages[message.message_id] = updated
                claimed.append(updated)
            self._lease_owner[lease_key] = owner
            self._lease_until[lease_key] = now + timedelta(seconds=lease_seconds)
            return ClaimOutcome(key=key, messages=tuple(claimed))

    async def complete_messages(self, message_ids: tuple[str, ...], owner: str) -> int:
        del owner
        count = 0
        async with self._lock:
            for message_id in message_ids:
                message = self._messages.get(message_id)
                if message is None or message.status is not MessageStatus.CLAIMED:
                    continue
                self._messages[message_id] = message.model_copy(
                    update={
                        "status": MessageStatus.PROCESSED,
                    }
                )
                count += 1
        return count

    async def release_expired_leases(self, now: datetime) -> int:
        now = ensure_aware(now)
        async with self._lock:
            expired_keys = [key for key, deadline in self._lease_until.items() if deadline <= now]
            released = 0
            for key in expired_keys:
                for message in self._messages.values():
                    if (
                        message.guild_id == key[0]
                        and message.subject_key == key[1]
                        and message.status is MessageStatus.CLAIMED
                    ):
                        self._messages[message.message_id] = message.model_copy(
                            update={
                                "status": MessageStatus.PENDING,
                            }
                        )
                        released += 1
                self._lease_until.pop(key, None)
                self._lease_owner.pop(key, None)
            return released

    async def dead_letter_messages(
        self,
        message_ids: tuple[str, ...],
        owner: str,
    ) -> int:
        del owner
        count = 0
        async with self._lock:
            for message_id in message_ids:
                message = self._messages.get(message_id)
                if message is None or message.status is not MessageStatus.CLAIMED:
                    continue
                self._messages[message_id] = message.model_copy(
                    update={
                        "status": MessageStatus.DEAD,
                    }
                )
                count += 1
        return count

    async def requeue_dead_letters(self, guild_id: str | None = None) -> int:
        count = 0
        async with self._lock:
            for message_id, message in self._messages.items():
                if message.status is not MessageStatus.DEAD:
                    continue
                if guild_id is not None and message.guild_id != guild_id:
                    continue
                self._messages[message_id] = message.model_copy(
                    update={
                        "status": MessageStatus.PENDING,
                    }
                )
                count += 1
        return count

    async def pending_count(self, guild_id: str) -> int:
        return sum(
            1
            for m in self._messages.values()
            if m.guild_id == guild_id and m.status is MessageStatus.PENDING
        )

    async def dead_letter_count(self, guild_id: str) -> int:
        return sum(
            1
            for m in self._messages.values()
            if m.guild_id == guild_id and m.status is MessageStatus.DEAD
        )

    async def recent_messages(self, guild_id: str, limit: int) -> tuple[StoredMessage, ...]:
        selected = [m for m in self._messages.values() if m.guild_id == guild_id]
        selected.sort(key=lambda m: m.created_at)
        return tuple(selected[-limit:])

    async def renew_lease(
        self,
        key: BatchKey,
        *,
        owner: str,
        now: datetime,
        lease_seconds: float,
    ) -> bool:
        from datetime import timedelta

        now = ensure_aware(now)
        lease_key = key.as_tuple
        async with self._lock:
            holder = self._lease_owner.get(lease_key)
            if holder != owner:
                return False
            self._lease_until[lease_key] = now + timedelta(seconds=lease_seconds)
            return True

    async def release_key(self, key: BatchKey, *, owner: str) -> None:
        lease_key = key.as_tuple
        async with self._lock:
            if self._lease_owner.get(lease_key) == owner:
                self._lease_owner.pop(lease_key, None)
                self._lease_until.pop(lease_key, None)

    async def prune_processed(self, *, older_than: datetime) -> int:
        cutoff = ensure_aware(older_than)
        doomed = [
            mid
            for mid, m in self._messages.items()
            if m.status is MessageStatus.PROCESSED and m.created_at < cutoff
        ]
        for mid in doomed:
            del self._messages[mid]
            self._order.remove(mid)
        return len(doomed)
