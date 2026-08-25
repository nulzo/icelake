"""Durable ingest queue port: message batching with keyed leases.

Fixes the reviewed bot bugs B2/B3/B4: atomic claim (no TOCTOU), lease expiry reclaim
(no orphans), and dead-lettering (no silent loss).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from discord_memory.models.common import FrozenModel, ensure_aware


class MessageStatus(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    PROCESSED = "processed"
    DEAD = "dead"


class StoredMessage(FrozenModel):
    """Persisted raw message awaiting extraction."""

    message_id: str
    guild_id: str
    channel_id: str = ""
    author_id: str
    subject_key: str
    content: str
    created_at: datetime
    author_username: str = ""
    author_display_name: str = ""
    author_is_bot: bool = False
    mention_ids: tuple[str, ...] = ()
    status: MessageStatus = MessageStatus.PENDING

    def __post_init__(self) -> None:
        ensure_aware(self.created_at)


class BatchKey(FrozenModel):
    """Lease key: ``(guild_id, subject_key)``; ``subject_key="__server__"`` is the
    community-scope batch. Identical mechanism for both scopes (fixes B3)."""

    guild_id: str
    subject_key: str

    @property
    def as_tuple(self) -> tuple[str, str]:
        return (self.guild_id, self.subject_key)


class ClaimOutcome(FrozenModel):
    """Result of one claim attempt. ``LOCKED`` means another worker holds the key."""

    key: BatchKey
    messages: tuple[StoredMessage, ...] = ()
    locked_by_other: bool = False


@runtime_checkable
class IngestQueue(Protocol):
    async def put_message(self, message: StoredMessage) -> bool:
        """Persist a pending message; ``False`` when the id already exists (idempotent)."""

    async def due_batch_keys(
        self,
        *,
        now: datetime,
        batch_size: int,
        max_age_seconds: float,
        limit: int,
    ) -> tuple[BatchKey, ...]:
        """Keys whose pending count reached ``batch_size`` OR oldest pending is stale."""

    async def claim_batch(
        self,
        key: BatchKey,
        *,
        now: datetime,
        lease_seconds: float,
        owner: str,
        limit: int,
    ) -> ClaimOutcome:
        """Atomically flip up to ``limit`` pending messages to CLAIMED under a lease."""

    async def complete_messages(self, message_ids: tuple[str, ...], owner: str) -> int:
        """Mark claimed messages processed (owner-checked, idempotent)."""

    async def release_expired_leases(self, now: datetime) -> int:
        """Reclaim CLAIMED messages whose lease expired back to PENDING."""

    async def dead_letter_messages(
        self,
        message_ids: tuple[str, ...],
        owner: str,
    ) -> int: ...

    async def requeue_dead_letters(self, guild_id: str | None = None) -> int: ...

    async def pending_count(self, guild_id: str) -> int: ...

    async def dead_letter_count(self, guild_id: str) -> int: ...

    async def recent_messages(self, guild_id: str, limit: int) -> tuple[StoredMessage, ...]:
        """Most recent messages regardless of status (community-window source)."""
