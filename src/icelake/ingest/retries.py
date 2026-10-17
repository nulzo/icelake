"""Durable batch retry scheduling with bounded exponential backoff."""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from pydantic import Field

from icelake.models.common import FrozenModel
from icelake.ports.queue import BatchKey
from icelake.ports.store import MemoryStore


class RetryState(FrozenModel):
    attempts: int = Field(default=0, ge=0)
    next_attempt_at: datetime | None = None
    error_kind: str = ""


async def read_retry(store: MemoryStore, key: BatchKey) -> RetryState:
    epoch = await store.get_cursor(key.guild_id, "retry_epoch") or "0"
    raw = await store.get_cursor(key.guild_id, f"retry:{epoch}:{key.subject_key}")
    return RetryState.model_validate_json(raw) if raw else RetryState()


async def save_retry(store: MemoryStore, key: BatchKey, state: RetryState) -> None:
    epoch = await store.get_cursor(key.guild_id, "retry_epoch") or "0"
    await store.set_cursor(
        key.guild_id, f"retry:{epoch}:{key.subject_key}", state.model_dump_json()
    )


def after_failure(
    previous: RetryState,
    *,
    now: datetime,
    error: Exception,
    initial_seconds: float,
    max_seconds: float,
) -> RetryState:
    """Full jitter with a nonzero floor avoids synchronized restart storms."""
    cap = min(max_seconds, initial_seconds * 2 ** min(previous.attempts, 20))
    return RetryState(
        attempts=previous.attempts + 1,
        next_attempt_at=now + timedelta(seconds=random.uniform(cap / 2, cap)),
        error_kind=type(error).__name__,
    )
