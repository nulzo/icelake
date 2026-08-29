"""Shared result primitives: pagination, token accounting, time helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict


class FrozenModel(BaseModel):
    """Base for all boundary models: immutable, strict, no extra keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Page[T](FrozenModel):
    """Opaque-cursor pagination envelope. ``next_cursor`` is ``None`` on the last page."""

    items: tuple[T, ...]
    next_cursor: str | None = None


class TokenUsage(FrozenModel):
    """Token accounting for one operation against its budget."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def utc_now() -> datetime:
    """Timezone-aware UTC now (the only clock reading style allowed at boundaries)."""
    return datetime.now(UTC)


def ensure_aware(value: datetime) -> datetime:
    """Reject naive datetimes; normalize aware ones to UTC."""
    if value.tzinfo is None:
        raise ValueError("naive datetimes are rejected; pass tz-aware values")
    return value.astimezone(UTC)


__all__ = ["FrozenModel", "Page", "TokenUsage", "ensure_aware", "utc_now"]
