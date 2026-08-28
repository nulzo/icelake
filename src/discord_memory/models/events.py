"""Ingestion boundary models: MessageEvent input and ObserveReceipt output."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum

from pydantic import Field, field_validator

from discord_memory.models.common import FrozenModel, ensure_aware


class MessageEvent(FrozenModel):
    """One observed Discord message. The only write-path input consumers construct.

    ``created_at`` must be timezone-aware; naive values raise immediately because they
    indicate broken integration code (API.md Part 5).
    """

    message_id: str = Field(min_length=1)
    guild_id: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    author_id: str = Field(min_length=1)
    content: str
    created_at: datetime

    author_username: str = ""
    author_display_name: str = ""
    author_is_bot: bool = False
    mention_ids: tuple[str, ...] = ()
    reply_to_message_id: str | None = None
    thread_parent_id: str | None = None
    edited: bool = False
    metadata: Mapping[str, str] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        return ensure_aware(value)

    @field_validator("metadata")
    @classmethod
    def _bounded_metadata(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        if len(value) > 16:
            raise ValueError("metadata supports at most 16 keys")
        for key, item in value.items():
            if len(key) > 64 or len(item) > 256:
                raise ValueError("metadata keys/values exceed size bounds")
        return value


class ObserveStatus(StrEnum):
    ACCEPTED = "accepted"
    IGNORED = "ignored"
    REJECTED = "rejected"


class IgnoreReason(StrEnum):
    BOT_AUTHOR = "bot_author"
    OPTED_OUT = "opted_out"
    EMPTY_CONTENT = "empty_content"
    DUPLICATE = "duplicate"
    IGNORED_PATTERN = "ignored_pattern"


class RejectReason(StrEnum):
    QUEUE_OVER_CAPACITY = "queue_over_capacity"
    STORAGE_UNAVAILABLE = "storage_unavailable"


class ObserveReceipt(FrozenModel):
    """Outcome of one :meth:`observe` call. Hot path reports; it never crashes bots."""

    message_id: str
    status: ObserveStatus
    reason: IgnoreReason | RejectReason | None = None


class BatchCompleted(FrozenModel):
    """One extraction batch finished (hook payload)."""

    guild_id: str
    subject_key: str
    adds: int = 0
    reinforces: int = 0
    supersessions: int = 0
    invalidations: int = 0
    skipped_reason: str | None = None


class FactCommitted(FrozenModel):
    """A fact was created or reinforced (hook payload)."""

    guild_id: str
    fact_id: str
    subject_id: str | None
    text: str
    was_reinforcement: bool = False


class FactSupersededEvent(FrozenModel):
    """A fact was replaced, refined in place, or retired (hook payload).

    ``new_fact_id`` equals ``old_fact_id`` for in-place refinements
    (``facts.update``) and is ``None`` when no successor exists
    (``facts.forget``, invalidate-without-successor).
    """

    guild_id: str
    old_fact_id: str
    new_fact_id: str | None
    reason: str = ""


class ExtractionFailed(FrozenModel):
    """Extraction errored for a job (hook payload)."""

    guild_id: str
    subject_key: str
    attempt: int
    error_kind: str


class BudgetWarning(FrozenModel):
    """Guild spend crossed a warning fraction of budget (hook payload)."""

    guild_id: str
    fraction_used: float
    next_step: str = ""


class ComponentDegraded(FrozenModel):
    """A subsystem failed over to degraded mode (hook payload)."""

    component: str
    detail: str
