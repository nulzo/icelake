"""Durable indexing work and bounded recovery reports."""

from datetime import datetime

from pydantic import Field

from icelake.models.common import FrozenModel


class ProjectionJob(FrozenModel):
    guild_id: str
    fact_id: str
    version: int = Field(ge=1)
    embedding: tuple[float, ...] | None = None
    next_attempt_at: datetime | None = None


class RecoveryReport(FrozenModel):
    attempted: int = 0
    completed: int = 0
    failed: int = 0
