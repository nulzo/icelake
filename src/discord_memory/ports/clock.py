"""Time and identifier ports plus system implementations."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from discord_memory.ids import prefixed
from discord_memory.models.common import ensure_aware, utc_now


class Clock(Protocol):
    """Time source port. Fakes make TTL/decay tests deterministic."""

    def now(self) -> datetime: ...


class IdGen(Protocol):
    """Identifier minting port."""

    def new_id(self, prefix: str) -> str: ...


class SystemClock:
    """Wall-clock implementation of :class:`Clock`."""

    def now(self) -> datetime:
        return utc_now()


class UlidIdGen:
    """ULID-based implementation of :class:`IdGen`."""

    def new_id(self, prefix: str) -> str:
        return prefixed(prefix)


class FixedClock:
    """Manually-advanced clock for tests, evals and replay harnesses."""

    def __init__(self, start: datetime) -> None:
        self._current = ensure_aware(start)

    def advance(self, seconds: float) -> None:
        self._current += timedelta(seconds=seconds)

    def now(self) -> datetime:
        return self._current


__all__ = ["Clock", "FixedClock", "IdGen", "SystemClock", "UlidIdGen"]
