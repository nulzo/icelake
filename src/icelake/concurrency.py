"""Task-reentrant async locking shared by transactional local adapters."""

from __future__ import annotations

import asyncio


class TaskLock:
    """Task-reentrant variant of ``asyncio.Lock``.

    A transaction scope must hold exclusivity for its whole body while store
    methods called *inside* it (same task) pass through. Other tasks block —
    SQLite is single-writer, so serialization is the correct behavior.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None
        self._depth = 0

    async def acquire(self) -> None:
        task = asyncio.current_task()
        if self._owner is not None and self._owner is task:
            self._depth += 1
            return
        await self._lock.acquire()
        self._owner = task
        self._depth = 1

    def release(self) -> None:
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()

    def held_by_current_task(self) -> bool:
        return self._owner is not None and self._owner is asyncio.current_task()

    async def __aenter__(self) -> TaskLock:
        await self.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.release()
