"""Typed event bus: single subscription point for hooks (API.md Part 12).

Handlers are sync callables dispatched via ``loop.call_soon`` when a loop is running;
they must be quick. Heavy work belongs in consumer-spawned tasks.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

from icelake.models.events import (
    BatchCompleted,
    ComponentDegraded,
    ExtractionFailed,
    FactCommitted,
    FactSupersededEvent,
)

logger = logging.getLogger(__name__)

E = TypeVar("E")
Handler = Callable[[object], None]

HookEvent = (
    BatchCompleted | FactCommitted | FactSupersededEvent | ExtractionFailed | ComponentDegraded
)


class EventBus:
    """Registry of typed handlers. One concern: fan-out dispatch."""

    def __init__(self) -> None:
        self._handlers: dict[type, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: type[E], handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def on(self, event_type: type[E]) -> Callable[[Handler], Handler]:
        """Decorator form of :meth:`subscribe`."""

        def decorator(handler: Handler) -> Handler:
            self.subscribe(event_type, handler)
            return handler

        return decorator

    def publish(self, event: HookEvent) -> None:
        handlers = self._handlers.get(type(event), [])
        for handler in handlers:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                loop.call_soon(self._safe_call, handler, event)
            else:
                self._safe_call(handler, event)

    @staticmethod
    def _safe_call(handler: Handler, event: object) -> None:
        try:
            handler(event)
        except Exception:
            logger.warning("event handler %r raised", handler, exc_info=True)


__all__ = ["EventBus", "HookEvent"]
