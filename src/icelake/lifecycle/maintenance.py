"""Scheduled maintenance: expiry sweep, cap pruning, forgetting.

Runs exclusively in worker context, throttled per guild — never on read paths
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import timedelta

from icelake.config import MemoryConfig
from icelake.ports.clock import Clock
from icelake.ports.store import MemoryStore

logger = logging.getLogger(__name__)

DEFAULT_MIN_INTERVAL_SECONDS = 3600.0


@dataclass(slots=True)
class MaintenanceReport:
    guild_id: str
    expired: int = 0
    forgotten: int = 0
    pruned: int = 0
    skipped_reason: str | None = None


class MaintenanceService:
    """Throttled per-guild hygiene sweeps."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        config: MemoryConfig,
        clock: Clock,
    ) -> None:
        self._store = store
        self._config = config
        self._clock = clock
        self._last_run: dict[str, float] = {}
        self.min_interval_seconds = DEFAULT_MIN_INTERVAL_SECONDS

    def due(self, guild_id: str) -> bool:
        last = self._last_run.get(guild_id)
        return last is None or time.monotonic() - last >= self.min_interval_seconds

    def mark_run(self, guild_id: str) -> None:
        self._last_run[guild_id] = time.monotonic()

    async def run_guild(self, guild_id: str, *, force: bool = False) -> MaintenanceReport:
        if not force and not self.due(guild_id):
            return MaintenanceReport(guild_id=guild_id, skipped_reason="throttled")
        now = self._clock.now()
        expired = await self._store.sweep_expired(guild_id, now)
        forgotten = await self._store.apply_forgetting(
            guild_id,
            now=now,
            retention_floor=self._config.lifecycle.forget_retention_floor,
            stability_days=self._config.lifecycle.decay_stability_days,
        )
        pruned = await self._store.prune_to_caps(
            guild_id,
            max_per_user=self._config.lifecycle.max_facts_per_user,
            max_server=self._config.lifecycle.max_server_facts,
            now=now,
        )
        if hasattr(self._store, "queue"):
            await self._store.queue.prune_processed(
                older_than=now - timedelta(days=self._config.privacy.processed_retention_days),
            )
        self.mark_run(guild_id)
        if expired or forgotten or pruned:
            logger.info(
                "maintenance guild=%s expired=%d forgotten=%d pruned=%d",
                guild_id,
                expired,
                forgotten,
                pruned,
            )
        return MaintenanceReport(
            guild_id=guild_id,
            expired=expired,
            forgotten=forgotten,
            pruned=pruned,
        )


__all__ = ["MaintenanceReport", "MaintenanceService"]
