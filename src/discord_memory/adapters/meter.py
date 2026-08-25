"""In-process meter with budget enforcement and graceful degradation ladder.

Budgets are per-guild, per-process (documented limitation; cross-process budgets need
store-backed counters in a future release). ``check_budget`` returns the active
degradation step so the pipeline can skip work before spending.
"""

from __future__ import annotations

import threading

from discord_memory.config import BudgetsConfig
from discord_memory.models.admin import BudgetStep, MeterSnapshot
from discord_memory.ports.clock import Clock

_WARN_FRACTION = 0.8


class InMemoryMeter:
    """Thread-safe counters by purpose plus daily/monthly guild budgets."""

    def __init__(self, budgets: BudgetsConfig, clock: Clock) -> None:
        self._budgets = budgets
        self._clock = clock
        self._lock = threading.Lock()
        self._calls: dict[str, int] = {}
        self._prompt_tokens: dict[str, int] = {}
        self._completion_tokens: dict[str, int] = {}
        self._counters: dict[str, float] = {}
        self._guild_day: dict[tuple[str, str], int] = {}
        self._guild_month: dict[tuple[str, str], int] = {}

    def record_llm(
        self,
        purpose: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None:
        del model
        with self._lock:
            self._calls[purpose] = self._calls.get(purpose, 0) + 1
            self._prompt_tokens[purpose] = self._prompt_tokens.get(purpose, 0) + prompt_tokens
            self._completion_tokens[purpose] = (
                self._completion_tokens.get(purpose, 0) + completion_tokens
            )

    def charge_guild(self, guild_id: str, *, prompt_tokens: int) -> None:
        """Attribute prompt spend to a guild for budget accounting."""
        now = self._clock.now()
        day_key = (guild_id, now.strftime("%Y-%m-%d"))
        month_key = (guild_id, now.strftime("%Y-%m"))
        with self._lock:
            self._guild_day[day_key] = self._guild_day.get(day_key, 0) + prompt_tokens
            self._guild_month[month_key] = self._guild_month.get(month_key, 0) + prompt_tokens

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    def check_budget(self, guild_id: str) -> BudgetStep:
        """Current degradation step for a guild based on configured ceilings."""
        now = self._clock.now()
        day_key = (guild_id, now.strftime("%Y-%m-%d"))
        month_key = (guild_id, now.strftime("%Y-%m"))
        with self._lock:
            day_used = self._guild_day.get(day_key, 0)
            month_used = self._guild_month.get(month_key, 0)
            daily = self._budgets.guild_daily_prompt_tokens
            monthly = self._budgets.guild_monthly_prompt_tokens
        over_daily = daily is not None and day_used >= daily
        over_monthly = monthly is not None and month_used >= monthly
        if over_daily or over_monthly:
            return BudgetStep.SKIP_EXTRACTION
        near_daily = daily is not None and day_used >= _WARN_FRACTION * daily
        near_monthly = monthly is not None and month_used >= _WARN_FRACTION * monthly
        if near_daily or near_monthly:
            return BudgetStep.SKIP_RECONCILE
        return BudgetStep.NONE

    def snapshot(self) -> MeterSnapshot:
        with self._lock:
            return MeterSnapshot(
                calls=dict(self._calls),
                prompt_tokens=dict(self._prompt_tokens),
                completion_tokens=dict(self._completion_tokens),
                estimated_cost_usd={},
            )


__all__ = ["InMemoryMeter"]
