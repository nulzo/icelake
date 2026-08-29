"""In-process meter with budget enforcement and graceful degradation ladder.

Budgets are per-guild, per-process (documented limitation; cross-process budgets need
store-backed counters in a future release). ``check_budget`` returns the active
degradation step so the pipeline can skip work before spending.
"""

from __future__ import annotations

import threading

from icelake.config import BudgetsConfig
from icelake.models.admin import BudgetStep, MeterSnapshot
from icelake.ports.clock import Clock
from icelake.ports.llm import ChatLLM, ChatRequest, ChatResponse, Meter

_WARN_FRACTION = 0.8


DEFAULT_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    # model-substring -> (input $/Mtok, output $/Mtok); first match wins
    "gemini-3.7-flash": (0.375, 1.875),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.0-flash": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class MeteredLLM:
    """ChatLLM decorator that records usage per purpose after each call.

    Wired around the config-built LLM in the composition root so production
    token spend is always metered; injected test doubles stay unmetered.
    """

    def __init__(self, inner: ChatLLM, meter: Meter) -> None:
        self._inner = inner
        self._meter = meter

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    async def complete(self, request: ChatRequest) -> ChatResponse:
        response = await self._inner.complete(request)
        self._meter.record_llm(
            request.purpose,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model=self._inner.model_name,
            cost_usd=response.cost_usd,
        )
        if request.guild_id is not None:
            self._meter.charge_guild(request.guild_id, prompt_tokens=response.prompt_tokens)
        return response


class InMemoryMeter:
    """Thread-safe counters by purpose plus daily/monthly guild budgets."""

    def __init__(
        self,
        budgets: BudgetsConfig,
        clock: Clock,
        *,
        pricing_usd_per_mtok: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._budgets = budgets
        self._clock = clock
        self._pricing = pricing_usd_per_mtok or DEFAULT_USD_PER_MTOK
        self._lock = threading.Lock()
        self._calls: dict[str, int] = {}
        self._prompt_tokens: dict[str, int] = {}
        self._completion_tokens: dict[str, int] = {}
        self._cost_usd: dict[str, float] = {}
        self._models: dict[str, str] = {}
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
        cost_usd: float | None = None,
    ) -> None:
        # Provider-reported cost wins; the static table is the fallback for
        # providers that don't report charges.
        cost = (
            cost_usd
            if cost_usd is not None
            else self._estimate_cost(model, prompt_tokens, completion_tokens)
        )
        with self._lock:
            self._calls[purpose] = self._calls.get(purpose, 0) + 1
            self._prompt_tokens[purpose] = self._prompt_tokens.get(purpose, 0) + prompt_tokens
            self._completion_tokens[purpose] = (
                self._completion_tokens.get(purpose, 0) + completion_tokens
            )
            if cost is not None:
                self._cost_usd[purpose] = self._cost_usd.get(purpose, 0.0) + cost

    def _estimate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float | None:
        for needle, (in_price, out_price) in self._pricing.items():
            if needle in model:
                return (
                    prompt_tokens / 1_000_000 * in_price + completion_tokens / 1_000_000 * out_price
                )
        return None

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
                estimated_cost_usd={k: round(v, 6) for k, v in self._cost_usd.items()},
            )


__all__ = ["InMemoryMeter", "MeteredLLM"]
