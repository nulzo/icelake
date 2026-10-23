"""Usage meter: in-process observability counters, store-backed guild budgets.

Call/token/cost counters are per-process — at N workers, aggregate them
externally (they are observability, not enforcement). Budget counters live in
the store so the degradation ladder is correct across processes.
``check_budget`` returns the active degradation step so the pipeline can skip
work before spending.
"""

from __future__ import annotations

import threading
from uuid import uuid4

from icelake.config import BudgetsConfig
from icelake.errors import BudgetExceededError
from icelake.models.admin import BudgetStep, MeterPurpose, MeterSnapshot
from icelake.ports.clock import Clock
from icelake.ports.llm import BudgetReservation, ChatLLM, ChatRequest, ChatResponse, Meter
from icelake.ports.store import MemoryStore

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
        # UTF-8 bytes conservatively bound text tokens; schema/framing get headroom.
        # Failed/cancelled calls retain the reservation because billing is unknown.
        reserved = None
        if request.guild_id:
            reserved = await self._meter.reserve(
                request.guild_id,
                prompt_tokens=len(request.model_dump_json().encode()) + 512,
            )
        response = await self._inner.complete(request)
        self._meter.record_llm(
            request.purpose,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            model=self._inner.model_name,
            cost_usd=response.cost_usd,
        )
        if reserved is not None:
            await self._meter.settle(reserved, prompt_tokens=response.prompt_tokens)
        return response


class UsageMeter:
    """Per-process counters by purpose plus store-backed guild budgets."""

    def __init__(
        self,
        budgets: BudgetsConfig,
        clock: Clock,
        *,
        store: MemoryStore,
        pricing_usd_per_mtok: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._budgets = budgets
        self._clock = clock
        self._store = store
        self._pricing = pricing_usd_per_mtok or DEFAULT_USD_PER_MTOK
        self._lock = threading.Lock()
        self._calls: dict[str, int] = {}
        self._prompt_tokens: dict[str, int] = {}
        self._completion_tokens: dict[str, int] = {}
        self._cost_usd: dict[str, float] = {}
        self._models: dict[str, str] = {}
        self._counters: dict[str, float] = {}

    def record_llm(
        self,
        purpose: MeterPurpose | str,
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

    async def reserve(self, guild_id: str, *, prompt_tokens: int) -> BudgetReservation:
        now = self._clock.now()
        reservation = BudgetReservation(
            guild_id=guild_id,
            day_key=now.strftime("%Y-%m-%d"),
            month_key=now.strftime("%Y-%m"),
            prompt_tokens=prompt_tokens,
        )
        async with self._store.transaction():
            await self._store.set_cursor(guild_id, "budget_fence", uuid4().hex)
            day, month = await self._store.guild_token_usage(
                guild_id, day_key=reservation.day_key, month_key=reservation.month_key
            )
            for used, limit in (
                (day, self._budgets.guild_daily_prompt_tokens),
                (month, self._budgets.guild_monthly_prompt_tokens),
            ):
                if limit is not None and used + prompt_tokens > limit:
                    raise BudgetExceededError("memory prompt budget exhausted")
            await self._store.charge_guild_tokens(
                guild_id,
                day_key=reservation.day_key,
                month_key=reservation.month_key,
                prompt_tokens=prompt_tokens,
            )
        return reservation

    async def settle(self, reservation: BudgetReservation, *, prompt_tokens: int) -> None:
        async with self._store.transaction():
            await self._store.charge_guild_tokens(
                reservation.guild_id,
                day_key=reservation.day_key,
                month_key=reservation.month_key,
                prompt_tokens=prompt_tokens - reservation.prompt_tokens,
            )

    async def charge_guild(self, guild_id: str, *, prompt_tokens: int) -> None:
        """Attribute prompt spend to a guild for budget accounting."""
        now = self._clock.now()
        await self._store.charge_guild_tokens(
            guild_id,
            day_key=now.strftime("%Y-%m-%d"),
            month_key=now.strftime("%Y-%m"),
            prompt_tokens=prompt_tokens,
        )

    def increment(self, name: str, value: float = 1.0) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0.0) + value

    async def check_budget(self, guild_id: str) -> BudgetStep:
        """Current degradation step for a guild based on configured ceilings."""
        now = self._clock.now()
        day_used, month_used = await self._store.guild_token_usage(
            guild_id,
            day_key=now.strftime("%Y-%m-%d"),
            month_key=now.strftime("%Y-%m"),
        )
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
                counters=dict(self._counters),
                calls=dict(self._calls),
                prompt_tokens=dict(self._prompt_tokens),
                completion_tokens=dict(self._completion_tokens),
                estimated_cost_usd={k: round(v, 6) for k, v in self._cost_usd.items()},
            )


__all__ = ["MeteredLLM", "UsageMeter"]
