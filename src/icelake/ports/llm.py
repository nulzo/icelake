"""LLM, embedding and metering ports."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from icelake.models.admin import BudgetStep, MeterPurpose, MeterSnapshot
from icelake.models.common import FrozenModel


class MessageRole(StrEnum):
    """Chat message roles (OpenAI chat-completions vocabulary)."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LlmMessage(FrozenModel):
    role: MessageRole
    content: str


class ChatRequest(FrozenModel):
    """One completion request.

    ``response_schema`` (a JSON-Schema dict) requests *native* structured output
    via ``response_format: {"type": "json_schema"}`` — enforced server-side on
    capable providers, best-effort elsewhere. Omit it for plain-text completions.
    ``temperature=None`` defers to the configured default; if that is also
    ``None`` the parameter is omitted entirely (endpoints that reject it).
    """

    messages: tuple[LlmMessage, ...]
    temperature: float | None = None
    max_tokens: int = 1024
    response_schema: dict[str, object] | None = None
    purpose: MeterPurpose | str = MeterPurpose.GENERAL
    timeout_seconds: float | None = None
    # Budget attribution metadata — providers ignore it; MeteredLLM charges it.
    guild_id: str | None = None


class ChatResponse(FrozenModel):
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Provider-reported charge (OpenRouter usage.cost); None = estimate from the
    # meter's price table instead.
    cost_usd: float | None = None


@runtime_checkable
class ChatLLM(Protocol):
    """OpenAI-chat-completions-compatible port (OpenRouter/OpenAI/Ollama/vLLM all fit).

    Implementations must be async, retry-safe (idempotent calls) and never raise for
    empty completions — return the text and let callers validate.
    """

    async def complete(self, request: ChatRequest) -> ChatResponse: ...

    @property
    def model_name(self) -> str: ...


@runtime_checkable
class LlmCache(Protocol):
    """Content-addressed completion cache (opt-in; sqlite backend ships one)."""

    async def get(self, key: str) -> ChatResponse | None: ...

    async def put(self, key: str, response: ChatResponse) -> None: ...


@runtime_checkable
class Embedder(Protocol):
    """Text-embedding port. Batched; async off-loop by contract."""

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...


@runtime_checkable
class Meter(Protocol):
    """Cost/usage metering with budget enforcement hooks."""

    def record_llm(
        self,
        purpose: MeterPurpose | str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
        cost_usd: float | None = None,
    ) -> None: ...

    def increment(self, name: str, value: float = 1.0) -> None: ...

    async def charge_guild(self, guild_id: str, *, prompt_tokens: int) -> None:
        """Attribute prompt spend to a guild for budget accounting."""
        ...

    async def check_budget(self, guild_id: str) -> BudgetStep:
        """Current degradation step for a guild's spend budget."""

    def snapshot(self) -> MeterSnapshot: ...
