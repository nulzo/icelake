"""LLM, embedding and metering ports."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from discord_memory.models.admin import BudgetStep, MeterSnapshot
from discord_memory.models.common import FrozenModel


class LlmMessage(FrozenModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(FrozenModel):
    """One completion request. ``json_mode`` asks the provider for JSON output."""

    messages: tuple[LlmMessage, ...]
    temperature: float = 0.0
    max_tokens: int = 1024
    json_mode: bool = False
    purpose: str = "general"
    timeout_seconds: float | None = None


class ChatResponse(FrozenModel):
    text: str
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


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
        purpose: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> None: ...

    def increment(self, name: str, value: float = 1.0) -> None: ...

    def check_budget(self, guild_id: str) -> BudgetStep:
        """Current degradation step for a guild's spend budget."""

    def snapshot(self) -> MeterSnapshot: ...
