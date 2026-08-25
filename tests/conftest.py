"""Shared fixtures and deterministic fakes for the test suite."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from discord_memory.api.client import DiscordMemory
from discord_memory.config import MemoryConfig
from discord_memory.models.events import MessageEvent
from discord_memory.ports.clock import FixedClock, IdGen, SystemClock, UlidIdGen
from discord_memory.ports.llm import ChatRequest, ChatResponse


class ScriptedLLM:
    """Deterministic ChatLLM fake: maps purpose → canned JSON responses."""

    def __init__(
        self,
        responses: dict[str, str | Callable[[ChatRequest], str]] | None = None,
    ) -> None:
        self.responses = responses or {}
        self.calls: list[ChatRequest] = []

    @property
    def model_name(self) -> str:
        return "scripted-test-model"

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        handler = self.responses.get(request.purpose)
        if handler is None:
            return ChatResponse(
                text="{}", prompt_tokens=10, completion_tokens=5, model=self.model_name
            )
        text = handler(request) if callable(handler) else handler
        return ChatResponse(
            text=text,
            prompt_tokens=len(request.messages[0].content) // 4,
            completion_tokens=len(text) // 4,
            model=self.model_name,
        )


def extraction_response(operations: list[dict]) -> str:
    return json.dumps({"operations": operations})


class ExplodingLLM(ScriptedLLM):
    """LLM that always raises — dead-letter path testing."""

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.calls.append(request)
        raise RuntimeError("llm provider exploded")


class SeqIdGen(IdGen):
    """Sequential ids for readable assertions."""

    def __init__(self) -> None:
        self._counters: dict[str, int] = {}
        self._delegate = UlidIdGen()

    def new_id(self, prefix: str) -> str:
        self._counters[prefix] = self._counters.get(prefix, 0) + 1
        return f"{prefix}_test_{self._counters[prefix]:04d}"


@pytest.fixture()
def fixed_clock():
    return FixedClock(datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC))


def make_config(**overrides) -> MemoryConfig:
    values: dict = {
        "storage": "sqlite://:memory:",
        "batching": {
            "batch_size_messages": 3,
            "max_age_seconds": 60,
            "lease_seconds": 120,
            "min_interval_seconds": 0,
        },
        "workers": {"enabled": False},
        "embeddings": "hashing",
        "extraction": {"noise_gate": True},
    }
    values.update(overrides)
    return MemoryConfig(**values)


@pytest.fixture()
def make_client(fixed_clock):
    """Factory producing fully-wired clients with scripted LLMs.

    ``llm=False`` builds a client with NO llm (degraded-mode paths).
    """

    def _make(
        llm=None, config: MemoryConfig | None = None, id_gen: IdGen | None = None
    ) -> tuple[DiscordMemory, ScriptedLLM | None]:
        if llm is False:
            client = DiscordMemory(
                config or make_config(),
                clock=fixed_clock,
                id_gen=id_gen or SeqIdGen(),
                llm=None,
            )
            return client, None
        resolved_llm = (
            llm
            if llm is not None
            else ScriptedLLM(
                {
                    "extraction": extraction_response([]),
                }
            )
        )
        client = DiscordMemory(
            config or make_config(),
            clock=fixed_clock,
            id_gen=id_gen or SeqIdGen(),
            llm=resolved_llm,
        )
        return client, resolved_llm

    return _make


@pytest.fixture()
def event_factory(fixed_clock):
    counter = {"n": 0}

    def _make(
        *,
        author_id: str = "100000000000000001",
        content: str = "",
        guild_id: str = "500000000000000001",
        mentions: tuple[str, ...] = (),
        display_name: str = "alice",
        username: str = "alice_u",
    ) -> MessageEvent:
        counter["n"] += 1
        return MessageEvent(
            message_id=f"900000000000000{counter['n']:03d}",
            guild_id=guild_id,
            channel_id="700000000000000001",
            author_id=author_id,
            content=content,
            created_at=fixed_clock.now(),
            author_display_name=display_name,
            author_username=username,
            mention_ids=mentions,
        )

    return _make


SYSTEM_CLOCK = SystemClock()
