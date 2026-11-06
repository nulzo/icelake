"""A provider outage preserves local context and does not repeat within a turn."""

import asyncio

import pytest

from icelake import DiscordMemory
from icelake.adapters.embedders import HashingEmbedder
from icelake.adapters.in_memory.vectors import InMemoryVectorIndex
from icelake.models.retrieval import RecallWarning
from tests.conftest import make_config

GUILD = "500000000000000001"
USER = "100000000000000001"


class ProviderFailure:
    def __init__(self, mode):
        self.mode = mode
        self.active = False
        self.calls = 0

    async def request(self):
        self.calls += 1
        if self.active:
            if self.mode == "timeout":
                await asyncio.Future()
            raise ConnectionError("provider unavailable")


class FlakyEmbedder(HashingEmbedder, ProviderFailure):
    def __init__(self, mode):
        HashingEmbedder.__init__(self)
        ProviderFailure.__init__(self, mode)

    async def embed(self, texts):
        await self.request()
        return await super().embed(texts)


class FlakyReranker(ProviderFailure):
    async def score(self, query, documents):
        await self.request()
        return tuple(0.9 for _ in documents)


@pytest.mark.parametrize("mode", ["error", "timeout"])
@pytest.mark.parametrize("failed_stages", [{"vector"}, {"reranker"}, {"vector", "reranker"}])
async def test_one_failure_per_stage_per_turn_preserves_names_and_local_memories(
    mode, failed_stages, caplog
):
    embedder, reranker = FlakyEmbedder(mode), FlakyReranker(mode)
    config = make_config(
        access={"enabled": False},
        retrieval={"channel_timeout_seconds": 0.1, "reranker": {"timeout_seconds": 0.05}},
    )
    async with DiscordMemory(
        config, llm=None, embedder=embedder, reranker=reranker, vectors=InMemoryVectorIndex()
    ) as memory:
        await memory.facts.remember(
            guild_id=GUILD, subject_id=USER, actor_id=USER, text="call me Nolan"
        )
        await memory.facts.remember(guild_id=GUILD, subject_id=USER, text="enjoys camping")
        await memory.facts.remember(
            guild_id=GUILD, subject_id=None, text="Server members enjoy camping"
        )
        embedder.active = "vector" in failed_stages
        reranker.active = "reranker" in failed_stages
        embedder.calls = reranker.calls = 0
        for turn in (1, 2):
            context = await asyncio.wait_for(
                memory.prompt_context(guild_id=GUILD, asker_id=USER, text="camping"), timeout=2
            )
            assert [n.name for n in context.names] == ["Nolan"]
            assert any(f.fact.subject_id == USER for f in context.facts)
            assert any(f.fact.subject_id is None for f in context.facts)
            assert (RecallWarning.DEGRADED_CHANNEL in context.warnings) is embedder.active
            assert (RecallWarning.RERANKER_UNAVAILABLE in context.warnings) is reranker.active
            assert embedder.calls == turn * (1 if embedder.active else 2)
            assert reranker.calls == turn * (1 if reranker.active else 2)
        before = embedder.calls, reranker.calls
        embedder.active = reranker.active = False
        context = await memory.prompt_context(guild_id=GUILD, asker_id=USER, text="camping")
        assert RecallWarning.DEGRADED_CHANNEL not in context.warnings
        assert RecallWarning.RERANKER_UNAVAILABLE not in context.warnings
        assert (embedder.calls, reranker.calls) == (before[0] + 2, before[1] + 2)
    assert f"error={'TimeoutError' if mode == 'timeout' else 'ConnectionError'}" in caplog.text
    assert "deadline_seconds=" in caplog.text
