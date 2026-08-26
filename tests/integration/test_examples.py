"""Keeps the runnable examples honest: they must execute against the real API."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(EXAMPLES.parent))


def test_relationship_queries_example_runs_end_to_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("examples.relationship_queries")
    asyncio = pytest.importorskip("asyncio")
    asyncio.run(module.main())
    out = capsys.readouterr().out
    assert "called_out" in out
    assert "stances" in out  # entity aggregation line present
    assert "active memories" in out


def test_ping_reply_bot_module_imports() -> None:
    pytest.importorskip("discord")
    module = importlib.import_module("examples.ping_reply_bot")
    assert hasattr(module, "PingReplyBot")
    assert hasattr(module, "build_memory")

    # mention-stripping helper is pure — verify it works
    stripped = module._strip_mention("<@123> what's up <@123>", 123)
    assert "<@" not in stripped
    assert "what's up" in stripped


def test_prompt_context_separates_asker_and_referenced_users() -> None:
    """The core ping-reply guarantee: facts stay labeled per person."""
    import asyncio
    from datetime import UTC, datetime

    from discord_memory import DiscordMemory, MemoryConfig, MessageEvent
    from tests.conftest import ScriptedLLM, extraction_response

    llm = ScriptedLLM(
        {
            "extraction": extraction_response(
                [
                    {
                        "subject_token": "p1",
                        "speaker_token": "p0",
                        "text": "bob was called a hacker by alice during the ranked match",
                        "category": "relationships",
                        "confidence": 0.9,
                        "source_message_indexes": [1],
                    },
                ]
            )
        }
    )
    memory = DiscordMemory(
        MemoryConfig(storage="sqlite://:memory:", workers={"enabled": False}),
        llm=llm,
    )

    async def flow() -> str:
        async with memory:
            await memory.observe(
                MessageEvent(
                    message_id="1",
                    guild_id="g",
                    channel_id="c",
                    author_id="alice_id",
                    content="@bot bob is such a hacker lol",
                    created_at=datetime.now(UTC),
                    author_display_name="alice",
                    mention_ids=("bob_id",),
                )
            )
            await memory.flush()
            ctx = await memory.prompt_context(
                guild_id="g",
                asker_id="carol_id",
                text="what happened between alice and bob?",
                mentioned_ids=("alice_id", "bob_id"),
            )
        return ctx.injection_block

    block = asyncio.run(flow())
    assert "REFERENCED USER" in block
    assert "hacker" in block


def test_omni_style_bot_module_imports() -> None:
    pytest.importorskip("discord")
    module = importlib.import_module("examples.omni_style_bot")
    assert hasattr(module, "OmniStyleBot")
    assert hasattr(module, "build_memory")

    stripped = module.strip_bot_mention("<@42> hello there", 42)
    assert stripped.strip() == "hello there"


def test_omni_style_turn_with_coreference_and_citations(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """omni's signature behaviors: multi-person context, coreference,
    server facts, and working citation links."""
    import asyncio
    from datetime import UTC, datetime

    from discord_memory import DiscordMemory, MemoryConfig, MessageEvent
    from tests.conftest import ScriptedLLM, extraction_response

    llm = ScriptedLLM(
        {
            "extraction": extraction_response(
                [
                    {
                        "subject_token": "p1",
                        "speaker_token": "p0",
                        "text": "bob was called a hacker by alice during the ranked match",
                        "category": "relationships",
                        "confidence": 0.9,
                        "source_message_indexes": [1],
                        "relations": [
                            {"verb": "called_out", "from_token": "p0", "to_token": "p1"},
                        ],
                    },
                    {
                        "subject_token": "server",
                        "text": "the community bonds over late night ranked gaming sessions",
                        "category": "culture",
                        "confidence": 0.85,
                        "source_message_indexes": [1],
                    },
                ]
            )
        }
    )
    memory = DiscordMemory(
        MemoryConfig(storage="sqlite://:memory:", workers={"enabled": False}),
        llm=llm,
    )

    async def flow() -> tuple[str, int]:
        memory.register_bot_id(42)
        async with memory:
            for mid, author_id, name, content, mentions in (
                (
                    "1",
                    "alice_id",
                    "alice",
                    "everyone saw bob cheating in that ranked match honestly",
                    ("bob_id",),
                ),
                ("2", "bob_id", "bob", "that loss was absolutely not fair and you all know it", ()),
            ):
                await memory.observe(
                    MessageEvent(
                        message_id=mid,
                        guild_id="g1",
                        channel_id="c1",
                        author_id=author_id,
                        content=content,
                        created_at=datetime.now(UTC),
                        author_display_name=name,
                        mention_ids=mentions,
                    )
                )
            await memory.flush()
            ctx = await memory.prompt_context(
                guild_id="g1",
                asker_id="carol_id",
                text="what happened between alice and bob?",
                mentioned_ids=("alice_id", "bob_id"),
            )
            linkified = ctx.apply_citations("Bob was accused of hacking [mem:1] during the match.")
            return ctx.injection_block, len(linkified)

    block, link_len = asyncio.run(flow())
    assert "REFERENCED USER" in block
    assert "SERVER COMMUNITY FACTS" in block
    # server fact deduped across both batches (regression guard)
    assert block.count("bonds over late night") == 1
    # citation resolved to a jump link
    assert link_len > len("[mem:1]")
