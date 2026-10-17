"""Keeps the runnable examples honest: they must execute against the real API."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
sys.path.insert(0, str(EXAMPLES.parent))


def test_citations_example_runs_end_to_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("examples.citations")
    asyncio = pytest.importorskip("asyncio")
    asyncio.run(module.main())
    out = capsys.readouterr().out
    assert "same URL returns the same source: True" in out
    assert "[[1]]" in out
    assert "the rust book never matched a span" in out
    assert "[3] bob" in out
    assert "persona voice cited: False" in out
    assert "two facts pointing at the same message keep number [1]" in out
    assert "footer numbers must walk Citations.citable" in out


def test_lifecycle_example_runs_end_to_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("examples.lifecycle")
    asyncio = pytest.importorskip("asyncio")
    asyncio.run(module.main())
    out = capsys.readouterr().out
    assert "'ali' -> 100000000000000001" in out
    assert "history after update+forget" in out
    assert "FactCommitted fired: 1" in out
    assert "action=remember" in out
    assert "observe after opt-out ignored=True" in out
    assert "import into a fresh store: 2 facts" in out  # opted-out evidence is excluded
    assert "healthy=True" in out


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


def test_name_lookup_tool_example_runs_end_to_end(
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module("examples.name_lookup_tool")
    asyncio = pytest.importorskip("asyncio")
    asyncio.run(module.main())
    out = capsys.readouterr().out
    # strict profile: bob's own facts, and NOT carol's claim about him
    profile = out.split("CONTRAST")[0]
    assert "minecraft" in profile
    assert "argued over the game night loss" not in profile
    # ambiguity never guesses; unknown names get the mention fallback
    assert "more than one member" in out
    assert "don't recognize" in out
    # contrast section proves recall IS cross-subject by design
    assert "[subject=carol] carol and bob argued over the game night loss" in out


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

    from icelake import DiscordMemory, MemoryConfig, MessageEvent
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
        MemoryConfig(
            storage="sqlite://:memory:",
            workers={"enabled": False},
            access={"require_audience": False},
        ),
        llm=llm,
    )

    async def flow() -> str:
        async with memory:
            await memory.observe(
                MessageEvent(
                    message_id="1",
                    guild_id="g",
                    channel_id="c",
                    author_id="100000000000000001",
                    content="@bot bob is such a hacker lol",
                    created_at=datetime.now(UTC),
                    author_display_name="alice",
                    mention_ids=("200000000000000002",),
                )
            )
            await memory.flush()
            ctx = await memory.prompt_context(
                guild_id="g",
                asker_id="300000000000000003",
                text="what happened between alice and bob?",
                mentioned_ids=("100000000000000001", "200000000000000002"),
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

    from icelake import Citations, DiscordMemory, MemoryConfig, MessageEvent
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
            ),
        }
    )
    memory = DiscordMemory(
        MemoryConfig(
            storage="sqlite://:memory:",
            workers={"enabled": False},
            access={"require_audience": False},
        ),
        llm=llm,
    )

    async def flow() -> tuple[str, int]:
        memory.register_bot_id(42)
        async with memory:
            for mid, author_id, name, content, mentions in (
                (
                    "1",
                    "100000000000000001",
                    "alice",
                    "everyone saw bob cheating in that ranked match honestly",
                    ("200000000000000002",),
                ),
                (
                    "2",
                    "200000000000000002",
                    "bob",
                    "that loss was absolutely not fair and you all know it",
                    (),
                ),
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
                asker_id="300000000000000003",
                text="what happened between alice and bob?",
                mentioned_ids=("100000000000000001", "200000000000000002"),
            )
            reply = "bob was called a hacker by alice during the ranked match"
            set_ = Citations.from_prompt_context(ctx)
            cited = await memory.cite(reply, set_)
            return ctx.injection_block, len(cited.text)

    block, link_len = asyncio.run(flow())
    assert "REFERENCED USER" in block
    assert "SERVER COMMUNITY FACTS" in block
    # server fact deduped across both batches (regression guard)
    assert block.count("bonds over late night") == 1
    # attribution wove a jump link into the plain reply
    assert link_len > len("bob was called a hacker during the match.")


@pytest.mark.parametrize("command", ["show", "lookup"])
async def test_live_bot_profile_commands_enforce_channel_audience(command):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from icelake import DiscordMemory, MemoryConfig, SourceRef

    module = importlib.import_module("examples.omni_style_bot")
    memory = DiscordMemory(MemoryConfig(storage="sqlite://:memory:", workers={"enabled": False}))
    bot = module.OmniStyleBot(memory)
    async with memory:
        for channel, text in (("10", "likes coffee"), ("20", "private address is secret")):
            await memory.facts.remember(
                guild_id="1",
                subject_id="2",
                text=text,
                source_refs=(
                    SourceRef(
                        message_id=channel,
                        guild_id="1",
                        channel_id=channel,
                        author_id="2",
                    ),
                ),
            )
        interaction = SimpleNamespace(
            guild_id=1,
            channel_id=10,
            user=SimpleNamespace(id=3),
            response=SimpleNamespace(send_message=AsyncMock()),
        )
        if command == "show":
            await module.OmniStyleBot.memory_show.callback(
                bot,
                interaction,
                SimpleNamespace(id=2, display_name="Alice"),
            )
        else:
            bot._resolve_lookup = AsyncMock(return_value=("2", "Alice"))
            await module.OmniStyleBot.memory_lookup.callback(bot, interaction, "Alice")
        sent = interaction.response.send_message.call_args.args[0]
        assert "likes coffee" in sent
        assert "private address" not in sent
        assert interaction.response.send_message.call_args.kwargs["ephemeral"]
    await bot.close()
