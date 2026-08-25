"""discord.py integration tests using stub objects (no gateway/network)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("discord")

from discord_memory.integrations.discord_py import (
    MemoryCog,
    _event_from_message,
    setup_discord_memory,
)
from tests.conftest import make_config


def _stub_member(member_id: int, name: str, display: str | None = None, bot: bool = False):
    return SimpleNamespace(
        id=member_id,
        name=name,
        display_name=display or name,
        bot=bot,
    )


def _stub_message(**overrides):
    author = overrides.get("author", _stub_member(123, "alice", "Alice"))
    guild = overrides.get("guild", SimpleNamespace(id=555))
    channel = overrides.get("channel", SimpleNamespace(id=777))
    mentions = overrides.get("mentions", [])
    reference = overrides.get("reference")
    message = SimpleNamespace(
        id=900,
        author=author,
        guild=guild,
        channel=channel,
        content=overrides.get("content", "hello from the stub"),
        created_at=datetime.now(UTC),
        mentions=mentions,
        reference=reference,
        thread=None,
    )
    return message


class TestEventConversion:
    def test_full_mapping(self) -> None:
        bob = _stub_member(456, "bob")
        message = _stub_message(
            mentions=[bob],
            reference=SimpleNamespace(message_id=888),
            content="hey @bob",
        )
        event = _event_from_message(message)
        assert event.message_id == "900"
        assert event.guild_id == "555"
        assert event.author_id == "123"
        assert event.mention_ids == ("456",)
        assert event.reply_to_message_id == "888"
        assert not event.author_is_bot

    def test_bot_author_flagged(self) -> None:
        bot_author = _stub_member(1, "somebot", bot=True)
        event = _event_from_message(_stub_message(author=bot_author))
        assert event.author_is_bot

    def test_naive_created_at_falls_back_to_utc_now(self) -> None:
        message = _stub_message()
        message.created_at = datetime.now().replace(tzinfo=None)  # naive
        event = _event_from_message(message)
        assert event.created_at.tzinfo is not None


class TestSetupAndCog:
    async def test_setup_registers_listeners_and_starts(self, make_client, fixed_clock):
        listeners: dict[str, list] = {}

        class StubBot:
            user = _stub_member(42, "memboto", bot=True)

            def listen(self, name):
                def decorator(fn):
                    listeners.setdefault(name, []).append(fn)
                    return fn

                return decorator

        config = make_config()
        memory, cog = await setup_discord_memory(
            StubBot(),
            config,
            clock=fixed_clock,
            llm=None,
        )
        assert set(listeners) == {"on_message", "on_member_update", "on_ready"}
        assert isinstance(cog, MemoryCog)
        assert not memory.started  # on_ready owns startup

        await listeners["on_ready"][0]()
        assert memory.started

        # on_message path observes into the queue
        on_message = listeners["on_message"][0]
        await on_message(_stub_message(content="a substantive chat message here"))
        assert await memory._queue.pending_count("555") == 1

        # on_ready registers the bot's own guard id (already registered here)
        # on_member_update re-indexes renames
        guild_stub = SimpleNamespace(id=555)
        before = _stub_member(123, "alice", "OldName")
        after = _stub_member(123, "alice", "NewName")
        before.guild = guild_stub
        after.guild = guild_stub
        await listeners["on_member_update"][0](before, after)
        aliases = await memory.identity.aliases_of("555", "123")
        assert any(record.alias_norm == "newname" for record in aliases)
        await memory.close()

    async def test_cog_commands(self, make_client, fixed_clock) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        cog = MemoryCog(client)

        sent = {}

        async def respond(text: str, ephemeral: bool = False) -> None:
            sent["text"] = text
            sent["ephemeral"] = ephemeral

        interaction = SimpleNamespace(
            user=_stub_member(123, "alice"),
            guild_id=555,
            response=SimpleNamespace(send_message=respond),
        )
        await cog.remember(interaction, "loves spicy ramen on fridays")
        assert "Noted" in sent["text"]
        facts_page = await client.facts.list_for_subject(
            "555",
            "123",
            include_server=False,
        )
        assert any("ramen" in f.text for f in facts_page.items)

        await cog.me(interaction)
        assert "Your stored memories" in sent["text"]

        await cog.forget_me(interaction)
        assert "purged" in sent["text"]
        await client.close()
