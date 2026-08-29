"""discord.py integration tests using stub objects (no gateway/network)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

pytest.importorskip("discord")

from icelake.integrations.discord_py import (
    MemoryCog,
    _ConfirmPurgeView,
    _event_from_message,
    prompt_from_message,
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


class StubBot:
    def __init__(self) -> None:
        self.user = _stub_member(42, "memboto", bot=True)
        self.listeners: dict[str, list] = {}
        self.cogs: dict[str, object] = {}

    def listen(self, name):
        def decorator(fn):
            self.listeners.setdefault(name, []).append(fn)
            return fn

        return decorator

    async def add_cog(self, cog) -> None:
        self.cogs[type(cog).__name__] = cog


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
    async def test_setup_registers_listeners_cog_and_starts(self, make_client, fixed_clock) -> None:
        bot = StubBot()
        config = make_config()
        memory = await setup_discord_memory(bot, config, clock=fixed_clock, llm=None)
        assert set(bot.listeners) == {"on_message", "on_member_update", "on_ready"}
        assert isinstance(bot.cogs.get("MemoryCog"), MemoryCog)
        assert not memory.started  # on_ready owns startup

        await bot.listeners["on_ready"][0]()
        assert memory.started

        on_message = bot.listeners["on_message"][0]
        await on_message(_stub_message(content="a substantive chat message here"))
        assert await memory._queue.pending_count("555") == 1

        guild_stub = SimpleNamespace(id=555)
        before = _stub_member(123, "alice", "OldName")
        after = _stub_member(123, "alice", "NewName")
        before.guild = guild_stub
        after.guild = guild_stub
        await bot.listeners["on_member_update"][0](before, after)
        aliases = await memory.identity.aliases_of("555", "123")
        assert any(record.alias_norm == "newname" for record in aliases)
        await memory.close()

    async def test_cog_commands(self, make_client, fixed_clock) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        cog = MemoryCog(client)

        sent = {}

        async def respond(text: str, ephemeral: bool = False, view=None) -> None:
            sent["text"] = text
            sent["ephemeral"] = ephemeral
            sent["view"] = view

        interaction = SimpleNamespace(
            user=_stub_member(123, "alice"),
            guild_id=555,
            response=SimpleNamespace(send_message=respond),
        )
        await cog.remember.callback(cog, interaction, "loves spicy ramen on fridays")
        assert "Noted" in sent["text"]
        facts_page = await client.facts.list_for_subject(
            "555",
            "123",
            include_server=False,
        )
        assert any("ramen" in f.text for f in facts_page.items)

        await cog.me.callback(cog, interaction)
        assert "Your stored memories" in sent["text"]
        await client.close()

    async def test_forget_is_two_phase(self, make_client) -> None:
        """Preview first; nothing is purged until the confirm button fires."""
        client, _ = make_client(llm=False)
        await client.start()
        await client.facts.remember(
            guild_id="555",
            subject_id="123",
            text="loves spicy ramen on fridays",
            actor_id="123",
        )
        cog = MemoryCog(client)

        sent = {}

        async def respond(text: str, ephemeral: bool = False, view=None) -> None:
            sent["text"] = text
            sent["view"] = view

        interaction = SimpleNamespace(
            user=_stub_member(123, "alice"),
            guild_id=555,
            response=SimpleNamespace(send_message=respond),
        )
        await cog.forget.callback(cog, interaction)
        assert "permanently purge" in sent["text"]
        assert isinstance(sent["view"], _ConfirmPurgeView)
        # Preview alone must not delete anything.
        page = await client.facts.list_for_subject("555", "123", include_server=False)
        assert len(page.items) == 1

        edited = {}

        async def edit_message(content: str, view=None) -> None:
            edited["content"] = content

        confirm_interaction = SimpleNamespace(
            user=_stub_member(123, "alice"),
            guild_id=555,
            response=SimpleNamespace(edit_message=edit_message, send_message=respond),
        )
        await sent["view"].confirm.callback(confirm_interaction)
        assert "Purged 1 memories" in edited["content"]
        page = await client.facts.list_for_subject("555", "123", include_server=False)
        assert len(page.items) == 0
        await client.close()

    async def test_forget_empty_memory_short_circuits(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        cog = MemoryCog(client)
        sent = {}

        async def respond(text: str, ephemeral: bool = False, view=None) -> None:
            sent["text"] = text

        interaction = SimpleNamespace(
            user=_stub_member(123, "alice"),
            guild_id=555,
            response=SimpleNamespace(send_message=respond),
        )
        await cog.forget.callback(cog, interaction)
        assert sent["text"] == "I hold no memories of you."
        await client.close()

    async def test_prompt_from_message_maps_and_builds(self, make_client) -> None:
        client, _ = make_client(llm=False)
        await client.start()
        bob = _stub_member(456, "bob")
        ctx = await prompt_from_message(
            client,
            _stub_message(content="what do you think of @bob?", mentions=[bob]),
        )
        assert ctx.injection_block.startswith("[MEMORY CONTEXT]")
        await client.close()
