"""Live-ACL adapter semantics with HTTP fakes, including private-thread membership."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from icelake.integrations.discord_permissions import (
    DiscordChannelAuthorizer,
    history_visible_to_all,
)
from icelake.models.facts import FactRecord, SourceRef
from icelake.permissions import Audience, AudienceGate


def surface():
    role = SimpleNamespace(permissions=discord.Permissions(read_message_history=True))
    guild = SimpleNamespace(default_role=role, fetch_channel=AsyncMock(), fetch_member=AsyncMock())
    parent = MagicMock(spec=discord.TextChannel)
    parent.guild = guild
    parent.overwrites = {}
    parent.overwrites_for.return_value = discord.PermissionOverwrite()
    parent.permissions_for.return_value = discord.Permissions(
        view_channel=True, read_message_history=True
    )
    guild.fetch_channel.return_value = parent
    client = SimpleNamespace(user=SimpleNamespace(id=42), fetch_guild=AsyncMock(return_value=guild))
    return client, guild, parent


async def test_fresh_role_and_member_resolution():
    client, guild, _ = surface()
    result = await DiscordChannelAuthorizer(client).allowed_channels(
        "555", Audience(channel_id="777", requester_id="123"), frozenset({"777", "private"})
    )
    assert result == {"777"}
    client.fetch_guild.assert_awaited_once_with(555)
    assert guild.fetch_member.await_count == 2


@pytest.mark.parametrize("kind", ["base_denied", "overwrite_denied", "requester_denied"])
async def test_unprovable_audiences_deny(kind):
    client, guild, parent = surface()
    if kind == "base_denied":
        guild.default_role.permissions.read_message_history = False
    elif kind == "overwrite_denied":
        parent.overwrites = {"some_member": discord.PermissionOverwrite(read_message_history=False)}
    else:
        parent.permissions_for.return_value = discord.Permissions.none()
    assert not await DiscordChannelAuthorizer(client).allowed_channels(
        "555", Audience(channel_id="777", requester_id="123"), frozenset({"777"})
    )


async def test_private_thread_membership_failure_is_closed():
    client, guild, parent = surface()
    thread = MagicMock(spec=discord.Thread)
    thread.parent_id = 888
    thread.is_private.return_value = True
    thread.fetch_member = AsyncMock(side_effect=RuntimeError("membership unknown"))
    guild.fetch_channel.side_effect = [thread, parent]
    record = FactRecord(
        id="f",
        guild_id="555",
        text="private fact",
        citations=(SourceRef(message_id="m", guild_id="555", channel_id="777", author_id="123"),),
    )
    assert not await AudienceGate(DiscordChannelAuthorizer(client)).filter(
        "555", Audience(channel_id="777", requester_id="123"), (record,)
    )
    thread.fetch_member.assert_awaited_once_with(123)


def test_everyone_history_allow_is_sufficient_without_denies():
    _, guild, parent = surface()
    guild.default_role.permissions.read_message_history = False
    parent.overwrites_for.return_value = discord.PermissionOverwrite(read_message_history=True)
    assert history_visible_to_all(parent)
