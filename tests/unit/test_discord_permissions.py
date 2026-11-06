"""Effective Discord permissions, including the production role-only regression."""

from itertools import combinations, product
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest

from icelake.integrations.discord_permissions import (
    DiscordChannelAuthorizer,
    history_visible_to_all,
)
from icelake.permissions import Audience

VIEW = discord.Permissions(view_channel=True)
HISTORY = discord.Permissions(read_message_history=True)
BOTH = discord.Permissions(view_channel=True, read_message_history=True)
ADMIN = discord.Permissions(administrator=True)
AUDIENCE = Audience(channel_id="777", requester_id="123")


def channel(guild, *, cid=777, overwrites=()):
    return discord.TextChannel(
        state=guild._state,
        guild=guild,
        data={
            "id": str(cid),
            "name": "general",
            "type": 0,
            "position": 0,
            "permission_overwrites": list(overwrites),
        },
    )


def member(guild, mid, roles):
    return discord.Member(
        guild=guild,
        state=guild._state,
        data={
            "user": {"id": str(mid), "username": "reader", "discriminator": "0", "avatar": None},
            "roles": [str(r) for r in roles],
            "flags": 0,
        },
    )


def overwrite(target, *, allow=0, deny=0, kind=0):
    return {"id": str(target), "type": kind, "allow": str(allow), "deny": str(deny)}


def surface(monkeypatch, *, base=None, roles=None, overwrites=()):
    # Use real discord.py objects so combined roles, administrator and implicit
    # permissions are evaluated by the same machinery as the deployed adapter.
    transport = discord.Client(intents=discord.Intents(members=True))
    role_pairs = roles if roles is not None else [(666, BOTH)]
    guild = discord.Guild(
        data={
            "id": "555",
            "name": "test",
            "owner_id": "999",
            "roles": [
                {"id": str(rid), "name": str(rid), "permissions": str(perms.value)}
                for rid, perms in [(555, base or discord.Permissions.none()), *role_pairs]
            ],
        },
        state=transport._connection,
    )
    parent = channel(guild, overwrites=overwrites)
    members = {mid: member(guild, mid, [rid for rid, _ in role_pairs]) for mid in (123, 42)}
    fetch_channel = AsyncMock(return_value=parent)
    fetch_channels = AsyncMock(return_value=[])
    fetch_member = AsyncMock(side_effect=lambda mid: members[mid])
    scans = []

    async def fetch_members(self, *, limit):
        scans.append(limit)
        for value in list(members.values())[:limit]:
            yield value

    monkeypatch.setattr(discord.Guild, "fetch_channel", fetch_channel)
    monkeypatch.setattr(discord.Guild, "fetch_channels", fetch_channels)
    monkeypatch.setattr(discord.Guild, "fetch_member", fetch_member)
    monkeypatch.setattr(discord.Guild, "fetch_members", fetch_members)
    client = SimpleNamespace(user=SimpleNamespace(id=42), fetch_guild=AsyncMock(return_value=guild))
    return SimpleNamespace(
        client=client,
        guild=guild,
        parent=parent,
        members=members,
        scans=scans,
        fetch_channel=fetch_channel,
        fetch_channels=fetch_channels,
        fetch_member=fetch_member,
    )


async def test_role_only_history_works_without_member_enumeration(monkeypatch):
    s = surface(monkeypatch)
    assert not s.guild.default_role.permissions.read_message_history
    assert history_visible_to_all(s.parent)
    assert await DiscordChannelAuthorizer(s.client).allowed_channels(
        "555", AUDIENCE, frozenset({"777"})
    ) == {"777"}
    assert not s.scans
    assert s.fetch_member.await_count == 2


async def test_administrators_and_unused_role_do_not_hide_all_memories(monkeypatch):
    s = surface(monkeypatch, roles=[(666, VIEW), (667, ADMIN)])
    assert not history_visible_to_all(s.parent)
    for mid in range(100, 141):
        s.members[mid] = member(s.guild, mid, [667])
    assert await DiscordChannelAuthorizer(s.client).allowed_channels(
        "555", AUDIENCE, frozenset({"777"})
    ) == {"777"}
    assert s.scans == [10001]


async def test_one_actual_reader_without_history_denies_disclosure(monkeypatch):
    s = surface(monkeypatch, base=VIEW, roles=[(666, HISTORY)])
    s.members[321] = member(s.guild, 321, [])
    assert not await DiscordChannelAuthorizer(s.client).allowed_channels(
        "555", AUDIENCE, frozenset({"777"})
    )


async def test_hidden_member_history_deny_does_not_block_other_readers(monkeypatch):
    s = surface(monkeypatch, overwrites=[overwrite(321, deny=HISTORY.value, kind=1)])
    s.members[321] = member(s.guild, 321, [])
    assert not history_visible_to_all(s.parent)
    assert await DiscordChannelAuthorizer(s.client).allowed_channels(
        "555", AUDIENCE, frozenset({"777"})
    ) == {"777"}


async def test_member_view_grant_without_history_requires_snapshot(monkeypatch):
    s = surface(monkeypatch, overwrites=[overwrite(321, allow=VIEW.value, kind=1)])
    s.members[321] = member(s.guild, 321, [])
    assert not history_visible_to_all(s.parent)
    assert not await DiscordChannelAuthorizer(s.client).allowed_channels(
        "555", AUDIENCE, frozenset({"777"})
    )


async def test_requester_and_bot_must_have_access(monkeypatch):
    s = surface(monkeypatch)
    for mid in (123, 42):
        original = s.members[mid]
        s.members[mid] = member(s.guild, mid, [])
        assert not await DiscordChannelAuthorizer(s.client).allowed_channels(
            "555", AUDIENCE, frozenset({"777"})
        )
        s.members[mid] = original


async def test_cross_channel_uses_entire_destination_audience(monkeypatch):
    s = surface(monkeypatch, roles=[(666, BOTH), (667, discord.Permissions.none())])
    source = channel(
        s.guild,
        cid=888,
        overwrites=[overwrite(555, deny=VIEW.value), overwrite(667, allow=BOTH.value)],
    )
    s.fetch_channels.return_value = [source]
    policy = DiscordChannelAuthorizer(s.client)
    assert await policy.allowed_channels("555", AUDIENCE, frozenset({"777", "888"})) == {
        "777",
        "888",
    }
    s.members[321] = member(s.guild, 321, [666])
    assert await policy.allowed_channels("555", AUDIENCE, frozenset({"777", "888"})) == {"777"}


async def test_public_sources_need_no_member_scan(monkeypatch):
    s = surface(monkeypatch, base=BOTH)
    s.fetch_channels.return_value = [channel(s.guild, cid=888)]
    assert await DiscordChannelAuthorizer(s.client).allowed_channels(
        "555", AUDIENCE, frozenset({"888"})
    ) == {"888"}
    assert not s.scans


async def test_membership_bound_is_not_mistaken_for_complete_audience(monkeypatch):
    s = surface(monkeypatch, roles=[(666, VIEW), (667, ADMIN)])
    with pytest.raises(RuntimeError, match="bounded"):
        await DiscordChannelAuthorizer(s.client, max_members=1).allowed_channels(
            "555", AUDIENCE, frozenset({"777"})
        )
    with pytest.raises(ValueError):
        DiscordChannelAuthorizer(s.client, max_members=0)


async def test_empty_membership_snapshot_cannot_authorize(monkeypatch):
    s = surface(monkeypatch, roles=[(666, VIEW), (667, ADMIN)])
    s.fetch_member.side_effect = None
    s.fetch_member.return_value = s.members[123]
    s.members.clear()
    with pytest.raises(RuntimeError, match="no readers"):
        await DiscordChannelAuthorizer(s.client).allowed_channels(
            "555", AUDIENCE, frozenset({"777"})
        )


async def test_private_thread_requires_requester_and_bot_membership(monkeypatch):
    s = surface(monkeypatch)
    from unittest.mock import MagicMock

    thread = MagicMock(spec=discord.Thread)
    thread.parent_id = 888
    thread.is_private.return_value = True
    thread.fetch_member = AsyncMock(side_effect=RuntimeError("membership unknown"))
    s.fetch_channel.side_effect = [thread, s.parent]
    with pytest.raises(RuntimeError, match="membership unknown"):
        await DiscordChannelAuthorizer(s.client).allowed_channels(
            "555", AUDIENCE, frozenset({"777"})
        )
    thread.fetch_member.assert_awaited_once_with(123)


def test_role_proof_against_effective_permissions_for_all_role_combinations(monkeypatch):
    # Exhaust all base/role grants for the two relevant bits, with representative
    # allow/deny overwrites. A successful shortcut must agree for every role set.
    perms = [discord.Permissions.none(), VIEW, HISTORY, BOTH]
    for base, first, second in product(perms, repeat=3):
        for overrides in (
            (),
            (overwrite(666, deny=VIEW.value),),
            (overwrite(555, deny=VIEW.value), overwrite(667, allow=VIEW.value)),
            (overwrite(666, allow=HISTORY.value),),
        ):
            s = surface(
                monkeypatch, base=base, roles=[(666, first), (667, second)], overwrites=overrides
            )
            if not history_visible_to_all(s.parent):
                continue
            for count in range(3):
                for selected in combinations((666, 667), count):
                    effective = s.parent.permissions_for(member(s.guild, 321, selected))
                    assert not effective.view_channel or effective.read_message_history
