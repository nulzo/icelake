"""Conservative live Discord authorization for memory disclosure."""

from __future__ import annotations

import discord

from icelake.permissions import Audience


class DiscordChannelAuthorizer:
    """Use fresh roles, overwrites and membership; deny unprovable audiences.

    Prove simple role policies without enumerating members. Otherwise use a
    bounded, complete REST membership snapshot and discord.py's effective
    permissions, never a partial gateway cache. Decisions are not cached.
    The core AudienceGate supplies the timeout and reports API failures.
    """

    def __init__(self, client: discord.Client, *, max_members: int = 10_000) -> None:
        if max_members < 1:
            raise ValueError("max_members must be positive")
        self._client = client
        self._max_members = max_members

    async def allowed_channels(
        self, guild_id: str, audience: Audience, source_channels: frozenset[str]
    ) -> frozenset[str]:
        # Keep the destination, requester and source checks in one flow so they
        # share fresh guild roles and at most one complete membership snapshot.
        if not source_channels or self._client.user is None:
            return frozenset()
        # fetch_guild constructs roles from the REST response, so permissions do
        # not depend on gateway role/member caches being fully populated.
        guild = await self._client.fetch_guild(int(guild_id))
        channel = await guild.fetch_channel(int(audience.channel_id))
        if not isinstance(channel, (discord.TextChannel, discord.Thread)):
            return frozenset()
        parent = (
            await guild.fetch_channel(channel.parent_id)
            if isinstance(channel, discord.Thread)
            else channel
        )
        if not isinstance(parent, (discord.TextChannel, discord.ForumChannel)):
            return frozenset()
        for user_id in (int(audience.requester_id), self._client.user.id):
            member = await guild.fetch_member(user_id)
            permissions = parent.permissions_for(member)
            if not (permissions.view_channel and permissions.read_message_history):
                return frozenset()
            if isinstance(channel, discord.Thread) and channel.is_private():
                # permissions_for does not establish private-thread membership.
                await channel.fetch_member(user_id)
        readers: tuple[discord.Member, ...] | None = None
        if not history_visible_to_all(parent):
            readers = await self._readers(guild, parent)
            if not all(parent.permissions_for(member).read_message_history for member in readers):
                return frozenset()
        allowed = source_channels & {audience.channel_id}
        if source_channels - allowed:
            # One REST request for all channel overwrites, independent of fact count.
            channels = await guild.fetch_channels()
            for source in channels:
                if (
                    not isinstance(source, discord.TextChannel)
                    or str(source.id) not in source_channels
                ):
                    continue
                if public_history_visible_to_all(source):
                    allowed |= {str(source.id)}
                    continue
                if readers is None:
                    readers = await self._readers(guild, parent)
                if all(_can_read_history(source, member) for member in readers):
                    allowed |= {str(source.id)}
        return frozenset(allowed)

    async def _readers(
        self, guild: discord.Guild, channel: discord.TextChannel | discord.ForumChannel
    ) -> tuple[discord.Member, ...]:
        readers = []
        count = 0
        async for member in guild.fetch_members(limit=self._max_members + 1):
            count += 1
            if count > self._max_members:
                raise RuntimeError("Discord audience exceeds the bounded membership check")
            if channel.permissions_for(member).view_channel:
                readers.append(member)
        if not readers:
            raise RuntimeError("Discord returned no readers for an accessible destination")
        return tuple(readers)


def _can_read_history(channel: discord.TextChannel, member: discord.Member) -> bool:
    permissions = channel.permissions_for(member)
    return permissions.view_channel and permissions.read_message_history


def public_history_visible_to_all(channel: discord.TextChannel) -> bool:
    """Sufficient proof that a source is public to every possible guild reader."""
    everyone = channel.guild.default_role
    overwrite = channel.overwrites_for(everyone)
    view = everyone.permissions.view_channel
    if overwrite.view_channel is not None:
        view = overwrite.view_channel
    return (
        view
        and history_visible_to_all(channel)
        and not any(value.view_channel is False for value in channel.overwrites.values())
    )


def history_visible_to_all(channel: discord.TextChannel | discord.ForumChannel) -> bool:
    """Prove history access from roles, or request a fresh membership check.

    Without history denies, adding roles cannot remove history. Each possible
    role that grants viewing must also grant history. Member view grants need
    their own history grant; uncached member overwrites are included by discord.py.
    """
    for target, overwrite in channel.overwrites.items():
        if overwrite.read_message_history is False:
            return False
        if (
            not isinstance(target, discord.Role)
            and overwrite.view_channel is True
            and overwrite.read_message_history is not True
        ):
            return False
    for role in channel.guild.roles:
        permissions = channel.permissions_for(role)
        if permissions.view_channel and not permissions.read_message_history:
            return False
    return True
