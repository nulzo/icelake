"""Conservative live Discord authorization for memory disclosure."""

from __future__ import annotations

import discord

from icelake.permissions import Audience


class DiscordChannelAuthorizer:
    """Use fresh roles, overwrites and membership; deny unprovable audiences.

    A channel is safe only when every potential reader has history access. If
    any overwrite denies history, this adapter abstains rather than enumerating
    an incomplete member cache. Cross-channel reads require a source readable
    with history by every guild member, with no denying overwrite.
    The core AudienceGate supplies the timeout and fails closed on API errors.
    """

    def __init__(self, client: discord.Client) -> None:
        self._client = client

    async def allowed_channels(
        self, guild_id: str, audience: Audience, source_channels: frozenset[str]
    ) -> frozenset[str]:
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
        if not isinstance(
            parent, (discord.TextChannel, discord.ForumChannel)
        ) or not history_visible_to_all(parent):
            return frozenset()
        for user_id in (int(audience.requester_id), self._client.user.id):
            member = await guild.fetch_member(user_id)
            permissions = parent.permissions_for(member)
            if not (permissions.view_channel and permissions.read_message_history):
                return frozenset()
            if isinstance(channel, discord.Thread) and channel.is_private():
                # permissions_for does not establish private-thread membership.
                await channel.fetch_member(user_id)
        allowed = source_channels & {audience.channel_id}
        if source_channels - allowed:
            # One REST request for all channel overwrites, independent of fact count.
            channels = await guild.fetch_channels()
            allowed |= frozenset(
                str(source.id)
                for source in channels
                if isinstance(source, discord.TextChannel)
                and str(source.id) in source_channels
                and public_history_visible_to_all(source)
            )
        return frozenset(allowed)


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
    """Sufficient, conservative proof that anyone who sees the channel sees history."""
    everyone = channel.guild.default_role
    overwrite = channel.overwrites_for(everyone)
    history = everyone.permissions.read_message_history
    if overwrite.read_message_history is not None:
        history = overwrite.read_message_history
    return history and not any(
        value.read_message_history is False for value in channel.overwrites.values()
    )
