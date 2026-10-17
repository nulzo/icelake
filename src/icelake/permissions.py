"""Audience authorization before memory reaches prompts or external rankers."""

from __future__ import annotations

import asyncio
from typing import Protocol

from pydantic import Field

from icelake.models.common import FrozenModel
from icelake.models.facts import FactRecord


class Audience(FrozenModel):
    """Trusted reply destination; the audience is everyone who can read it.

    Construct this from transport metadata, never model output. A requester who
    can read a private channel cannot authorize disclosure into a public reply.
    """

    channel_id: str = Field(min_length=1)
    requester_id: str = Field(min_length=1)


class ChannelAuthorizer(Protocol):
    """Resolve current source-channel visibility for the ENTIRE reply audience.

    Implementations must account for Discord overwrites, private threads,
    membership, and history permissions. Return only proven-safe source IDs.
    Failures and unknown channels must deny access; decisions are not cached.
    """

    async def allowed_channels(
        self, guild_id: str, audience: Audience, source_channels: frozenset[str]
    ) -> frozenset[str]: ...


def source_channels(record: FactRecord) -> frozenset[str]:
    """All evidence channels constrain a derived fact, including mixed sources."""
    if not record.citations or any(
        not source.channel_id or source.guild_id != record.guild_id for source in record.citations
    ):
        return frozenset()
    return frozenset(source.channel_id for source in record.citations)


class AudienceGate:
    """Default to same-channel disclosure; optionally inject a live ACL adapter."""

    def __init__(
        self, authorizer: ChannelAuthorizer | None = None, *, timeout_seconds: float = 3.0
    ) -> None:
        self._authorizer = authorizer
        self._timeout_seconds = timeout_seconds

    async def filter(
        self, guild_id: str, audience: Audience, records: tuple[FactRecord, ...]
    ) -> tuple[FactRecord, ...]:
        candidates = frozenset(channel for r in records for channel in source_channels(r))
        allowed = candidates & {audience.channel_id}
        if self._authorizer is not None:
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    allowed = candidates & await self._authorizer.allowed_channels(
                        guild_id, audience, candidates
                    )
            except Exception:
                return ()
        return tuple(
            r
            for r in records
            if r.guild_id == guild_id
            and source_channels(r)
            and source_channels(r).issubset(allowed)
        )
