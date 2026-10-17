from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Protocol

from icelake.models.facts import FactRecord, ProfileSummary, SummarySource


class SummaryStore(Protocol):
    """Source-of-truth reads required to validate a derived digest."""

    async def get_facts(
        self, guild_id: str, fact_ids: tuple[str, ...]
    ) -> tuple[FactRecord, ...]: ...

    async def get_opt_out(self, guild_id: str, user_id: str) -> bool: ...


def summary_source(record: FactRecord) -> SummarySource:
    """Bind a digest to the exact stored source, including its validity and provenance."""
    return SummarySource(
        fact_id=record.id,
        fingerprint=hashlib.sha256(
            record.model_dump_json(
                include={
                    "id",
                    "guild_id",
                    "subject_id",
                    "text",
                    "category",
                    "scope",
                    "attribution",
                    "valid_from",
                    "valid_until",
                    "supersedes_id",
                    "superseded_by_id",
                    "citations",
                    "related_user_ids",
                    "entity_slugs",
                    "tags",
                    "expires_at",
                    "governance",
                }
            ).encode()
        ).hexdigest(),
    )


async def summary_is_valid(
    store: SummaryStore, summary: ProfileSummary, *, now: datetime | None = None
) -> bool:
    """Reject ungrounded, changed, expired, deleted or consent-blocked digest sources."""
    if summary.schema_version != 2 or not summary.sources:
        return False
    users = {summary.subject_id} if summary.subject_id else set()
    now = now if now is not None else datetime.now(UTC)
    now = now.replace(tzinfo=now.tzinfo or UTC).astimezone(UTC)
    records = {
        record.id: record
        for record in await store.get_facts(
            summary.guild_id, tuple(source.fact_id for source in summary.sources)
        )
    }
    for source in summary.sources:
        record = records.get(source.fact_id)
        if (
            record is None
            or record.guild_id != summary.guild_id
            or record.subject_id != summary.subject_id
            or not record.is_active
            or not record.is_valid_at(now)
            or summary_source(record) != source
        ):
            return False
        if (
            record.expires_at is not None
            and record.expires_at.replace(tzinfo=record.expires_at.tzinfo or UTC).astimezone(UTC)
            <= now
        ):
            return False
        users.update(record.related_user_ids)
        users.update(c.author_id for c in record.citations)
        users.update(
            uid
            for uid in (
                record.subject_id,
                record.attribution.speaker_id,
                record.attribution.actor_id,
            )
            if uid
        )
    for user_id in users:
        if await store.get_opt_out(summary.guild_id, user_id):
            return False
    return True
