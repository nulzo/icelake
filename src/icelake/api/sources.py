"""Source revisions fence extraction and retire memories after edits or deletion."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from uuid import uuid4

from pydantic import field_validator

from icelake.models.common import FrozenModel, ensure_aware
from icelake.models.facts import FactHistoryEntry, FactHistoryKind
from icelake.ports.store import MemoryStore


class SourceRevision(FrozenModel):
    revision: str
    changed_at: datetime
    deleted: bool = False

    @field_validator("changed_at")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        return ensure_aware(value)


def source_key(message_id: str) -> str:
    return "source:" + sha256(message_id.encode()).hexdigest()


async def source_is_current(
    store: MemoryStore,
    guild_id: str,
    message_id: str,
    revision: str,
) -> bool:
    raw = await store.get_cursor(guild_id, source_key(message_id))
    if raw is None:
        return not revision
    state = SourceRevision.model_validate_json(raw)
    return not state.deleted and state.revision == revision


async def invalidate_source(
    store: MemoryStore,
    guild_id: str,
    message_id: str,
    state: SourceRevision,
) -> int:
    """Atomically fence stale observations and invalidate all affected active facts.

    Transactions serialize with ingestion's guild commit fence. Duplicate or
    out-of-order gateway events cannot resurrect deleted or superseded evidence.
    Existing source snapshots remain in history; they stop being current evidence.
    """
    count = 0
    async with store.transaction():
        await store.set_cursor(guild_id, "commit_fence", uuid4().hex)
        raw = await store.get_cursor(guild_id, source_key(message_id))
        if raw:
            previous = SourceRevision.model_validate_json(raw)
            if previous.deleted or previous.changed_at >= state.changed_at:
                return 0
        await store.set_cursor(guild_id, source_key(message_id), state.model_dump_json())
        while records := await store.facts_for_source(guild_id, message_id, limit=100):
            for record in records:
                await store.transition_fact(
                    guild_id, record.id, valid_until=state.changed_at, updated_at=state.changed_at
                )
                await store.drop_evidence_from_edges(guild_id, record.id, until=state.changed_at)
                await store.append_history(
                    guild_id,
                    record.id,
                    FactHistoryEntry(
                        at=state.changed_at,
                        kind=FactHistoryKind.INVALIDATED,
                        fact_version=record.version,
                        detail="source deleted" if state.deleted else "source edited",
                    ),
                )
                count += 1
    return count
