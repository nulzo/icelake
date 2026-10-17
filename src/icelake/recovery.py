"""Replay rebuildable vector projections from a transactional outbox."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from icelake.models.recovery import ProjectionJob, RecoveryReport
from icelake.ports.llm import Embedder
from icelake.ports.store import MemoryStore
from icelake.ports.vectors import VectorIndex, VectorItem

logger = logging.getLogger(__name__)


async def recover_projections(
    store: MemoryStore,
    vectors: VectorIndex | None,
    embedder: Embedder | None,
    *,
    limit: int = 32,
    timeout_seconds: float = 30.0,
    retry_delay_seconds: float = 30.0,
) -> RecoveryReport:
    """Retry a bounded slice; failures retain work and never undo committed memory.

    Index adapters must reject older VectorItem revisions. Multiple projectors
    can safely replay the same job; acknowledgment compares the queued version.
    """
    if vectors is None or embedder is None:
        return RecoveryReport()
    jobs = await store.pending_projections(limit)
    completed = 0
    attempted = 0
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    for job in jobs:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        attempted += 1
        try:
            async with asyncio.timeout(remaining):
                if await project_fact(store, vectors, embedder, job):
                    completed += 1
        except Exception:
            await store.defer_projection(
                job, datetime.now(UTC) + timedelta(seconds=retry_delay_seconds)
            )
            logger.warning(
                "projection retained guild=%s fact=%s", job.guild_id, job.fact_id, exc_info=True
            )
    return RecoveryReport(attempted=attempted, completed=completed, failed=attempted - completed)


async def project_fact(
    store: MemoryStore,
    vectors: VectorIndex,
    embedder: Embedder,
    job: ProjectionJob,
) -> bool:
    """Rebuild from current authoritative content, then acknowledge this revision."""
    record = await store.get_fact(job.guild_id, job.fact_id)
    if record is None:
        await vectors.delete((job.fact_id,))
    else:
        if job.embedding is not None and job.version == record.version:
            embedding = job.embedding
        else:
            (embedding,) = await embedder.embed((record.text,))
        # Recheck erasure after an external call, before projecting content.
        current = await store.get_fact(job.guild_id, job.fact_id)
        if current is None or current.version != record.version:
            return False
        await vectors.upsert(
            (
                VectorItem(
                    id=record.id,
                    guild_id=record.guild_id,
                    subject_id=record.subject_id,
                    embedding=embedding,
                    version=record.version,
                ),
            )
        )
    await store.complete_projection(job)
    return True
