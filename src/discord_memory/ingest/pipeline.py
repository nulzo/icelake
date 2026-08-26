"""Ingestion pipeline: claims leased batches, extracts, reconciles, commits (§4).

Guarantees: claim is atomic; failures dead-letter instead of silently acking (B4);
leases expire so crashed workers never strand messages (B2); the extraction LLM only
ever sees minted roster tokens (§3.1).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from discord_memory.config import MemoryConfig
from discord_memory.identity.guards import SubjectGate
from discord_memory.ingest.context_builder import build_extraction_context
from discord_memory.ingest.executor import CommitSummary, FactCommitter
from discord_memory.ingest.extraction import FactExtractor
from discord_memory.ingest.gates import batch_worth_extracting, normalize_text
from discord_memory.ingest.reconcile import Reconciler
from discord_memory.ingest.roster import Roster
from discord_memory.models.admin import BudgetStep
from discord_memory.models.events import BatchCompleted, ExtractionFailed
from discord_memory.models.facts import FactRecord, SourceRef, SourceRole
from discord_memory.models.operations import ProposedFact, ReconcileKind
from discord_memory.ports.clock import Clock, IdGen
from discord_memory.ports.llm import ChatLLM, Embedder, Meter
from discord_memory.ports.queue import BatchKey, IngestQueue, StoredMessage
from discord_memory.ports.store import MemoryStore
from discord_memory.ports.vectors import VectorIndex

if TYPE_CHECKING:
    from discord_memory.api.events import EventBus

logger = logging.getLogger(__name__)

SERVER_SUBJECT_KEY = "__server__"


def _build_source_refs(messages: tuple[StoredMessage, ...]) -> tuple[SourceRef, ...]:
    """Snapshot every claimed message into a frozen citation ref."""
    return tuple(
        SourceRef(
            message_id=message.message_id,
            channel_id=message.channel_id,
            guild_id=message.guild_id,
            author_id=message.author_id,
            author_name=message.author_display_name or message.author_username,
            content_snippet=message.content[:280],
            created_at=message.created_at,
            role=SourceRole.SUPPORTING,
        )
        for message in messages
    )


def _pick_citations(indexes: tuple[int, ...], refs: tuple[SourceRef, ...]) -> tuple[SourceRef, ...]:
    """Cite the messages the model referenced; mark the last one primary."""
    if not refs:
        return ()
    picked: list[SourceRef] = []
    for index in indexes or (1,):
        if 1 <= index <= len(refs):
            picked.append(refs[index - 1])
    if not picked:
        picked = [refs[0]]
    primary = picked[-1].model_copy(update={"role": SourceRole.PRIMARY})
    return (*picked[:-1], primary)


@dataclass(slots=True)
class BatchReport:
    key: BatchKey
    messages_processed: int = 0
    summary: CommitSummary | None = None
    skipped_reason: str | None = None


class IngestPipeline:
    """Worker-facing orchestration over one claimed batch at a time."""

    def __init__(
        self,
        *,
        config: MemoryConfig,
        clock: Clock,
        id_gen: IdGen,
        queue: IngestQueue,
        store: MemoryStore,
        vectors: VectorIndex | None,
        embedder: Embedder | None,
        llm: ChatLLM | None,
        meter: Meter,
        subject_gate: SubjectGate,
    ) -> None:
        self._config = config
        self._clock = clock
        self._queue = queue
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self._llm = llm
        self._meter = meter
        self._subject_gate = subject_gate
        self._extractor = FactExtractor(llm, config.extraction)
        self._reconciler = Reconciler(store, vectors, llm, embedder, config.extraction)
        self._committer = FactCommitter(
            store=store,
            vectors=vectors,
            embedder=embedder,
            clock=clock,
            id_gen=id_gen,
            config=config,
        )
        self.owner = f"pipeline-{id(self):x}"
        self.event_bus: EventBus | None = None
        self.consolidation: object | None = None

    def attach_event_bus(self, bus: EventBus) -> None:
        """Wire an event bus for hook dispatch (called by the facade)."""
        self.event_bus = bus

    def attach_consolidation(self, consolidation: object) -> None:
        """Wire profile-summary regeneration (called by the facade)."""
        self.consolidation = consolidation

    async def run_pending(self, *, limit_batches: int = 8) -> int:
        """Process all due batches; returns how many were processed."""
        now = self._clock.now()
        keys = await self._queue.due_batch_keys(
            now=now,
            batch_size=self._config.batching.batch_size_messages,
            max_age_seconds=self._config.batching.max_age_seconds,
            limit=limit_batches * 4,
        )
        processed = 0
        for key in keys[:limit_batches]:
            report = await self.process_key(key)
            if report and report.summary is not None:
                processed += 1
        return processed

    async def flush_subject(self, guild_id: str, subject_key: str) -> BatchReport:
        key = BatchKey(guild_id=guild_id, subject_key=subject_key)
        if subject_key != SERVER_SUBJECT_KEY:
            return await self.process_key(key)
        return await self._process_server_window(key)

    async def process_key(self, key: BatchKey) -> BatchReport:
        limit = (
            self._config.batching.server_scope_window
            if key.subject_key == SERVER_SUBJECT_KEY
            else self._config.batching.batch_size_messages
        )
        claim = await self._queue.claim_batch(
            key,
            now=self._clock.now(),
            lease_seconds=self._config.batching.lease_seconds,
            owner=self.owner,
            limit=limit,
        )
        if claim.locked_by_other or not claim.messages:
            reason = "locked" if claim.locked_by_other else "empty"
            if self.event_bus is not None and reason == "locked":
                self.event_bus.publish(
                    BatchCompleted(
                        guild_id=key.guild_id, subject_key=key.subject_key, skipped_reason=reason
                    ),
                )
            return BatchReport(key=key, skipped_reason=reason)
        message_ids = tuple(m.message_id for m in claim.messages)

        async def _heartbeat() -> None:
            interval = max(5.0, self._config.batching.lease_seconds / 3)
            while True:
                await asyncio.sleep(interval)
                await self._queue.renew_lease(
                    key,
                    owner=self.owner,
                    now=self._clock.now(),
                    lease_seconds=self._config.batching.lease_seconds,
                )

        heartbeat = asyncio.create_task(_heartbeat())
        try:
            report = await self._process_claimed(key, claim.messages)
            await self._queue.complete_messages(message_ids, owner=self.owner)
            return report
        except Exception as exc:
            logger.exception("Batch %s failed permanently", key.as_tuple)
            await self._queue.dead_letter_messages(message_ids, owner=self.owner)
            if self.event_bus is not None:
                self.event_bus.publish(
                    ExtractionFailed(
                        guild_id=key.guild_id,
                        subject_key=key.subject_key,
                        attempt=1,
                        error_kind=type(exc).__name__,
                    )
                )
            return BatchReport(key=key, skipped_reason=f"error:{type(exc).__name__}")
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

    async def _process_server_window(self, key: BatchKey) -> BatchReport:
        """Community-scope extraction over the recent guild window.

        The ``__server__`` key lease is won first (same CAS as user batches), so
        concurrent workers cannot double-extract the community window.
        """
        claim = await self._queue.claim_batch(
            key,
            now=self._clock.now(),
            lease_seconds=self._config.batching.lease_seconds,
            owner=self.owner,
            limit=self._config.batching.server_scope_window,
        )
        if claim.locked_by_other:
            return BatchReport(key=key, skipped_reason="locked")
        window: tuple[StoredMessage, ...] = claim.messages or await self._queue.recent_messages(
            key.guild_id,
            self._config.batching.server_scope_window,
        )
        if not window:
            return BatchReport(key=key, skipped_reason="empty")
        summary = CommitSummary()
        try:
            report = await self._process_claimed(key, window)
            if report.summary is not None:
                summary = report.summary
        except Exception:
            logger.exception("Server window %s failed", key.as_tuple)
            return BatchReport(key=key, skipped_reason="error")
        # Window messages were already acked by their own user batches; this
        # pass is read-only and dedup gates keep repeated windows idempotent.
        return BatchReport(
            key=key,
            messages_processed=len(window),
            summary=summary,
        )

    async def _process_claimed(
        self,
        key: BatchKey,
        messages: tuple[StoredMessage, ...],
    ) -> BatchReport:
        guild_id = key.guild_id
        is_server_scope = key.subject_key == SERVER_SUBJECT_KEY
        subject_key_author = None if is_server_scope else key.subject_key

        texts = tuple((m.author_username or m.author_id, m.content) for m in messages)
        joined_text = " ".join(content for _, content in texts)

        if self._config.extraction.noise_gate and not batch_worth_extracting(
            tuple(m.content for m in messages),
        ):
            return BatchReport(key=key, messages_processed=len(messages), skipped_reason="noise")

        step = self._meter.check_budget(guild_id)
        if step is BudgetStep.SKIP_EXTRACTION:
            return BatchReport(key=key, messages_processed=len(messages), skipped_reason="budget")

        roster = self._build_roster(messages, author_id=subject_key_author)
        batch_embedding = None
        if self._embedder is not None and normalize_text(joined_text):
            (batch_embedding,) = await self._embedder.embed((joined_text,))
        existing_block = await build_extraction_context(
            store=self._store,
            vectors=self._vectors,
            guild_id=guild_id,
            subject_id=None if is_server_scope else key.subject_key,
            batch_text=joined_text,
            batch_embedding=batch_embedding,
        )

        result = await self._extractor.extract(
            roster=roster,
            messages=texts,
            existing_memories_block=existing_block,
        )
        summary = CommitSummary()
        for text, reason in result.rejected:
            logger.debug("Rejected candidate (%s): %s", reason, text)

        candidates = [
            (vetted.proposal, vetted.subject_id, vetted.speaker_id) for vetted in result.vetted
        ]
        mentioned_ids = {mention_id for message in messages for mention_id in message.mention_ids}
        source_refs = _build_source_refs(messages)
        if candidates:
            await self._commit_candidates(
                guild_id=guild_id,
                key=key,
                candidates=candidates,
                roster=roster,
                embeddings_by_text={},
                batch_embedding=batch_embedding,
                summary=summary,
                mentioned_ids=tuple(mentioned_ids),
                source_refs=source_refs,
            )

        completed = BatchCompleted(
            guild_id=guild_id,
            subject_key=key.subject_key,
            adds=summary.adds,
            reinforces=summary.reinforces,
            supersessions=summary.supersessions,
            invalidations=summary.invalidations,
        )
        if self.event_bus is not None:
            self.event_bus.publish(completed)
        await self._maybe_refresh_summary(key, summary.adds)
        return BatchReport(key=key, messages_processed=len(messages), summary=summary)

    async def _maybe_refresh_summary(
        self,
        key: BatchKey,
        adds: int,
    ) -> None:
        """Regenerate the profile digest after enough new facts (PLAN.md Part 7)."""
        threshold = self._config.extraction.auto_consolidate_after_adds
        if (
            adds < threshold
            or threshold <= 0
            or key.subject_key == SERVER_SUBJECT_KEY
            or self.consolidation is None
        ):
            return
        try:
            await self.consolidation.regenerate_profile(  # type: ignore[attr-defined]
                guild_id=key.guild_id,
                subject_id=key.subject_key,
            )
        except Exception:
            logger.exception("summary refresh failed for %s", key.as_tuple)

    async def _commit_candidates(
        self,
        *,
        guild_id: str,
        key: BatchKey,
        candidates: list[tuple[ProposedFact, str | None, str | None]],
        roster: Roster,
        embeddings_by_text: dict[str, tuple[float, ...]],
        batch_embedding: tuple[float, ...] | None,
        summary: CommitSummary,
        mentioned_ids: tuple[str, ...] = (),
        source_refs: tuple[SourceRef, ...] = (),
    ) -> None:
        batch_subject = None if key.subject_key == SERVER_SUBJECT_KEY else key.subject_key
        plan = await self._reconciler.build_plan(
            candidates,
            guild_id=guild_id,
            batch_subject_id=batch_subject,
            embeddings_by_text=embeddings_by_text,
        )
        decisions_map = await self._reconciler.resolve_collisions(plan.collisions)

        for proposal, subject_id, speaker_id in plan.direct_adds:
            record = await self._committer.commit_add(
                proposal=proposal,
                subject_id=subject_id,
                speaker_id=speaker_id,
                guild_id=guild_id,
                roster=roster,
                mentioned_ids=mentioned_ids,
                source_refs=_pick_citations(proposal.source_message_indexes, source_refs),
            )
            summary.adds += 1
            self._publish_fact(record, reinforced=False)

        for index, collision in enumerate(plan.collisions):
            decisions = decisions_map.get(index, ())
            handled = False
            for decision in decisions:
                if decision.kind is ReconcileKind.NOOP:
                    # "same meaning" => strengthen the existing fact.
                    target = next(
                        (
                            n
                            for n in collision.neighbors
                            if n.id == decision.target_id or len(collision.neighbors) == 1
                        ),
                        None,
                    )
                    if target is not None:
                        await self._committer.commit_reinforce(
                            target,
                            collision.candidate,
                        )
                    summary.reinforces += 1
                    handled = True
                    break
                target = next(
                    (n for n in collision.neighbors if n.id == decision.target_id),
                    None,
                )
                if target is None:
                    continue
                if decision.kind is ReconcileKind.UPDATE:
                    await self._committer.commit_supersede(
                        old_record=target,
                        proposal=collision.candidate,
                        subject_id=collision.subject_id,
                        speaker_id=collision.speaker_id,
                        reason=decision.reason,
                        guild_id=guild_id,
                        roster=roster,
                    )
                    summary.supersessions += 1
                    handled = True
                    break
                if decision.kind is ReconcileKind.INVALIDATE:
                    await self._committer.commit_invalidate(
                        old_record=target,
                        reason=decision.reason,
                    )
                    summary.invalidations += 1
                    break
            if not handled:
                # Conservative default on unresolved collisions: reinforce the
                # strongest neighbor (exact dup first, then top semantic match)
                # instead of adding a near-duplicate.
                neighbor = (
                    collision.duplicates[0]
                    if collision.duplicates
                    else (collision.semantic_neighbors[0] if collision.semantic_neighbors else None)
                )
                if neighbor is not None:
                    await self._committer.commit_reinforce(neighbor, collision.candidate)
                    summary.reinforces += 1
                else:
                    record = await self._committer.commit_add(
                        proposal=collision.candidate,
                        subject_id=collision.subject_id,
                        speaker_id=collision.speaker_id,
                        guild_id=guild_id,
                        roster=roster,
                        mentioned_ids=mentioned_ids,
                        source_refs=_pick_citations(
                            collision.candidate.source_message_indexes,
                            source_refs,
                        ),
                    )
                    summary.adds += 1
                    self._publish_fact(record, reinforced=False)

    def _publish_fact(self, record: FactRecord, *, reinforced: bool) -> None:
        if self.event_bus is not None:
            from discord_memory.models.events import FactCommitted

            self.event_bus.publish(
                FactCommitted(
                    guild_id=record.guild_id,
                    fact_id=record.id,
                    subject_id=record.subject_id,
                    text=record.text,
                    was_reinforcement=reinforced,
                )
            )

    def _build_roster(
        self,
        messages: tuple[StoredMessage, ...],
        *,
        author_id: str | None,
    ) -> Roster:
        roster = Roster()
        seen: set[str] = set()

        def add(user_id: str | None, display_name: str) -> None:
            if user_id and user_id not in seen:
                seen.add(user_id)
                roster.add(user_id, display_name)

        if author_id:
            first = messages[0]
            add(author_id, first.author_display_name or first.author_username or author_id)
        for message in messages:
            add(message.author_id, message.author_display_name or message.author_username)
            for mention_id in message.mention_ids:
                add(mention_id, mention_id)
        return roster
