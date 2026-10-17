"""Ingestion pipeline: claims leased batches, extracts, reconciles, commits (§4).

Guarantees: claim is atomic; failures dead-letter instead of silently acking (B4);
leases expire so crashed workers never strand messages (B2);

Cohesion exception: orchestration stays here rather than scattering the commit
branches across modules. The commit function is intentionally one auditable unit
covering plan validation, all mutations, receipts and post-commit notifications;
recovery, retries, extraction and reconciliation remain separate concerns.

The extraction LLM only ever sees minted roster tokens (§3.1).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from typing import TYPE_CHECKING
from uuid import uuid4

from icelake.config import MemoryConfig
from icelake.identity.aliases import (
    extract_self_name_aliases,
    extract_stated_name_aliases,
    is_third_party_name_reference,
    is_valid_alias,
    normalize_alias,
    weight_for_source,
)
from icelake.identity.guards import SubjectGate
from icelake.ingest.context_builder import build_extraction_context
from icelake.ingest.executor import CommitSummary, FactCommitter
from icelake.ingest.extraction import ExtractionResult, FactExtractor
from icelake.ingest.gates import batch_worth_extracting, normalize_text
from icelake.ingest.reconcile import Reconciler
from icelake.ingest.retries import RetryState, after_failure, read_retry, save_retry
from icelake.ingest.roster import Roster
from icelake.models.admin import BudgetStep
from icelake.models.events import BatchCompleted, ExtractionFailed
from icelake.models.facts import FactRecord, SourceRef, SourceRole
from icelake.models.identity import AliasSource
from icelake.models.operations import ProposedFact, ReconcileDecision, ReconcileKind
from icelake.ports.clock import Clock, IdGen
from icelake.ports.llm import ChatLLM, Embedder, Meter
from icelake.ports.queue import BatchKey, IngestQueue, StoredMessage
from icelake.ports.store import MemoryStore
from icelake.ports.vectors import VectorIndex
from icelake.recovery import recover_projections

if TYPE_CHECKING:
    from icelake.api.events import EventBus
    from icelake.consolidation.service import ConsolidationService

logger = logging.getLogger(__name__)

SERVER_SUBJECT_KEY = "__server__"
_SNIPPET_CHARS = 160


def _receipt_key(key: BatchKey, message_id: str) -> str:
    scope = "server" if key.subject_key == SERVER_SUBJECT_KEY else "user"
    return "ingest:" + scope + ":" + sha256(message_id.encode()).hexdigest()


def _snippet(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= _SNIPPET_CHARS:
        return collapsed
    return collapsed[: _SNIPPET_CHARS - 1] + "…"


def _extract_skip_reason(
    result: ExtractionResult,
    candidates: list[tuple[ProposedFact, str | None, str | None]],
    summary: CommitSummary,
) -> str | None:
    """Classify a finished extraction that wrote nothing to the store."""
    if summary.adds or summary.reinforces or summary.supersessions or summary.invalidations:
        return None
    if not result.vetted and not result.rejected:
        return "empty_ops"
    if not candidates:
        return "gated" if result.rejected and not result.vetted else "filtered"
    return None


def _window_sort_key(message: StoredMessage) -> str:
    """Monotonic window ordering key: ISO timestamp + id tiebreak."""
    return f"{message.created_at.isoformat()}|{message.message_id}"


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
    for index in dict.fromkeys(indexes):
        if 1 <= index <= len(refs):
            picked.append(refs[index - 1])
    if not picked:
        return ()
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
        reconcile_llm: ChatLLM | None = None,
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
        self._extractor = FactExtractor(llm, config.extraction, max_tokens=config.llm.max_tokens)
        self._reconciler = Reconciler(
            store, vectors, reconcile_llm or llm, embedder, config.extraction, clock=clock
        )
        self._committer = FactCommitter(
            store=store,
            vectors=vectors,
            embedder=embedder,
            clock=clock,
            id_gen=id_gen,
            config=config,
            bot_guard=subject_gate._guard if subject_gate else None,
        )
        self.owner = f"pipeline-{uuid4().hex}"
        self.event_bus: EventBus | None = None
        self.consolidation: ConsolidationService | None = None

    def attach_event_bus(self, bus: EventBus) -> None:
        """Wire an event bus for hook dispatch (called by the facade)."""
        self.event_bus = bus

    def attach_consolidation(self, consolidation: ConsolidationService) -> None:
        """Wire profile-summary regeneration (called by the facade)."""
        self.consolidation = consolidation

    def _skip_batch(
        self,
        key: BatchKey,
        *,
        reason: str,
        messages: tuple[StoredMessage, ...] = (),
        detail: str = "",
    ) -> BatchReport:
        """Log a skip, optionally emit BatchCompleted, and return the report."""
        extra = f" {detail}" if detail else ""
        logger.debug(
            "batch skipped guild=%s subject=%s reason=%s messages=%d%s snippet=%r",
            key.guild_id,
            key.subject_key,
            reason,
            len(messages),
            extra,
            _snippet(" ".join(m.content for m in messages)),
        )
        if self.event_bus is not None:
            self.event_bus.publish(
                BatchCompleted(
                    guild_id=key.guild_id,
                    subject_key=key.subject_key,
                    skipped_reason=reason,
                )
            )
        return BatchReport(
            key=key,
            messages_processed=len(messages),
            skipped_reason=reason,
        )

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
        retry = await read_retry(self._store, key)
        if retry.attempts >= self._config.recovery.max_attempts:
            retry = RetryState()  # New pending input after a terminal batch starts a fresh budget.
        if retry.next_attempt_at is not None and retry.next_attempt_at > self._clock.now():
            return self._skip_batch(key, reason="retry_backoff")
        limit = (
            self._config.batching.server_scope_window
            if key.subject_key == SERVER_SUBJECT_KEY
            else self._config.batching.batch_size_messages
        )
        owner = f"{self.owner}:{uuid4().hex}"
        claim = await self._queue.claim_batch(
            key,
            now=self._clock.now(),
            lease_seconds=self._config.batching.lease_seconds,
            owner=owner,
            limit=limit,
        )
        if claim.locked_by_other or not claim.messages:
            if not claim.locked_by_other:
                await self._queue.release_key(key, owner=owner)
            reason = "locked" if claim.locked_by_other else "empty"
            return self._skip_batch(key, reason=reason, messages=claim.messages)
        message_ids = tuple(m.message_id for m in claim.messages)

        heartbeat = asyncio.create_task(self._heartbeat(key, owner, asyncio.current_task()))
        try:
            report = await self._process_claimed(key, claim.messages, lease_owner=owner)
            if report.skipped_reason != "budget":
                await self._queue.complete_messages(message_ids, owner=owner)
            await save_retry(self._store, key, RetryState())
            await recover_projections(
                self._store,
                self._vectors,
                self._embedder,
                limit=self._config.recovery.projection_batch_size,
                timeout_seconds=self._config.recovery.projection_timeout_seconds,
                retry_delay_seconds=self._config.recovery.retry_initial_seconds,
            )
            return report
        except Exception as exc:
            logger.exception("Batch %s retained after failure", key.as_tuple)
            settings = self._config.recovery
            retry = after_failure(
                retry,
                now=self._clock.now(),
                error=exc,
                initial_seconds=settings.retry_initial_seconds,
                max_seconds=settings.retry_max_seconds,
            )
            await save_retry(self._store, key, retry)
            if retry.attempts >= settings.max_attempts:
                await self._queue.dead_letter_messages(message_ids, owner=owner)
            if self.event_bus is not None:
                self.event_bus.publish(
                    ExtractionFailed(
                        guild_id=key.guild_id,
                        subject_key=key.subject_key,
                        attempt=retry.attempts,
                        error_kind=type(exc).__name__,
                    )
                )
            return BatchReport(key=key, skipped_reason=f"error:{type(exc).__name__}")
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            # The lease guards in-flight work only; lingering after completion
            # would stall other processes/owners for the remaining lease window.
            # Empty claims skip it: the lease belongs to the in-flight batch.
            if message_ids:
                await self._queue.release_key(key, owner=owner)

    async def _heartbeat(
        self, key: BatchKey, owner: str, processing_task: asyncio.Task[object] | None
    ) -> None:
        try:
            while True:
                await asyncio.sleep(max(5.0, self._config.batching.lease_seconds / 3))
                if not await self._queue.renew_lease(
                    key,
                    owner=owner,
                    now=self._clock.now(),
                    lease_seconds=self._config.batching.lease_seconds,
                ):
                    break
        except Exception:
            logger.exception("ingest lease renewal failed")
        if processing_task is not None:
            processing_task.cancel("ingest lease lost")

    async def _process_server_window(self, key: BatchKey) -> BatchReport:
        """Community-scope extraction over the recent guild window.

        The ``__server__`` key lease is won first (same CAS as user batches), so
        concurrent workers cannot double-extract the community window.
        """
        retry = await read_retry(self._store, key)
        if retry.attempts >= self._config.recovery.max_attempts:
            return self._skip_batch(key, reason="server_retry_exhausted")
        if retry.next_attempt_at is not None and retry.next_attempt_at > self._clock.now():
            return self._skip_batch(key, reason="retry_backoff")
        owner = f"{self.owner}:{uuid4().hex}"
        claim = await self._queue.claim_batch(
            key,
            now=self._clock.now(),
            lease_seconds=self._config.batching.lease_seconds,
            owner=owner,
            limit=self._config.batching.server_scope_window,
        )
        if claim.locked_by_other:
            return self._skip_batch(key, reason="locked")
        heartbeat = asyncio.create_task(self._heartbeat(key, owner, asyncio.current_task()))
        try:
            window: tuple[StoredMessage, ...] = claim.messages or await self._queue.recent_messages(
                key.guild_id,
                self._config.batching.server_scope_window,
            )

            # Watermark: only messages NEWER than the last community pass. Without
            # this, every heartbeat re-extracted the same window and the model's
            # reworded paraphrases leaked in as duplicate facts.
            watermark = await self._store.get_cursor(key.guild_id, "server_window")
            if watermark is not None:
                window = tuple(
                    message for message in window if _window_sort_key(message) > watermark
                )
            if not window:
                return self._skip_batch(key, reason="empty")
            if len(window) < self._config.batching.batch_size_messages:
                # Community roll-up waits for batch-sized volume; per-message
                # culture extraction doubles cost and races the user batches.
                need = self._config.batching.batch_size_messages
                return self._skip_batch(
                    key,
                    reason="below_min_volume",
                    messages=window,
                    detail=(
                        f"have={len(window)} need={need} before community extraction; no LLM call"
                    ),
                )

            summary = CommitSummary()
            try:
                report = await self._process_claimed(key, window, lease_owner=owner)
                if report.skipped_reason == "budget":
                    return report
                if report.summary is not None:
                    summary = report.summary
            except Exception as exc:
                settings = self._config.recovery
                await save_retry(
                    self._store,
                    key,
                    after_failure(
                        retry,
                        now=self._clock.now(),
                        error=exc,
                        initial_seconds=settings.retry_initial_seconds,
                        max_seconds=settings.retry_max_seconds,
                    ),
                )
                logger.exception("Server window %s failed", key.as_tuple)
                return self._skip_batch(key, reason="error", messages=window)
            # Advance the watermark to the newest processed message so the next
            # pass only ever sees genuinely new activity.
            newest = max(window, key=lambda m: (m.created_at, m.message_id))
            await self._store.set_cursor(
                key.guild_id,
                "server_window",
                f"{newest.created_at.isoformat()}|{newest.message_id}",
            )
            await save_retry(self._store, key, RetryState())
            return BatchReport(
                key=key,
                messages_processed=len(window),
                summary=summary,
            )
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            await self._queue.release_key(key, owner=owner)

    async def _process_claimed(
        self,
        key: BatchKey,
        messages: tuple[StoredMessage, ...],
        *,
        lease_owner: str | None = None,
    ) -> BatchReport:
        reports = []
        for channel in dict.fromkeys(message.channel_id for message in messages):
            subset = tuple(message for message in messages if message.channel_id == channel)
            reports.append(await self._process_channel(key, subset, lease_owner=lease_owner))
        summary = CommitSummary()
        for report in reports:
            if report.summary is not None:
                summary.adds += report.summary.adds
                summary.reinforces += report.summary.reinforces
                summary.supersessions += report.summary.supersessions
                summary.invalidations += report.summary.invalidations
        return BatchReport(
            key=key,
            messages_processed=sum(r.messages_processed for r in reports),
            summary=summary if any(r.summary is not None for r in reports) else None,
            skipped_reason=next(
                (r.skipped_reason for r in reports if r.skipped_reason == "budget"),
                reports[-1].skipped_reason if reports else "empty",
            ),
        )

    async def _process_channel(
        self,
        key: BatchKey,
        messages: tuple[StoredMessage, ...],
        *,
        lease_owner: str | None = None,
    ) -> BatchReport:
        messages = tuple(
            [
                message
                for message in messages
                if await self._store.get_cursor(key.guild_id, _receipt_key(key, message.message_id))
                is None
            ]
        )
        if not messages:
            return self._skip_batch(key, reason="already_committed")
        guild_id = key.guild_id
        is_server_scope = key.subject_key == SERVER_SUBJECT_KEY
        subject_key_author = None if is_server_scope else key.subject_key

        snapshot = await self._governance_snapshot(key, messages, [])
        if await self._consent_revoked(key, messages, [], snapshot):
            return self._skip_batch(key, reason="consent_revoked", messages=messages)
        if not self._config.access.require_audience:
            await self._mine_message_aliases(guild_id, messages)

        # store_raw_messages=False stores a content hash for audit/dedup. The
        # hash is not message text — sending it to the LLM would be a privacy
        # violation and garbage extraction, so skip the batch entirely.
        if not self._config.privacy.store_raw_messages:
            return self._skip_batch(key, reason="privacy_no_raw", messages=messages)

        texts = tuple((m.author_username or m.author_id, m.content) for m in messages)
        joined_text = " ".join(content for _, content in texts)

        if self._config.extraction.noise_gate and not batch_worth_extracting(
            tuple(m.content for m in messages),
        ):
            return self._skip_batch(key, reason="noise", messages=messages)

        step = await self._meter.check_budget(guild_id)
        if step is BudgetStep.SKIP_EXTRACTION:
            return self._skip_batch(key, reason="budget", messages=messages)

        roster = await self._build_roster(messages, author_id=subject_key_author)
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
            now=self._clock.now(),
            source_channel=messages[0].channel_id,
        )

        result = await self._extractor.extract(
            roster=roster,
            messages=texts,
            existing_memories_block=existing_block,
            guild_id=guild_id,
        )
        summary = CommitSummary()
        for text, reason in result.rejected:
            logger.debug(
                "candidate rejected guild=%s subject=%s reason=%s text=%r",
                guild_id,
                key.subject_key,
                reason,
                _snippet(text),
            )

        # In-batch candidate dedup: identical/near-identical proposals within
        # one response collapse into the first occurrence.
        seen_norms: set[tuple[str | None, str]] = set()
        candidates = []
        for vetted in result.vetted:
            authors = tuple(
                dict.fromkeys(
                    messages[index - 1].author_id
                    for index in vetted.proposal.source_message_indexes
                )
            )
            if vetted.speaker_id is not None and vetted.speaker_id not in authors:
                result.rejected.append((vetted.proposal.text, "speaker_not_in_sources"))
                continue
            if vetted.speaker_id is None and authors:
                primary_index = tuple(dict.fromkeys(vetted.proposal.source_message_indexes))[-1]
                vetted.speaker_id = messages[primary_index - 1].author_id
            bound = vetted.proposal.model_copy(
                update={"text": roster.bind_names(vetted.proposal.text)},
            )
            key_tuple = (vetted.subject_id, normalize_text(bound.text))
            if key_tuple in seen_norms:
                logger.debug(
                    "in-batch duplicate dropped guild=%s subject=%s text=%r",
                    guild_id,
                    vetted.subject_id,
                    _snippet(bound.text),
                )
                continue
            seen_norms.add(key_tuple)
            candidates.append((bound, vetted.subject_id, vetted.speaker_id))
        if await self._consent_revoked(key, messages, candidates, snapshot):
            return self._skip_batch(
                key,
                reason="consent_revoked",
                messages=messages,
                detail="opt-out or purge during extraction; LLM output discarded",
            )
        if is_server_scope:
            # The community pass owns guild-scope facts only. User-anchored
            # facts belong to per-user batches — otherwise both passes extract
            # the same messages concurrently and race duplicates into the store.
            before_filter = len(candidates)
            candidates = [c for c in candidates if c[1] is None]
            dropped = before_filter - len(candidates)
            if dropped:
                logger.debug(
                    "server-scope filtered user-anchored candidates guild=%s dropped=%d",
                    guild_id,
                    dropped,
                )
        mentioned_ids = self._subject_gate.exclude_bots(
            mention_id for message in messages for mention_id in message.mention_ids
        )
        source_refs = _build_source_refs(messages)
        if candidates:
            if not self._config.access.require_audience:
                await self._mine_fact_aliases(guild_id, candidates)
            await self._commit_candidates(
                guild_id=guild_id,
                key=key,
                candidates=candidates,
                roster=roster,
                embeddings_by_text={},
                batch_embedding=batch_embedding,
                summary=summary,
                mentioned_ids=mentioned_ids,
                source_refs=source_refs,
                skip_reconcile=step is BudgetStep.SKIP_RECONCILE,
                governance=snapshot,
                lease_owner=lease_owner,
            )

        else:
            async with self._store.transaction():
                for message in messages:
                    await self._store.set_cursor(
                        guild_id,
                        _receipt_key(key, message.message_id),
                        self._clock.now().isoformat(),
                    )

        skip_reason = _extract_skip_reason(result, candidates, summary)
        if skip_reason is not None:
            logger.debug(
                "batch skipped guild=%s subject=%s reason=%s messages=%d snippet=%r",
                guild_id,
                key.subject_key,
                skip_reason,
                len(messages),
                _snippet(joined_text),
            )
        completed = BatchCompleted(
            guild_id=guild_id,
            subject_key=key.subject_key,
            adds=summary.adds,
            reinforces=summary.reinforces,
            supersessions=summary.supersessions,
            invalidations=summary.invalidations,
            skipped_reason=skip_reason,
        )
        if self.event_bus is not None:
            self.event_bus.publish(completed)
        await self._maybe_refresh_summary(key, summary.adds)
        return BatchReport(
            key=key,
            messages_processed=len(messages),
            summary=summary,
            skipped_reason=skip_reason,
        )

    async def _governance_snapshot(
        self,
        key: BatchKey,
        messages: tuple[StoredMessage, ...],
        candidates: list[tuple[ProposedFact, str | None, str | None]],
    ) -> dict[str, int]:
        """Per-participant erasure generations captured BEFORE any mutation.

        The snapshot is the fence: facts written from this batch carry it, so a
        purge by any participant afterwards invalidates them on every read.
        Generations come from the enqueue-time message stamps (not fresh reads)
        so work already pending before a purge stays fenced after it.
        """
        users: set[str] = set()
        if key.subject_key != SERVER_SUBJECT_KEY:
            users.add(key.subject_key)
        users.update(message.author_id for message in messages)
        users.update(mention_id for message in messages for mention_id in message.mention_ids)
        for _proposal, subject_id, speaker_id in candidates:
            users.update(uid for uid in (subject_id, speaker_id) if uid)
        return {
            user_id: min(
                (
                    message.governance.get(user_id, 0)
                    for message in messages
                    if user_id == message.author_id or user_id in message.mention_ids
                ),
                default=0,
            )
            for user_id in users
        }

    async def _consent_revoked(
        self,
        key: BatchKey,
        messages: tuple[StoredMessage, ...],
        candidates: list[tuple[ProposedFact, str | None, str | None]],
        snapshot: dict[str, int],
    ) -> bool:
        """Post-LLM recheck: consent or purge landed since the snapshot.

        The claim lease cannot fence this — the user may have purged between
        claim and commit. Compares CURRENT generations against the snapshot so
        LLM output predating a purge is discarded, not stored.
        """
        guild_id = key.guild_id
        if (
            key.subject_key != SERVER_SUBJECT_KEY
            and await self._subject_gate.allows(guild_id, key.subject_key) is False
        ):
            return True
        for message in messages:
            if await self._subject_gate.allows(guild_id, message.author_id) is False:
                return True
        for _proposal, subject_id, speaker_id in candidates:
            for user_id in (subject_id, speaker_id):
                if user_id is not None and not await self._subject_gate.allows(guild_id, user_id):
                    return True
        current = await self._store.governance_generations(guild_id, tuple(snapshot))
        return any(current.get(user_id, 0) > generation for user_id, generation in snapshot.items())

    async def _maybe_refresh_summary(
        self,
        key: BatchKey,
        adds: int,
    ) -> None:
        """Regenerate the profile digest after enough lifetime facts."""
        if key.subject_key == SERVER_SUBJECT_KEY or self.consolidation is None:
            return
        try:
            await self.consolidation.maybe_refresh_profile(
                guild_id=key.guild_id,
                subject_id=key.subject_key,
                adds=adds,
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
        skip_reconcile: bool = False,
        governance: dict[str, int] | None = None,
        lease_owner: str | None = None,
    ) -> None:
        batch_subject = None if key.subject_key == SERVER_SUBJECT_KEY else key.subject_key
        stamped = governance if governance is not None else {}
        if candidates and self._embedder is not None:
            # One batched embed call per batch; reconcile lookups and the
            # commit upsert reuse these instead of embedding per candidate.
            pending = {
                norm: candidate.text
                for candidate, _, _ in candidates
                if (norm := normalize_text(candidate.text)) not in embeddings_by_text
            }
            if pending:
                vectors = await self._embedder.embed(tuple(pending.values()))
                for norm, vector in zip(pending, vectors, strict=True):
                    embeddings_by_text[norm] = vector
        plan = await self._reconciler.build_plan(
            candidates,
            guild_id=guild_id,
            batch_subject_id=batch_subject,
            embeddings_by_text=embeddings_by_text,
            source_channel=source_refs[0].channel_id if source_refs else None,
        )
        # Budget degradation never invents a reconciliation judgment.
        decisions_map: dict[int, tuple[ReconcileDecision, ...]] = {}
        if not skip_reconcile:
            decisions_map = await self._reconciler.resolve_collisions(
                plan.collisions, guild_id=guild_id
            )

        if any(not decisions_map.get(index) for index in range(len(plan.collisions))):
            raise RuntimeError(
                "reconciliation unresolved; batch retained in dead letters for retry"
            )

        if lease_owner is not None and not await self._queue.renew_lease(
            key,
            owner=lease_owner,
            now=self._clock.now(),
            lease_seconds=self._config.batching.lease_seconds,
        ):
            raise asyncio.CancelledError("ingest lease lost before commit")
        current = await self._store.governance_generations(guild_id, tuple(stamped))
        if any(generation < current.get(user_id, 0) for user_id, generation in stamped.items()):
            logger.debug(
                "batch stale after reconcile planning guild=%s subject=%s — mutations dropped",
                guild_id,
                key.subject_key,
            )
            return
        notifications: list[Callable[[], None]] = []
        async with self._store.transaction():
            # A guild commit fence also serializes Mongo snapshot writers. Planning
            # happens outside transactions; changed neighbors invalidate the plan.
            await self._store.set_cursor(guild_id, "commit_fence", uuid4().hex)
            if any(
                [
                    await self._store.get_cursor(guild_id, _receipt_key(key, ref.message_id))
                    is not None
                    for ref in source_refs
                ]
            ):
                raise RuntimeError("input committed concurrently; retry with remaining messages")
            targets = [record for record, *_ in plan.reinforces]
            targets.extend(
                record for collision in plan.collisions for record in collision.neighbors
            )
            for planned_record in targets:
                current_record = await self._store.get_fact(guild_id, planned_record.id)
                if current_record is None or current_record.version != planned_record.version:
                    raise RuntimeError("reconciliation plan changed; retry from current memory")
            for proposal, subject, _speaker in plan.direct_adds:
                duplicate = await self._store.find_duplicate(
                    guild_id, subject, normalize_text(proposal.text)
                )
                if duplicate is not None and duplicate.is_active:
                    from icelake.permissions import source_channels

                    if not source_refs or source_channels(duplicate) == {source_refs[0].channel_id}:
                        raise RuntimeError("concurrent duplicate; rebuild reconciliation plan")
            for record, proposal, _subject_id, _speaker_id in plan.reinforces:
                # Deterministic reinforce: exact/near-duplicate — no LLM judgment needed.
                await self._committer.commit_reinforce(
                    record,
                    proposal,
                    source_refs=_pick_citations(proposal.source_message_indexes, source_refs),
                )
                summary.reinforces += 1

            for proposal, subject_id, speaker_id in plan.direct_adds:
                record = await self._committer.commit_add(
                    proposal=proposal,
                    subject_id=subject_id,
                    speaker_id=speaker_id,
                    guild_id=guild_id,
                    roster=roster,
                    mentioned_ids=mentioned_ids,
                    source_refs=_pick_citations(proposal.source_message_indexes, source_refs),
                    skip_embedding=True,
                    embedding=embeddings_by_text.get(normalize_text(proposal.text)),
                    governance=stamped,
                )
                summary.adds += 1
                notifications.append(partial(self._publish_fact, record, reinforced=False))

            for index, collision in enumerate(plan.collisions):
                resolved = False
                add_candidate = False
                invalidated: tuple[str, str] | None = None  # (old_fact_id, reason)
                for decision in decisions_map.get(index, ()):
                    if decision.kind is ReconcileKind.NOOP:
                        # "same meaning" => strengthen the existing fact.
                        target = next(
                            (n for n in collision.neighbors if n.id == decision.target_id),
                            None,
                        )
                        if target is not None:
                            await self._committer.commit_reinforce(
                                target,
                                collision.candidate,
                                source_refs=_pick_citations(
                                    collision.candidate.source_message_indexes, source_refs
                                ),
                            )
                            summary.reinforces += 1
                            resolved = True
                            break
                        continue
                    if decision.kind is ReconcileKind.ADD:
                        add_candidate = True
                        break
                    target = next(
                        (n for n in collision.neighbors if n.id == decision.target_id),
                        None,
                    )
                    if target is None:
                        continue
                    if decision.kind is ReconcileKind.UPDATE:
                        fresh, _old = await self._committer.commit_supersede(
                            old_record=target,
                            proposal=collision.candidate,
                            subject_id=collision.subject_id,
                            speaker_id=collision.speaker_id,
                            reason=decision.reason,
                            guild_id=guild_id,
                            roster=roster,
                            mentioned_ids=mentioned_ids,
                            source_refs=_pick_citations(
                                collision.candidate.source_message_indexes,
                                source_refs,
                            ),
                            governance=stamped,
                        )
                        summary.supersessions += 1
                        notifications.append(
                            partial(
                                self._publish_superseded,
                                guild_id,
                                target.id,
                                fresh.id,
                                decision.reason,
                            )
                        )
                        resolved = True
                        break
                    if decision.kind is ReconcileKind.INVALIDATE:
                        await self._committer.commit_invalidate(
                            old_record=target,
                            reason=decision.reason,
                        )
                        summary.invalidations += 1
                        invalidated = (target.id, decision.reason)
                        # The candidate is the new truth — commit it below.
                        add_candidate = True
                        break
                if resolved:
                    continue
                if not add_candidate:
                    # Similarity cannot establish agreement. Preserve the existing
                    # fact unchanged when no valid reconciliation judgment exists.
                    continue
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
                    skip_embedding=True,
                    embedding=embeddings_by_text.get(normalize_text(collision.candidate.text)),
                    governance=stamped,
                )
                summary.adds += 1
                notifications.append(partial(self._publish_fact, record, reinforced=False))
                if invalidated is not None:
                    notifications.append(
                        partial(
                            self._publish_superseded,
                            guild_id,
                            invalidated[0],
                            record.id,
                            invalidated[1],
                        )
                    )
            for ref in source_refs:
                await self._store.set_cursor(
                    guild_id, _receipt_key(key, ref.message_id), self._clock.now().isoformat()
                )

        for notify in notifications:
            notify()

    async def _mine_message_aliases(
        self,
        guild_id: str,
        messages: tuple[StoredMessage, ...],
    ) -> None:
        """Learn real-name aliases from raw messages.

        Runs on message text, not just extracted facts: a name mentioned in a
        noise-gated batch is still learned. First-person patterns only, bound
        to the speaker — third-person patterns here would bind names others
        mention to the wrong user.
        """
        for message in messages:
            if message.author_is_bot:
                continue
            found = extract_self_name_aliases(message.content)
            await self._upsert_name_aliases(guild_id, message.author_id, message.content, found)

    async def _mine_fact_aliases(
        self,
        guild_id: str,
        candidates: list[tuple[ProposedFact, str | None, str | None]],
    ) -> None:
        """Harvest names from LLM-normalized fact text ("nulzo's name is
        Nolan Gregory"), bound to the fact subject. Skips server-scope and
        third-party-attributed facts, where the subject didn't state it."""
        for proposal, subject_id, speaker_id in candidates:
            if subject_id is None or (speaker_id is not None and speaker_id != subject_id):
                continue
            found = extract_stated_name_aliases(proposal.text)
            await self._upsert_name_aliases(guild_id, subject_id, proposal.text, found)

    async def _upsert_name_aliases(
        self,
        guild_id: str,
        user_id: str,
        text: str,
        found: list[tuple[str, float]],
    ) -> None:
        for surface, _weight in found:
            if is_third_party_name_reference(text, surface):
                continue
            alias_norm = normalize_alias(surface)
            if not is_valid_alias(alias_norm):
                continue
            await self._store.upsert_alias(
                guild_id,
                alias_norm,
                user_id,
                AliasSource.REAL_NAME,
                weight_for_source(AliasSource.REAL_NAME),
            )

    def _publish_fact(self, record: FactRecord, *, reinforced: bool) -> None:
        if self.event_bus is not None:
            from icelake.models.events import FactCommitted

            self.event_bus.publish(
                FactCommitted(
                    guild_id=record.guild_id,
                    fact_id=record.id,
                    subject_id=record.subject_id,
                    text=record.text,
                    was_reinforcement=reinforced,
                )
            )

    def _publish_superseded(
        self, guild_id: str, old_fact_id: str, new_fact_id: str | None, reason: str
    ) -> None:
        if self.event_bus is not None:
            from icelake.models.events import FactSupersededEvent

            self.event_bus.publish(
                FactSupersededEvent(
                    guild_id=guild_id,
                    old_fact_id=old_fact_id,
                    new_fact_id=new_fact_id,
                    reason=reason,
                )
            )

    async def _build_roster(
        self,
        messages: tuple[StoredMessage, ...],
        *,
        author_id: str | None,
    ) -> Roster:
        """Build a participant roster with alias-enriched display names.

        Mention IDs are resolved through the alias index so the LLM sees
        human-readable names instead of raw snowflakes.
        """
        roster = Roster()
        seen: set[str] = set()
        guild_id = messages[0].guild_id if messages else ""

        async def add(user_id: str | None, display_name: str) -> None:
            if not user_id or user_id in seen:
                return
            seen.add(user_id)
            aliases = await self._store.aliases_for_user(guild_id, user_id)
            best_alias = max(aliases, key=lambda r: r.weight) if aliases else None
            name = (
                best_alias.alias_norm
                if best_alias and best_alias.source.rank >= 60
                else display_name
            )
            roster.add(user_id, name)

        if author_id:
            first = messages[0]
            await add(author_id, first.author_display_name or first.author_username or author_id)
        for message in messages:
            await add(message.author_id, message.author_display_name or message.author_username)
            for mention_id in self._subject_gate.exclude_bots(message.mention_ids):
                await add(mention_id, mention_id)
        return roster
