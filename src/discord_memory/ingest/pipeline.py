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
from discord_memory.identity.aliases import (
    extract_self_name_aliases,
    extract_stated_name_aliases,
    is_third_party_name_reference,
    is_valid_alias,
    normalize_alias,
    weight_for_source,
)
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
from discord_memory.models.identity import AliasSource
from discord_memory.models.operations import ProposedFact, ReconcileKind
from discord_memory.ports.clock import Clock, IdGen
from discord_memory.ports.llm import ChatLLM, Embedder, Meter
from discord_memory.ports.queue import BatchKey, IngestQueue, StoredMessage
from discord_memory.ports.store import MemoryStore
from discord_memory.ports.vectors import VectorIndex, VectorItem

if TYPE_CHECKING:
    from discord_memory.api.events import EventBus
    from discord_memory.consolidation.service import ConsolidationService

logger = logging.getLogger(__name__)

SERVER_SUBJECT_KEY = "__server__"


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
            store, vectors, reconcile_llm or llm, embedder, config.extraction
        )
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
        self.consolidation: ConsolidationService | None = None

    def attach_event_bus(self, bus: EventBus) -> None:
        """Wire an event bus for hook dispatch (called by the facade)."""
        self.event_bus = bus

    def attach_consolidation(self, consolidation: ConsolidationService) -> None:
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
            # The lease guards in-flight work only; lingering after completion
            # would stall other processes/owners for the remaining lease window.
            # Empty claims skip it: the lease belongs to the in-flight batch.
            if message_ids:
                await self._queue.release_key(key, owner=self.owner)

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
                return BatchReport(key=key, skipped_reason="empty")
            if len(window) < self._config.batching.batch_size_messages:
                # Community roll-up waits for batch-sized volume; per-message
                # culture extraction doubles cost and races the user batches.
                return BatchReport(key=key, skipped_reason="below_min_volume")

            summary = CommitSummary()
            try:
                report = await self._process_claimed(key, window)
                if report.summary is not None:
                    summary = report.summary
            except Exception:
                logger.exception("Server window %s failed", key.as_tuple)
                return BatchReport(key=key, skipped_reason="error")
            # Advance the watermark to the newest processed message so the next
            # pass only ever sees genuinely new activity.
            newest = max(window, key=lambda m: (m.created_at, m.message_id))
            await self._store.set_cursor(
                key.guild_id,
                "server_window",
                f"{newest.created_at.isoformat()}|{newest.message_id}",
            )
            return BatchReport(
                key=key,
                messages_processed=len(window),
                summary=summary,
            )
        finally:
            await self._queue.release_key(key, owner=self.owner)

    async def _process_claimed(
        self,
        key: BatchKey,
        messages: tuple[StoredMessage, ...],
    ) -> BatchReport:
        guild_id = key.guild_id
        is_server_scope = key.subject_key == SERVER_SUBJECT_KEY
        subject_key_author = None if is_server_scope else key.subject_key

        await self._mine_message_aliases(guild_id, messages)

        texts = tuple((m.author_username or m.author_id, m.content) for m in messages)
        joined_text = " ".join(content for _, content in texts)

        if self._config.extraction.noise_gate and not batch_worth_extracting(
            tuple(m.content for m in messages),
        ):
            return BatchReport(key=key, messages_processed=len(messages), skipped_reason="noise")

        step = self._meter.check_budget(guild_id)
        if step is BudgetStep.SKIP_EXTRACTION:
            return BatchReport(key=key, messages_processed=len(messages), skipped_reason="budget")

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
        )

        result = await self._extractor.extract(
            roster=roster,
            messages=texts,
            existing_memories_block=existing_block,
            guild_id=guild_id,
        )
        summary = CommitSummary()
        for text, reason in result.rejected:
            logger.debug("Rejected candidate (%s): %s", reason, text)

        # In-batch candidate dedup: identical/near-identical proposals within
        # one response collapse into the first occurrence.
        seen_norms: set[tuple[str | None, str]] = set()
        candidates = []
        for vetted in result.vetted:
            bound = vetted.proposal.model_copy(
                update={"text": roster.bind_names(vetted.proposal.text)},
            )
            key_tuple = (vetted.subject_id, normalize_text(bound.text))
            if key_tuple in seen_norms:
                continue
            seen_norms.add(key_tuple)
            candidates.append((bound, vetted.subject_id, vetted.speaker_id))
        if is_server_scope:
            # The community pass owns guild-scope facts only. User-anchored
            # facts belong to per-user batches — otherwise both passes extract
            # the same messages concurrently and race duplicates into the store.
            candidates = [c for c in candidates if c[1] is None]
        mentioned_ids = self._subject_gate.exclude_bots(
            mention_id for message in messages for mention_id in message.mention_ids
        )
        source_refs = _build_source_refs(messages)
        if candidates:
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
        """Regenerate the profile digest after enough lifetime facts (PLAN.md Part 7)."""
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
    ) -> None:
        batch_subject = None if key.subject_key == SERVER_SUBJECT_KEY else key.subject_key
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
        )
        decisions_map = await self._reconciler.resolve_collisions(
            plan.collisions, guild_id=guild_id
        )

        committed_records = []
        for record, proposal, _subject_id, _speaker_id in plan.reinforces:
            # Deterministic reinforce: exact/near-duplicate — no LLM judgment needed.
            await self._committer.commit_reinforce(record, proposal)
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
            )
            summary.adds += 1
            committed_records.append(record)
            self._publish_fact(record, reinforced=False)

        for index, collision in enumerate(plan.collisions):
            resolved = False
            add_candidate = False
            invalidated: tuple[str, str] | None = None  # (old_fact_id, reason)
            for decision in decisions_map.get(index, ()):
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
                    )
                    summary.supersessions += 1
                    self._publish_superseded(guild_id, target.id, fresh.id, decision.reason)
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
                # Conservative default on unresolved collisions: reinforce the
                # strongest semantic neighbor instead of adding a near-duplicate.
                neighbor = collision.semantic_neighbors[0] if collision.semantic_neighbors else None
                if neighbor is not None:
                    await self._committer.commit_reinforce(neighbor, collision.candidate)
                    summary.reinforces += 1
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
            )
            summary.adds += 1
            committed_records.append(record)
            self._publish_fact(record, reinforced=False)
            if invalidated is not None:
                self._publish_superseded(guild_id, invalidated[0], record.id, invalidated[1])

        # One vector upsert pass; candidates were embedded before build_plan.
        if committed_records and self._vectors is not None:
            for record in committed_records:
                embedding = embeddings_by_text.get(normalize_text(record.text))
                if embedding is None:
                    continue
                await self._vectors.upsert(
                    (
                        VectorItem(
                            id=record.id,
                            guild_id=record.guild_id,
                            subject_id=record.subject_id,
                            embedding=embedding,
                        ),
                    )
                )

    async def _mine_message_aliases(
        self,
        guild_id: str,
        messages: tuple[StoredMessage, ...],
    ) -> None:
        """Learn real-name aliases from raw messages (PLAN.md §3.2 write-time).

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

    def _publish_superseded(
        self, guild_id: str, old_fact_id: str, new_fact_id: str | None, reason: str
    ) -> None:
        if self.event_bus is not None:
            from discord_memory.models.events import FactSupersededEvent

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
