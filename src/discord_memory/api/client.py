"""DiscordMemory: the consumer facade (API.md Parts 2-3).

Composition root — builds adapters from config, applies port overrides, owns the
worker lifecycle, and exposes the namespaced capability groups. All heavy lifting
lives in focused services; this class only wires and delegates.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable

from discord_memory.adapters.embedders import build_embedder
from discord_memory.api.classify import CommandClassifier, UserMemoryCommand
from discord_memory.api.events import EventBus
from discord_memory.api.facts_api import FactsApi
from discord_memory.api.groups import AdminApi, GraphApi, IdentityApi
from discord_memory.config import MemoryConfig
from discord_memory.consolidation.service import ConsolidationService
from discord_memory.errors import ConfigError, StorageUnavailableError
from discord_memory.identity.guards import BotGuard, ConsentPolicy, SubjectGate
from discord_memory.ingest.pipeline import SERVER_SUBJECT_KEY, IngestPipeline
from discord_memory.lifecycle.maintenance import MaintenanceService
from discord_memory.models.admin import (
    ComponentHealth,
    GuildStats,
    HealthReport,
    HealthStatus,
    MeterSnapshot,
)
from discord_memory.models.common import TokenUsage
from discord_memory.models.events import (
    IgnoreReason,
    MessageEvent,
    ObserveReceipt,
    ObserveStatus,
    RejectReason,
)
from discord_memory.models.identity import AliasSource
from discord_memory.models.retrieval import (
    PromptContext,
    RecallQuery,
    RecallResult,
    RecallWarning,
    Resolution,
    Scope,
    ScoredFact,
)
from discord_memory.ports.clock import Clock, IdGen, SystemClock, UlidIdGen
from discord_memory.ports.llm import ChatLLM, Embedder, LlmCache, Meter
from discord_memory.ports.queue import IngestQueue, StoredMessage
from discord_memory.ports.store import MemoryStore
from discord_memory.ports.vectors import VectorIndex
from discord_memory.retrieval.injection import InjectionBuilder, estimate_tokens
from discord_memory.retrieval.service import RecallService

logger = logging.getLogger(__name__)
log = logger

_MISSING = object()
_SERVER_BLOCK_FACTS = 3


class _OpsApi:
    """Worker control plane — the ``memory.ops`` namespace."""

    def __init__(self, client: DiscordMemory) -> None:
        self._client = client

    async def run_pending(self, *, limit_batches: int = 8) -> int:
        """Process due batches now (cron/external-scheduler mode)."""
        processed = await self._client._pipeline.run_pending(limit_batches=limit_batches)
        for guild_id in list(self._client.active_guilds):
            if self._client.maintenance.due(guild_id):
                await self._client.maintenance.run_guild(guild_id)
        return processed

    async def backfill_aliases(
        self,
        guild_id: str,
        members: Iterable[tuple[str, str, str]],
    ) -> int:
        """Bulk identity backfill from a member directory.

        ``members`` yields ``(user_id, username, display_name)``. Fixes cold-start
        resolution: users who never spoke since install become resolvable.
        """
        registered = 0
        for user_id, username, display_name in members:
            if username and not username.isdigit():
                await self._client.identity.register_alias(
                    guild_id,
                    user_id,
                    username,
                    source=AliasSource.DISCORD_USERNAME,
                )
                registered += 1
            if display_name and not display_name.isdigit():
                await self._client.identity.register_alias(
                    guild_id,
                    user_id,
                    display_name,
                    source=AliasSource.DISPLAY_NAME,
                )
        return registered

    async def retry_dead_letters(self, guild_id: str | None = None) -> int:
        """Re-drive poison jobs after fixing an underlying failure."""
        return await self._client._queue.requeue_dead_letters(guild_id)

    def meter_snapshot(self) -> MeterSnapshot:
        """Cumulative token/call counters by purpose."""
        return self._client._meter.snapshot()

    async def health(self) -> HealthReport:
        client = self._client
        storage_ok = await client._store.ping()
        components = [
            ComponentHealth(
                component="storage",
                status=HealthStatus.OK if storage_ok else HealthStatus.DOWN,
            ),
            ComponentHealth(
                component="llm",
                status=HealthStatus.OK if client._llm is not None else HealthStatus.DEGRADED,
                detail="" if client._llm is not None else "extraction disabled (no llm)",
            ),
        ]
        pending = 0
        dead = 0
        for guild_id in list(client.active_guilds):
            pending += await client._queue.pending_count(guild_id)
            dead += await client._queue.dead_letter_count(guild_id)
        return HealthReport(
            components=tuple(components),
            pending_messages=pending,
            dead_letters=dead,
        )


class DiscordMemory:
    """The only class consumers need. See docs/API.md for the full contract."""

    def __init__(self, config: MemoryConfig, **overrides: object) -> None:
        self.config = config
        self.events = EventBus()
        clock = overrides.get("clock", _MISSING)
        self._clock: Clock = SystemClock() if clock is _MISSING else clock  # type: ignore[assignment]
        id_gen = overrides.get("id_gen", _MISSING)
        self._id_gen: IdGen = UlidIdGen() if id_gen is _MISSING else id_gen  # type: ignore[assignment]

        store_override = overrides.get("store", _MISSING)
        self._store: MemoryStore = (
            _build_store(config) if store_override is _MISSING else store_override  # type: ignore[assignment]
        )
        queue_override = overrides.get("queue", _MISSING)
        backend_queue = getattr(self._store, "queue", None)
        self._queue: IngestQueue = (
            backend_queue if queue_override is _MISSING else queue_override  # type: ignore[assignment]
        ) or _in_memory_queue()

        vectors_override = overrides.get("vectors", _MISSING)
        if vectors_override is _MISSING:
            self._vectors: VectorIndex | None = getattr(self._store, "vectors", None)
        else:
            self._vectors = vectors_override  # type: ignore[assignment]

        embedder_override = overrides.get("embedder", _MISSING)
        self._embedder: Embedder | None = (
            build_embedder(config.embeddings)
            if embedder_override is _MISSING
            else embedder_override  # type: ignore[assignment]
        )

        meter_override = overrides.get("meter", _MISSING)
        if meter_override is _MISSING:
            from discord_memory.adapters.meter import InMemoryMeter

            self._meter: Meter = InMemoryMeter(config.budgets, self._clock)
        else:
            self._meter = meter_override  # type: ignore[assignment]

        llm_override = overrides.get("llm", _MISSING)
        llm_cache: LlmCache | None = None
        if config.llm.cache_responses:
            candidate = getattr(self._store, "llm_cache", None)
            llm_cache = candidate if isinstance(candidate, LlmCache) else None
        self._llm: ChatLLM | None = (
            _build_llm(config, self._meter, cache=llm_cache)
            if llm_override is _MISSING
            else llm_override  # type: ignore[assignment]
        )
        small_llm_override = overrides.get("small_llm", _MISSING)
        if small_llm_override is not _MISSING:
            self._small_llm: ChatLLM | None = small_llm_override  # type: ignore[assignment]
        elif llm_override is _MISSING and config.llm.small_model:
            self._small_llm = _build_llm(
                config, self._meter, model=config.llm.small_model, cache=llm_cache
            )
        else:
            self._small_llm = self._llm

        self._guard = BotGuard()
        consent = ConsentPolicy(self._store)
        self._subject_gate = SubjectGate(self._guard, consent)

        self._pipeline = IngestPipeline(
            config=config,
            clock=self._clock,
            id_gen=self._id_gen,
            queue=self._queue,
            store=self._store,
            vectors=self._vectors,
            embedder=self._embedder,
            llm=self._llm,
            reconcile_llm=self._small_llm,
            meter=self._meter,
            subject_gate=self._subject_gate,
        )
        self._pipeline.attach_event_bus(self.events)
        self.maintenance = MaintenanceService(
            store=self._store,
            config=config,
            clock=self._clock,
        )

        async def _group_gate() -> None:
            await self.ensure_started()

        self.facts = FactsApi(
            store=self._store,
            vectors=self._vectors,
            embedder=self._embedder,
            clock=self._clock,
            id_gen=self._id_gen,
            config=config,
            subject_gate=self._subject_gate,
            startup_gate=_group_gate,
        )
        self.identity = IdentityApi(self._store, startup_gate=_group_gate)
        self.graph = GraphApi(store=self._store, startup_gate=_group_gate)
        self.admin = AdminApi(self._store, startup_gate=_group_gate)
        self.ops = _OpsApi(self)
        self._classifier = CommandClassifier(self._small_llm)
        self._injection = InjectionBuilder()
        self._consolidation = ConsolidationService(
            store=self._store,
            llm=self._small_llm,
            embedder=self._embedder,
            config=config,
        )
        self._pipeline.attach_consolidation(self._consolidation)

        self.started = False
        self.closing = False
        self._shutdown_done = False
        self.worker_tasks: list[asyncio.Task[None]] = []
        self.active_guilds: set[str] = set()

    # -- lifecycle ---------------------------------------------------------------

    async def ensure_started(self) -> None:
        """Open storage + launch workers on first use (idempotent, lazy).

        Consumers may skip ``start()`` entirely: the first memory call
        initializes the backend automatically (mem0-parity frictionless setup).
        """
        if self.started or self.closing:
            return
        await self.start()

    async def start(self) -> None:
        """Open storage and launch workers (idempotent)."""
        if self.started or self._shutdown_done:
            return
        await self._store.setup()
        self.started = True
        if self.config.workers.enabled:
            for index in range(self.config.workers.count):
                task = asyncio.create_task(
                    self._worker_loop(),
                    name=f"discord-memory-worker-{index}",
                )
                self.worker_tasks.append(task)

    async def close(self, *, drain: bool = True, timeout_seconds: float = 30.0) -> None:
        """Stop accepting work; drain in-flight batches when ``drain=True``."""
        if not self.started or self.closing:
            return
        self.closing = True
        self._shutdown_done = True
        if self.worker_tasks:
            try:
                # Let in-flight batches finish: the loop exits after its current
                # pass, so their claims complete instead of stranding mid-commit.
                await asyncio.wait_for(
                    asyncio.gather(*self.worker_tasks, return_exceptions=True),
                    timeout_seconds,
                )
            except TimeoutError:
                logger.warning("worker drain timeout; cancelling in-flight batches")
                for task in self.worker_tasks:
                    task.cancel()
                await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        self.worker_tasks.clear()
        if drain:
            try:
                await asyncio.wait_for(
                    self.ops.run_pending(limit_batches=10_000),
                    timeout_seconds,
                )
            except TimeoutError:
                logger.warning("drain timeout exceeded; abandoning in-flight batches")
        await self._store.close()
        self.started = False

    async def __aenter__(self) -> DiscordMemory:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # -- ingestion ------------------------------------------------------------------

    async def observe(self, event: MessageEvent) -> ObserveReceipt:
        """Record one message for passive extraction. Never raises operational errors."""
        try:
            await self.ensure_started()
        except Exception:
            log.exception("lazy start failed")
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.REJECTED,
                reason=RejectReason.STORAGE_UNAVAILABLE,
            )
        try:
            return await self._observe_inner(event)
        except StorageUnavailableError:
            # Expected after close(); a listener must never see this raise.
            log.info("observe on closed/unavailable storage -> rejected receipt")
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.REJECTED,
                reason=RejectReason.STORAGE_UNAVAILABLE,
            )

    async def _observe_inner(self, event: MessageEvent) -> ObserveReceipt:
        observe_cfg = self.config.observe

        # 1. Bot guard (structural — never a subject).
        self._guard.note_author(event.author_id, is_bot=event.author_is_bot)
        if event.author_is_bot or self._guard.is_bot(event.author_id):
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.IGNORED,
                reason=IgnoreReason.BOT_AUTHOR,
            )

        # 2. Consent (policy decision).
        if await self._subject_gate.allows(event.guild_id, event.author_id) is False:
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.IGNORED,
                reason=IgnoreReason.OPTED_OUT,
            )

        # 3. Content quality gates.
        if not event.content.strip() or len(event.content.strip()) < observe_cfg.min_message_chars:
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.IGNORED,
                reason=IgnoreReason.EMPTY_CONTENT,
            )
        for pattern in observe_cfg.ignore_patterns:
            if re.search(pattern, event.content):
                return ObserveReceipt(
                    message_id=event.message_id,
                    status=ObserveStatus.IGNORED,
                    reason=IgnoreReason.IGNORED_PATTERN,
                )

        import hashlib

        stored_content = (
            event.content
            if self.config.privacy.store_raw_messages
            else hashlib.sha256(event.content.encode()).hexdigest()[:16]
        )
        message = StoredMessage(
            message_id=event.message_id,
            guild_id=event.guild_id,
            channel_id=event.channel_id,
            author_id=event.author_id,
            subject_key=event.author_id,
            content=stored_content,
            created_at=event.created_at,
            author_username=event.author_username,
            author_display_name=event.author_display_name,
            author_is_bot=event.author_is_bot,
            mention_ids=event.mention_ids,
        )
        max_depth = self.config.observe.max_queue_depth_per_guild
        try:
            accepted = await self._queue.put_message(message, max_depth=max_depth)
        except Exception:
            log.exception("observe: queue persistence failed")
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.REJECTED,
                reason=RejectReason.STORAGE_UNAVAILABLE,
            )
        if not accepted:
            # Distinguish duplicate vs capacity by checking pending count.
            pending = await self._queue.pending_count(event.guild_id)
            if max_depth is not None and pending >= max_depth:
                return ObserveReceipt(
                    message_id=event.message_id,
                    status=ObserveStatus.REJECTED,
                    reason=RejectReason.QUEUE_OVER_CAPACITY,
                )
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.IGNORED,
                reason=IgnoreReason.DUPLICATE,
            )
        self.active_guilds.add(event.guild_id)
        try:
            await self._register_author_alias(event)
        except Exception:
            log.warning("alias registration failed for %s", event.author_id)
        return ObserveReceipt(message_id=event.message_id, status=ObserveStatus.ACCEPTED)

    async def observe_many(
        self,
        events: tuple[MessageEvent, ...],
    ) -> tuple[ObserveReceipt, ...]:
        """Bulk ingest for history backfill; per-item receipts preserve order."""
        receipts: list[ObserveReceipt] = []
        for event in events:
            receipts.append(await self.observe(event))
        return tuple(receipts)

    async def flush(self, guild_id: str | None = None) -> int:
        """Force-extract every due batch (optionally one guild). Returns processed count."""
        keys = await self._queue.due_batch_keys(
            now=self._clock.now(),
            batch_size=1,
            max_age_seconds=0.0,
            limit=self.config.batching.batch_size_messages * 128,
        )
        processed = 0
        for key in keys:
            if guild_id is not None and key.guild_id != guild_id:
                continue
            report = await self._pipeline.process_key(key)
            if report.summary is not None:
                processed += 1
        return processed

    # -- retrieval ---------------------------------------------------------------

    async def recall(self, query: RecallQuery) -> RecallResult:
        """Explicit retrieval; see :class:`RecallQuery` for the query model."""
        await self.ensure_started()
        consent = ConsentPolicy(self._store)

        async def _blocked(guild_id: str, user_id: str) -> bool:
            return await consent.is_blocked(guild_id, user_id)

        async def _reinforce_recalled(fact_ids: list[str]) -> None:
            # mem0-style decay loop (opt-in): recalled facts get their decay
            # clock reset, so frequently-served knowledge floats up over time.
            await self._store.touch_facts(
                query.guild_id,
                tuple(fact_ids),
                accessed_at=self._clock.now(),
            )

        service = RecallService(
            store=self._store,
            vectors=self._vectors,
            embedder=self._embedder,
            config=self.config.retrieval,
            guard=self._guard,
            clock=self._clock,
            is_subject_blocked=_blocked,
            on_recalled=(
                _reinforce_recalled if self.config.retrieval.reinforce_on_recall else None
            ),
        )
        return await service.recall(query)

    async def prompt_context(
        self,
        *,
        guild_id: str,
        asker_id: str,
        text: str | None = None,
        mentioned_ids: tuple[str, ...] = (),
        thread_participant_ids: tuple[str, ...] = (),
        token_budget_tokens: int | None = None,
    ) -> PromptContext:
        """One-call turn context: subjects resolved, facts injected, citations bound.

        This is the frictionless hot path — pass the current message and mention ids;
        get back a paste-ready injection block plus citation bindings.
        """
        await self.ensure_started()
        budget = token_budget_tokens or self.config.retrieval.default_token_budget
        subjects, resolutions, warnings_list = await self._resolve_subjects(
            guild_id=guild_id,
            candidates=[asker_id, *mentioned_ids, *thread_participant_ids],
        )
        warnings: list[RecallWarning] = list(warnings_list)

        result = await self.recall(
            RecallQuery(
                guild_id=guild_id,
                text=text,
                subject_ids=tuple(subjects),
                scope=Scope.SUBJECTS,
            )
        )
        server_result = await self.recall(
            RecallQuery(
                guild_id=guild_id,
                text=text,
                scope=Scope.SERVER,
                top_k=_SERVER_BLOCK_FACTS,
                max_per_subject=_SERVER_BLOCK_FACTS,
            )
        )

        sections: dict[str, tuple[ScoredFact, ...]] = {}
        summaries: dict[str, str | None] = {}
        alias_notes: dict[str, str] = {}

        asker_summary_doc = await self._store.get_summary(guild_id, asker_id)
        summaries["asker"] = asker_summary_doc.text if asker_summary_doc else None
        sections["asker"] = tuple(sf for sf in result.facts if sf.fact.subject_id == asker_id)

        other_map: dict[str, list[ScoredFact]] = {}
        for scored in result.facts:
            subject = scored.fact.subject_id
            if subject and subject != asker_id:
                other_map.setdefault(subject, []).append(scored)
        for subject, scored_list in other_map.items():
            key = f"user:{subject}"
            sections[key] = tuple(scored_list)
            doc = await self._store.get_summary(guild_id, subject)
            summaries[key] = doc.text if doc else None
            alias_records = await self._store.aliases_for_user(guild_id, subject)
            names = sorted({record.alias_norm for record in alias_records})
            if len(names) > 1:
                alias_notes[key] = (
                    f"Coreference: these names all refer to ONE person: {', '.join(names)}."
                )

        sections["server"] = tuple(server_result.facts)
        server_doc = await self._store.get_summary(guild_id, None)
        summaries["server"] = server_doc.text if server_doc else None

        block, citations, trimmed = self._injection.build(
            asker_id=asker_id,
            facts_by_section=sections,
            summaries=summaries,
            token_budget=budget,
            guild_id=guild_id,
            alias_notes=alias_notes,
        )
        if trimmed:
            warnings.append(RecallWarning.BUDGET_TRIMMED)
        usage_estimate = estimate_tokens(block)
        return PromptContext(
            injection_block=block,
            facts=result.facts + server_result.facts,
            citations=citations,
            resolutions=tuple(resolutions),
            asker_summary=summaries["asker"],
            usage=TokenUsage(prompt_tokens=usage_estimate),
            warnings=tuple(dict.fromkeys(warnings)),
        )

    def register_bot_id(self, user_id: int | str) -> None:
        """Teach the bot-guard its own id (never a memory subject)."""
        self._guard.register(str(user_id))

    # -- misc surface -------------------------------------------------------------

    async def extract_now(self, event: MessageEvent) -> ObserveReceipt:
        """Synchronous add: observe + immediate flush. mem0-parity for tests
        and onboarding flows; production bots should prefer ``observe``."""
        receipt = await self.observe(event)
        if receipt.status is not ObserveStatus.ACCEPTED:
            return receipt
        await self.flush(event.guild_id)
        return receipt

    async def stats(self, guild_id: str) -> GuildStats:
        """Guild memory statistics snapshot."""
        await self.ensure_started()
        return await self._store.guild_stats(guild_id)

    async def classify_command(self, text: str) -> UserMemoryCommand:
        """Detect 'remember X' / 'forget Y' intents. Execution stays consumer-side."""
        return await self._classifier.classify(text)

    async def regenerate_summaries(self, guild_id: str, user_ids: tuple[str, ...] = ()) -> int:
        """Manually refresh profile digests (normally background-scheduled)."""
        count = 0
        targets = user_ids
        if not targets:
            page = await self._store.list_facts(
                guild_id, subject_id="__list__", active_only=False, limit=1
            )
            del page
            rows = await self._store.top_strength_facts(guild_id, subject_ids=None, limit=500)
            targets = tuple({record.subject_id for record in rows if record.subject_id})
        for subject_id in targets:
            summary = await self._consolidation.regenerate_profile(
                guild_id=guild_id,
                subject_id=subject_id,
            )
            if summary is not None:
                count += 1
        server_summary = await self._consolidation.regenerate_profile(
            guild_id=guild_id,
            subject_id=None,
        )
        if server_summary is not None:
            count += 1
        return count

    # -- internals --------------------------------------------------------------

    async def _resolve_subjects(
        self,
        *,
        guild_id: str,
        candidates: list[str],
    ) -> tuple[list[str], list[Resolution], list[RecallWarning]]:
        subjects: list[str] = []
        resolutions: list[Resolution] = []
        warnings: list[RecallWarning] = []
        for identifier in dict.fromkeys(candidates):
            resolution = await self.identity.resolve(guild_id, identifier)
            if resolution.resolved is None and not resolution.ambiguous:
                # Ladder rung 3 (PLAN §3.2): stored name-facts. A single
                # unambiguous self-name match yields a flagged candidate —
                # never auto-attributed without the confidence signal.
                fallback = await self._resolve_via_name_facts(guild_id, identifier)
                if fallback is not None:
                    resolution = fallback
            resolutions.append(resolution)
            if resolution.resolved is None:
                if resolution.ambiguous:
                    warnings.append(RecallWarning.IDENTITY_AMBIGUOUS)
                continue
            subject = resolution.resolved.user_id
            if subject not in subjects and not self._guard.is_bot(subject):
                subjects.append(subject)
        return subjects, resolutions, warnings

    async def _resolve_via_name_facts(
        self,
        guild_id: str,
        identifier: str,
    ) -> Resolution | None:
        """Search stored name-facts for a unique self-name match."""
        import re as _re

        from discord_memory.models.identity import (
            AliasSource,
            Resolution,
            ResolvedCandidate,
        )

        hits = await self._store.search_facts_text(
            guild_id,
            identifier,
            limit=10,
        )
        matched_users: dict[str, float] = {}
        pattern = _re.compile(
            r"\b(my name(?:'s| is)|call me|goes by)\s+"
            + _re.escape(identifier.strip().lower())
            + r"\b",
            _re.IGNORECASE,
        )
        for record, score in hits:
            if record.subject_id is None or record.attribution.type.value == "third_party":
                continue
            if pattern.search(record.text.lower()):
                matched_users.setdefault(
                    record.subject_id,
                    max(score, matched_users.get(record.subject_id, 0)),
                )
        if len(matched_users) != 1:
            return None
        user_id = next(iter(matched_users))
        candidate = ResolvedCandidate(
            user_id=user_id,
            matched_alias=identifier.lower(),
            source=AliasSource.REAL_NAME,
            weight=0.6,
            confidence=round(0.6 * matched_users[user_id], 4),
        )
        return Resolution(
            identifier=identifier,
            resolved=candidate,
            candidates=(candidate,),
        )

    async def _register_author_alias(self, event: MessageEvent) -> None:
        display = event.author_display_name or event.author_username
        if display and not display.isdigit():
            await self.identity.handle_member_rename(
                event.guild_id,
                event.author_id,
                display,
            )

    async def _worker_loop(self) -> None:
        poll = self.config.workers.poll_interval_seconds
        heartbeat_every = max(
            1,
            int(self.config.workers.heartbeat_seconds / max(poll, 0.001)),
        )
        ticks = 0
        while not self.closing:
            try:
                await self._queue.release_expired_leases(self._clock.now())
                processed = await self.ops.run_pending(limit_batches=4)
                if processed == 0:
                    await asyncio.sleep(poll)
                ticks += 1
                if ticks % heartbeat_every == 0:
                    for guild_id in list(self.active_guilds):
                        await self._pipeline.flush_subject(
                            guild_id,
                            SERVER_SUBJECT_KEY,
                        )
                        if self.maintenance.due(guild_id):
                            await self.maintenance.run_guild(guild_id)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("worker loop error")
                await asyncio.sleep(poll)


def _in_memory_queue() -> IngestQueue:
    from discord_memory.adapters.in_memory.queue import InMemoryIngestQueue

    return InMemoryIngestQueue()


def _build_llm(
    config: MemoryConfig,
    meter: Meter,
    *,
    model: str | None = None,
    cache: LlmCache | None = None,
) -> ChatLLM | None:
    if not config.llm.enabled:
        return None
    from discord_memory.adapters.llm_cache import CachedLLM
    from discord_memory.adapters.llm_openai_compat import OpenAICompatLLM
    from discord_memory.adapters.meter import MeteredLLM

    llm_config = (
        config.llm if model is None else config.llm.model_copy(update={"model": model})
    )
    llm: ChatLLM = OpenAICompatLLM(llm_config)
    if cache is not None:
        llm = CachedLLM(llm, cache)
    return MeteredLLM(llm, meter)


def _build_store(config: MemoryConfig) -> MemoryStore:
    backend = config.storage.backend
    if backend == "sqlite":
        from discord_memory.adapters.sqlite.store import SqliteStore

        return SqliteStore(config.storage.url)
    if backend == "mongo":
        from discord_memory.adapters.mongo import MongoStore

        url = config.storage.url
        database = url.rsplit("/", 1)[-1].split("?")[0] or "discord_memory"
        if database in {"mongodb", "mongodb+srv"} or not database:
            database = "discord_memory"
        return MongoStore(url, database=database)
    raise ConfigError(
        f"storage backend {backend!r} requires an adapter package "
        "(e.g. pip install discord-memory[mongo]) or a store override",
    )


__all__ = ["DiscordMemory"]
