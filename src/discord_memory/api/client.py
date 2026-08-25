"""DiscordMemory: the consumer facade (API.md Parts 2-3).

Composition root — builds adapters from config, applies port overrides, owns the
worker lifecycle, and exposes the namespaced capability groups. All heavy lifting
lives in focused services; this class only wires and delegates.
"""

from __future__ import annotations

import asyncio
import logging

from discord_memory.adapters.embedders import build_embedder
from discord_memory.api.classify import CommandClassifier, UserMemoryCommand
from discord_memory.api.events import EventBus
from discord_memory.api.facts_api import FactsApi
from discord_memory.api.groups import AdminApi, GraphApi, IdentityApi
from discord_memory.config import MemoryConfig
from discord_memory.consolidation.service import ConsolidationService
from discord_memory.errors import ConfigError
from discord_memory.identity.guards import BotGuard, ConsentPolicy, SubjectGate
from discord_memory.ingest.pipeline import SERVER_SUBJECT_KEY, IngestPipeline
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
)
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
from discord_memory.ports.llm import ChatLLM, Embedder, Meter
from discord_memory.ports.queue import IngestQueue, StoredMessage
from discord_memory.ports.store import MemoryStore
from discord_memory.ports.vectors import VectorIndex
from discord_memory.retrieval.injection import InjectionBuilder, estimate_tokens
from discord_memory.retrieval.service import RecallService

logger = logging.getLogger(__name__)

_MISSING = object()
_SERVER_BLOCK_FACTS = 3


class _OpsApi:
    """Worker control plane — the ``memory.ops`` namespace."""

    def __init__(self, client: DiscordMemory) -> None:
        self._client = client

    async def run_pending(self, *, limit_batches: int = 8) -> int:
        """Process due batches now (cron/external-scheduler mode)."""
        return await self._client._pipeline.run_pending(limit_batches=limit_batches)

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

        llm_override = overrides.get("llm", _MISSING)
        self._llm: ChatLLM | None = (
            _build_llm(config) if llm_override is _MISSING else llm_override  # type: ignore[assignment]
        )

        meter_override = overrides.get("meter", _MISSING)
        if meter_override is _MISSING:
            from discord_memory.adapters.meter import InMemoryMeter

            self._meter: Meter = InMemoryMeter(config.budgets, self._clock)
        else:
            self._meter = meter_override  # type: ignore[assignment]

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
            meter=self._meter,
            subject_gate=self._subject_gate,
        )
        self._pipeline.attach_event_bus(self.events)

        self.facts = FactsApi(
            store=self._store,
            vectors=self._vectors,
            embedder=self._embedder,
            clock=self._clock,
            id_gen=self._id_gen,
            config=config,
            subject_gate=self._subject_gate,
        )
        self.identity = IdentityApi(self._store)
        self.graph = GraphApi(store=self._store)
        self.admin = AdminApi(self._store)
        self.ops = _OpsApi(self)
        self._classifier = CommandClassifier(self._llm)
        self._injection = InjectionBuilder()
        self._consolidation = ConsolidationService(
            store=self._store,
            llm=self._llm,
            embedder=self._embedder,
            config=config,
        )

        self.started = False
        self.closing = False
        self.worker_tasks: list[asyncio.Task[None]] = []
        self.active_guilds: set[str] = set()

    # -- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        """Open storage and launch workers (idempotent)."""
        if self.started:
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
        self._guard.note_author(event.author_id, is_bot=event.author_is_bot)
        if event.author_is_bot or self._guard.is_bot(event.author_id):
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.IGNORED,
                reason=IgnoreReason.BOT_AUTHOR,
            )
        if await self._subject_gate.allows(event.guild_id, event.author_id) is False:
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.IGNORED,
                reason=IgnoreReason.OPTED_OUT,
            )
        if not event.content.strip():
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.IGNORED,
                reason=IgnoreReason.EMPTY_CONTENT,
            )

        message = StoredMessage(
            message_id=event.message_id,
            guild_id=event.guild_id,
            channel_id=event.channel_id,
            author_id=event.author_id,
            subject_key=event.author_id,
            content=event.content,
            created_at=event.created_at,
            author_username=event.author_username,
            author_display_name=event.author_display_name,
            author_is_bot=event.author_is_bot,
            mention_ids=event.mention_ids,
        )
        accepted = await self._queue.put_message(message)
        if not accepted:
            return ObserveReceipt(
                message_id=event.message_id,
                status=ObserveStatus.IGNORED,
                reason=IgnoreReason.DUPLICATE,
            )
        self.active_guilds.add(event.guild_id)
        await self._register_author_alias(event)
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
        service = RecallService(
            store=self._store,
            vectors=self._vectors,
            embedder=self._embedder,
            config=self.config.retrieval,
            guard=self._guard,
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

        sections["server"] = tuple(server_result.facts)
        server_doc = await self._store.get_summary(guild_id, None)
        summaries["server"] = server_doc.text if server_doc else None

        block, citations, trimmed = self._injection.build(
            asker_id=asker_id,
            facts_by_section=sections,
            summaries=summaries,
            token_budget=budget,
            guild_id=guild_id,
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

    # -- misc surface -------------------------------------------------------------

    async def stats(self, guild_id: str) -> GuildStats:
        """Guild memory statistics snapshot."""
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
            resolutions.append(resolution)
            if resolution.resolved is None:
                if resolution.ambiguous:
                    warnings.append(RecallWarning.IDENTITY_AMBIGUOUS)
                continue
            subject = resolution.resolved.user_id
            if subject not in subjects and not self._guard.is_bot(subject):
                subjects.append(subject)
        return subjects, resolutions, warnings

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
                        await self._maybe_server_batch(guild_id)
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("worker loop error")
                await asyncio.sleep(poll)

    async def _maybe_server_batch(self, guild_id: str) -> None:
        await self._pipeline.flush_subject(guild_id, SERVER_SUBJECT_KEY)


def _in_memory_queue() -> IngestQueue:
    from discord_memory.adapters.in_memory.queue import InMemoryIngestQueue

    return InMemoryIngestQueue()


def _build_llm(config: MemoryConfig) -> ChatLLM | None:
    if not config.llm.enabled:
        return None
    from discord_memory.adapters.llm_openai_compat import OpenAICompatLLM

    return OpenAICompatLLM(config.llm)


def _build_store(config: MemoryConfig) -> MemoryStore:
    backend = config.storage.backend
    if backend == "sqlite":
        from discord_memory.adapters.sqlite.store import SqliteStore

        return SqliteStore(config.storage.url)
    raise ConfigError(
        f"storage backend {backend!r} requires an adapter package "
        "(e.g. pip install discord-memory[postgres]) or a store override",
    )


__all__ = ["DiscordMemory"]
