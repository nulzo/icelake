"""Recall orchestration: channels → RRF → hybrid rerank → hard filters (§5)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any

from icelake.config import RetrievalConfig
from icelake.identity.guards import BotGuard
from icelake.lifecycle.strength import retention, strength_signal
from icelake.models.facts import FactRecord
from icelake.models.retrieval import (
    CHANNELS_DEFAULT,
    ChannelName,
    ChannelSet,
    RecallQuery,
    RecallResult,
    RecallWarning,
    Scope,
    ScoreComponents,
    ScoredFact,
)
from icelake.ports.clock import Clock
from icelake.ports.llm import Embedder
from icelake.ports.store import MemoryStore
from icelake.ports.vectors import VectorIndex
from icelake.retrieval import channels as ch
from icelake.scoring.fusion import (
    RankedChannel,
    RerankInputs,
    RerankResult,
    hybrid_rerank,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)

_FUTURE = datetime.max.replace(tzinfo=UTC)


class RecallService:
    """Read-path orchestrator. Zero LLM calls by design; never raises."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        vectors: VectorIndex | None,
        embedder: Embedder | None,
        config: RetrievalConfig,
        guard: BotGuard | None = None,
        is_subject_blocked: Any = None,
        on_recalled: Any = None,
        clock: Clock | None = None,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self._config = config
        self._guard = guard
        self._is_subject_blocked = is_subject_blocked
        self._on_recalled = on_recalled
        self._clock = clock

    async def recall(self, query: RecallQuery) -> RecallResult:
        selected = query.channels if query.channels else CHANNELS_DEFAULT
        subject_ids: tuple[str, ...] | None = tuple(query.subject_ids) or None
        server_only = query.scope is Scope.SERVER
        if query.scope is Scope.GUILD:
            subject_ids = None  # guild-wide union: no subject restriction

        pair_fact_ids = await self._pair_intersection(query)
        entity_slug = await self._resolve_entity_hint(query)

        outputs, degraded = await self._run_channels(
            selected,
            guild_id=query.guild_id,
            text=query.text or "",
            subject_ids=subject_ids,
            server_only=server_only,
            as_of=query.as_of,
        )

        if pair_fact_ids:
            outputs.append(
                ch.ChannelOutput(
                    channel=ChannelName.LINKS,
                    ranked_ids=tuple(pair_fact_ids),
                ),
            )
        if entity_slug is not None:
            outputs.append(await self._entity_hint_channel(query, entity_slug))

        ranked_channels = [
            RankedChannel(channel=out.channel, ranked_ids=out.ranked_ids)
            for out in outputs
            if out.ranked_ids
        ]
        fused = reciprocal_rank_fusion(
            ranked_channels,
            k=self._config.rrf_k,
            pool_size=self._config.rerank_pool_size,
        )
        # One fetch for the fused pool: feeds the strength component and is
        # reused by _materialize instead of a second get_facts round-trip.
        pool_records = await self._store.get_facts(query.guild_id, tuple(c.fact_id for c in fused))
        semantic_map: dict[str, float] = {}
        lexical_map: dict[str, float] = {}
        entity_map: dict[str, float] = {}
        for out in outputs:
            semantic_map.update(out.semantic)
            lexical_map.update(out.lexical)
            entity_map.update(out.entity)

        scored = hybrid_rerank(
            fused,
            RerankInputs(
                semantic=semantic_map,
                lexical=lexical_map,
                entity=entity_map,
                strength=self._strength_map(pool_records),
            ),
            weight_semantic=self._config.weight_semantic,
            weight_lexical=self._config.weight_lexical,
            weight_entity=self._config.weight_entity,
            weight_strength=self._config.weight_strength,
        )
        return await self._materialize(query, scored, degraded, records=pool_records)

    def _strength_map(self, records: tuple[FactRecord, ...]) -> dict[str, float]:
        """Recency-aware strength: Ebbinghaus retention times log-scaled strength."""
        if self._clock is None:
            return {}
        now = self._clock.now()
        return {
            record.id: strength_signal(
                strength=record.strength,
                retention_value=retention(
                    last_reinforced_at=record.last_reinforced_at or record.created_at or now,
                    now=now,
                    strength=record.strength,
                ),
            )
            for record in records
        }

    async def _run_channels(
        self,
        selected: ChannelSet,
        *,
        guild_id: str,
        text: str,
        subject_ids: tuple[str, ...] | None,
        server_only: bool,
        as_of: datetime | None = None,
    ) -> tuple[list[ch.ChannelOutput], list[str]]:
        tasks: dict[ChannelName, object] = {}

        if ChannelName.VECTOR in selected:
            tasks[ChannelName.VECTOR] = ch.vector_channel(
                vectors=self._vectors,
                embedder=self._embedder,
                guild_id=guild_id,
                query_text=text,
                subject_ids=None if server_only else subject_ids,
                server_only=server_only,
                limit=self._config.recall_limit,
                candidate_cap=self._config.candidate_cap,
            )
        if ChannelName.KEYWORD in selected:
            tasks[ChannelName.KEYWORD] = ch.keyword_channel(
                store=self._store,
                guild_id=guild_id,
                query_text=text,
                subject_ids=None if server_only else subject_ids,
                server_only=server_only,
                limit=self._config.recall_limit,
                as_of=as_of,
            )
        if ChannelName.LINKS in selected and subject_ids:
            tasks[ChannelName.LINKS] = ch.links_channel(
                store=self._store,
                guild_id=guild_id,
                subject_ids=subject_ids,
                limit=self._config.recall_limit,
            )
        if ChannelName.BASELINE in selected:
            tasks[ChannelName.BASELINE] = ch.baseline_channel(
                store=self._store,
                guild_id=guild_id,
                subject_ids=subject_ids,
                server_only=server_only,
                limit=self._config.max_per_subject * max(1, len(subject_ids or (1,))),
            )
        if ChannelName.GRAPH_HOP in selected and subject_ids:
            tasks[ChannelName.GRAPH_HOP] = ch.graph_hop_channel(
                store=self._store,
                guild_id=guild_id,
                subject_ids=subject_ids,
                depth=self._config.hop_depth,
                fan_out_per_node=self._config.fan_out_per_node,
                limit=self._config.recall_limit,
            )
        if ChannelName.ENTITY in selected:
            tasks[ChannelName.ENTITY] = ch.entity_channel(
                store=self._store,
                guild_id=guild_id,
                query_text=text,
                limit=self._config.recall_limit,
            )

        keys = list(tasks)
        coros: list[Coroutine[Any, Any, ch.ChannelOutput]] = [
            tasks[key]  # type: ignore[misc]
            for key in keys
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        outputs: list[ch.ChannelOutput] = []
        degraded: list[str] = []
        for key, result in zip(keys, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("channel %s raised", key, exc_info=result)
                degraded.append(key.value)
                continue
            outputs.append(result)
        return outputs, degraded

    async def _pair_intersection(self, query: RecallQuery) -> list[str]:
        """Facts joined to BOTH users of each ``pair_ids`` pair (Q3/Q2 link-intersect).

        All pairs resolve concurrently; each intersection is two indexed
        ``links_for_node`` lookups, so cost stays O(pairs), not O(facts).
        """
        if not query.pair_ids:
            return []
        from icelake.models.graph import NodeType

        async def linked_ids(user_id: str) -> set[str]:
            return {
                record.id
                for _row, record in await self._store.links_for_node(
                    query.guild_id,
                    NodeType.USER,
                    user_id,
                    active_only=True,
                    limit=self._config.recall_limit,
                )
            }

        async def intersect(a: str, b: str) -> list[str]:
            by_a, by_b = await asyncio.gather(linked_ids(a), linked_ids(b))
            return [fact_id for fact_id in by_a if fact_id in by_b]

        intersections = await asyncio.gather(
            *(intersect(a, b) for a, b in query.pair_ids if a != b)
        )
        ranked: list[str] = []
        seen: set[str] = set()
        for fact_ids in intersections:
            for fact_id in fact_ids:
                if fact_id not in seen:
                    seen.add(fact_id)
                    ranked.append(fact_id)
        return ranked

    async def _resolve_entity_hint(self, query: RecallQuery) -> str | None:
        """Resolve ``entity_hint`` surface name to a canonical slug (Q5)."""
        from icelake.identity.aliases import alias_slug, normalize_alias

        if not query.entity_hint:
            return None
        normalized = normalize_alias(query.entity_hint)
        slug = await self._store.resolve_entity_alias(query.guild_id, normalized)
        return slug or alias_slug(query.entity_hint)

    async def _entity_hint_channel(
        self,
        query: RecallQuery,
        slug: str,
    ) -> ch.ChannelOutput:
        from icelake.models.graph import NodeType

        linked = await self._store.links_for_node(
            query.guild_id,
            NodeType.ENTITY,
            slug,
            active_only=True,
            limit=self._config.recall_limit,
        )
        return ch.ChannelOutput(
            channel=ChannelName.ENTITY,
            ranked_ids=tuple(record.id for _row, record in linked),
            entity={record.id: 1.0 for _row, record in linked},
        )

    async def _materialize(
        self,
        query: RecallQuery,
        scored: list[RerankResult],
        degraded: list[str],
        *,
        records: tuple[FactRecord, ...],
    ) -> RecallResult:
        fact_ids = [fact_id for fact_id, score, _, _ in scored]
        by_id = {record.id: record for record in records}
        scores = {fact_id: score for fact_id, score, _, _ in scored}
        components_map = {fact_id: comps for fact_id, _, comps, _ in scored}
        channels_map = {fact_id: chans for fact_id, _, _, chans in scored}

        per_subject: dict[str, int] = {}
        facts: list[ScoredFact] = []
        warnings: list[str] = []
        trimmed = False
        for fact_id in fact_ids:
            record = by_id.get(fact_id)
            if record is None:
                continue
            if query.as_of is not None:
                # Time travel: validity window decides; supersession flags are
                # irrelevant because they describe LATER knowledge.
                start = record.valid_from
                end = record.valid_until or _FUTURE
                if start is not None and start > query.as_of:
                    continue
                if end <= query.as_of:
                    continue
            elif not record.is_active:
                continue
            if scores[fact_id] < query.min_score:
                continue
            subject = record.subject_id
            guarded = (
                self._guard is not None and subject is not None and (self._guard.is_bot(subject))
            )
            if guarded:
                continue
            if (
                self._is_subject_blocked is not None
                and subject is not None
                and await self._is_subject_blocked(query.guild_id, subject)
            ):
                continue
            if record.subject_id and record.subject_id in query.exclude_ids:
                continue
            cap_key = record.subject_id or "__server__"
            if per_subject.get(cap_key, 0) >= query.max_per_subject:
                continue
            if len(facts) >= query.top_k:
                trimmed = True
                break
            per_subject[cap_key] = per_subject.get(cap_key, 0) + 1
            comps = components_map.get(fact_id, (0.0, 0.0, 0.0, 0.0))
            facts.append(
                ScoredFact(
                    fact=record,
                    score=scores[fact_id],
                    components=ScoreComponents(
                        semantic=comps[0],
                        lexical=comps[1],
                        entity=comps[2],
                        strength=comps[3],
                    ),
                    matched_channels=channels_map.get(fact_id, ()),
                )
            )
        if trimmed:
            warnings.append("budget_trimmed")
        if self._on_recalled is not None and facts:
            try:
                await self._on_recalled([f.fact.id for f in facts])
            except Exception:
                logger.warning("on_recalled callback failed", exc_info=True)
        return RecallResult(
            facts=tuple(facts),
            degraded_channels=tuple(degraded),
            warnings=_to_warnings(warnings),
        )


def _to_warnings(names: list[str]) -> tuple[RecallWarning, ...]:
    from icelake.models.retrieval import RecallWarning

    mapped: list[RecallWarning] = []
    for name in names:
        try:
            mapped.append(RecallWarning(name))
        except ValueError:
            continue
    return tuple(mapped)


__all__ = ["RecallService"]
