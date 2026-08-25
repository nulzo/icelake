"""Recall orchestration: channels → RRF → hybrid rerank → hard filters (§5)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

from discord_memory.config import RetrievalConfig
from discord_memory.identity.guards import BotGuard
from discord_memory.models.retrieval import (
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
from discord_memory.ports.llm import Embedder
from discord_memory.ports.store import MemoryStore
from discord_memory.ports.vectors import VectorIndex
from discord_memory.retrieval import channels as ch
from discord_memory.scoring.fusion import (
    RankedChannel,
    RerankInputs,
    RerankResult,
    hybrid_rerank,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._embedder = embedder
        self._config = config
        self._guard = guard

    async def recall(self, query: RecallQuery) -> RecallResult:
        selected = query.channels if query.channels else CHANNELS_DEFAULT
        subject_ids: tuple[str, ...] | None = tuple(query.subject_ids) or None
        server_only = query.scope is Scope.SERVER
        if query.scope is Scope.SUBJECTS and subject_ids:
            pass
        elif query.scope is not Scope.SERVER:
            subject_ids = None

        outputs, degraded = await self._run_channels(
            selected,
            guild_id=query.guild_id,
            text=query.text or "",
            subject_ids=subject_ids,
            server_only=server_only,
        )

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
        semantic_map: dict[str, float] = {}
        lexical_map: dict[str, float] = {}
        for out in outputs:
            semantic_map.update(out.semantic)
            lexical_map.update(out.lexical)

        scored = hybrid_rerank(
            fused,
            RerankInputs(semantic=semantic_map, lexical=lexical_map),
            weight_semantic=self._config.weight_semantic,
            weight_lexical=self._config.weight_lexical,
            weight_entity=self._config.weight_entity,
            weight_strength=self._config.weight_strength,
        )
        return await self._materialize(query, scored, degraded)

    async def _run_channels(
        self,
        selected: ChannelSet,
        *,
        guild_id: str,
        text: str,
        subject_ids: tuple[str, ...] | None,
        server_only: bool,
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

    async def _materialize(
        self,
        query: RecallQuery,
        scored: list[RerankResult],
        degraded: list[str],
    ) -> RecallResult:
        fact_ids = [fact_id for fact_id, score, _, _ in scored]
        records = await self._store.get_facts(query.guild_id, tuple(fact_ids))
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
            if record is None or not record.is_active:
                continue
            if scores[fact_id] < query.min_score:
                continue
            subject = record.subject_id
            guarded = (
                self._guard is not None and subject is not None and (self._guard.is_bot(subject))
            )
            if guarded:
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
        return RecallResult(
            facts=tuple(facts),
            degraded_channels=tuple(degraded),
            warnings=_to_warnings(warnings),
        )


def _to_warnings(names: list[str]) -> tuple[RecallWarning, ...]:
    from discord_memory.models.retrieval import RecallWarning

    mapped: list[RecallWarning] = []
    for name in names:
        try:
            mapped.append(RecallWarning(name))
        except ValueError:
            continue
    return tuple(mapped)


__all__ = ["RecallService"]
