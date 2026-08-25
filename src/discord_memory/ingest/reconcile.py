"""Reconciliation stage: collision detection and conditional LLM phase-2 (§4.3).

A candidate collides when an exact-normalized duplicate exists or a scoped vector
neighbor scores above ``reconcile_collision_threshold``. Only colliding candidates
trigger the reconcile LLM call — non-colliding candidates commit as plain ADDs.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import ValidationError

from discord_memory.config import ExtractionConfig
from discord_memory.ingest.gates import normalize_text
from discord_memory.models.facts import FactRecord
from discord_memory.models.operations import (
    ProposedFact,
    ReconcileDecision,
    ReconcileOutput,
)
from discord_memory.ports.llm import ChatLLM, ChatRequest, Embedder, LlmMessage
from discord_memory.ports.store import MemoryStore
from discord_memory.ports.vectors import VectorIndex
from discord_memory.prompts import extraction as prompts

logger = logging.getLogger(__name__)

MAX_NEIGHBORS_PER_CANDIDATE = 6


@dataclass(slots=True)
class Collision:
    """Neighbors of one candidate that require reconciliation."""

    candidate: ProposedFact
    subject_id: str | None
    speaker_id: str | None
    duplicates: tuple[FactRecord, ...] = ()
    semantic_neighbors: tuple[FactRecord, ...] = ()

    @property
    def neighbors(self) -> tuple[FactRecord, ...]:
        return self.duplicates + self.semantic_neighbors


@dataclass(slots=True)
class ReconcilePlan:
    """Split of vetted candidates into direct adds and collision-requiring ones."""

    direct_adds: list[tuple[ProposedFact, str | None, str | None]] = field(
        default_factory=list,
    )
    collisions: list[Collision] = field(default_factory=list)


def _parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def parse_reconcile_output(text: str) -> ReconcileOutput | None:
    """Strict-parse a reconcile LLM response; ``None`` on any malformed output."""
    try:
        return ReconcileOutput.model_validate(_parse_json_object(text))
    except (ValueError, ValidationError):
        return None


class Reconciler:
    """Detects collisions and resolves them against existing memories."""

    def __init__(
        self,
        store: MemoryStore,
        vectors: VectorIndex | None,
        llm: ChatLLM | None,
        embedder: Embedder | None,
        config: ExtractionConfig,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._llm = llm
        self._embedder = embedder
        self._config = config

    async def build_plan(
        self,
        candidates: list[tuple[ProposedFact, str | None, str | None]],
        *,
        guild_id: str,
        batch_subject_id: str | None,
        embeddings_by_text: dict[str, tuple[float, ...]],
    ) -> ReconcilePlan:
        """Partition candidates into direct adds vs collisions needing reconciliation."""
        plan = ReconcilePlan()
        for candidate, cand_subject, cand_speaker in candidates:
            normalized = normalize_text(candidate.text)
            duplicate = await self._store.find_duplicate(
                guild_id,
                cand_subject or batch_subject_id,
                normalized,
            )
            if duplicate is not None:
                plan.collisions.append(
                    Collision(
                        candidate=candidate,
                        subject_id=cand_subject,
                        speaker_id=cand_speaker,
                        duplicates=(duplicate,),
                    )
                )
                continue
            embedding = embeddings_by_text.get(normalized)
            if embedding is None and self._embedder is not None:
                (embedding,) = await self._embedder.embed((candidate.text,))
                embeddings_by_text[normalized] = embedding
            neighbors = await self._semantic_neighbors(
                candidate=candidate,
                guild_id=guild_id,
                cand_subject=cand_subject,
                batch_subject_id=batch_subject_id,
                embedding=embedding,
            )
            if neighbors:
                plan.collisions.append(
                    Collision(
                        candidate=candidate,
                        subject_id=cand_subject,
                        speaker_id=cand_speaker,
                        semantic_neighbors=neighbors,
                    )
                )
            else:
                plan.direct_adds.append((candidate, cand_subject, cand_speaker))
        return plan

    async def _semantic_neighbors(
        self,
        *,
        candidate: ProposedFact,
        guild_id: str,
        cand_subject: str | None,
        batch_subject_id: str | None,
        embedding: tuple[float, ...] | None,
    ) -> tuple[FactRecord, ...]:
        if self._vectors is None or embedding is None:
            return ()
        scope_ids: tuple[str, ...] | None
        if cand_subject is not None:
            scope_ids = (cand_subject,)
        elif batch_subject_id is not None:
            scope_ids = (batch_subject_id,)
        else:
            scope_ids = None
        server_only = cand_subject is None and batch_subject_id is None
        hits = await self._vectors.search(
            embedding,
            guild_id=guild_id,
            subject_ids=scope_ids,
            server_only=server_only,
            limit=MAX_NEIGHBORS_PER_CANDIDATE,
            candidate_cap=100,
        )
        threshold = self._config.reconcile_collision_threshold
        strong = [hit for hit in hits if hit.score >= threshold]
        if not strong:
            return ()
        records = await self._store.get_facts(guild_id, tuple(hit.id for hit in strong))
        active = [r for r in records if r.is_active]
        return tuple(active)

    async def resolve_collisions(
        self,
        collisions: list[Collision],
    ) -> dict[int, tuple[ReconcileDecision, ...]]:
        """Run the phase-2 LLM per collision; returns decisions keyed by collision index.

        Malformed responses degrade safely to "add anyway" — callers treat a missing
        entry as ADD.
        """
        results: dict[int, tuple[ReconcileDecision, ...]] = {}
        for index, collision in enumerate(collisions):
            if self._llm is None or not collision.neighbors:
                continue
            neighbors_block = "\n".join(
                f"- id={record.id} :: {record.text}" for record in collision.neighbors
            )
            prompt = prompts.render_reconcile_prompt(
                candidate_text=collision.candidate.text,
                neighbors_block=neighbors_block,
            )
            response = await self._llm.complete(
                ChatRequest(
                    messages=(
                        LlmMessage(role="system", content=prompts.RECONCILE_SYSTEM_PROMPT),
                        LlmMessage(role="user", content=prompt),
                    ),
                    json_mode=True,
                    max_tokens=900,
                    purpose="reconcile",
                )
            )
            output = parse_reconcile_output(response.text)
            if output is None:
                logger.warning("Reconcile output unparseable; defaulting to ADD")
                continue
            known_ids = {record.id for record in collision.neighbors}
            safe: list[ReconcileDecision] = []
            for decision in output.decisions:
                target_known = decision.target_id is not None and decision.target_id in known_ids
                if decision.kind.value in {"noop", "add"} or target_known:
                    safe.append(decision)
                else:
                    logger.warning(
                        "Dropping %s decision with unknown target_id",
                        decision.kind.value,
                    )
            results[index] = tuple(safe)
        return results
