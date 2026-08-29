"""Reconciliation stage: collision detection and conditional LLM phase-2 (§4.3).

A candidate collides when an exact-normalized duplicate exists or a scoped vector
neighbor scores above ``reconcile_collision_threshold``. Only colliding candidates
trigger the reconcile LLM call — non-colliding candidates commit as plain ADDs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from icelake.config import ExtractionConfig
from icelake.ingest.gates import normalize_text
from icelake.models.facts import FactRecord
from icelake.models.operations import (
    ProposedFact,
    ReconcileDecision,
    ReconcileKind,
    ReconcileOutput,
)
from icelake.ports.llm import ChatLLM, Embedder, LlmMessage
from icelake.ports.store import MemoryStore
from icelake.ports.vectors import VectorIndex, cosine
from icelake.prompts import extraction as prompts
from icelake.structured import complete_structured

logger = logging.getLogger(__name__)

MAX_NEIGHBORS_PER_CANDIDATE = 6

# Conflicts are lexically dissimilar ("moved to Seattle" vs "lives in Omaha"), so
# pure cosine misses them; category is the conflict scope. Same-category neighbors
# above this floor join the collision set and let the reconcile LLM arbitrate.
SAME_CATEGORY_COLLISION_FLOOR = 0.35

# Negation / life-event phrasing signals a possible contradiction. Model-assigned
# categories are too inconsistent to gate those pairs ("loves Red Bull" landed in
# `preferences`, "quit drinking Red Bull" in `general` — they never collided), so
# state-change candidates face reconcile against any neighbor above the floor.
STATE_CHANGE = re.compile(
    r"\b(quit|stopped|no longer|not anymore|anymore|never|"
    r"(?:do|does|did)(?:n'?t| not)\b|"
    r"moved|promoted|hired|fired|graduated|switched|left|broke up|divorced|retired|sold)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class Collision:
    """Semantic neighbors of one candidate that require LLM reconciliation."""

    candidate: ProposedFact
    subject_id: str | None
    speaker_id: str | None
    semantic_neighbors: tuple[FactRecord, ...] = ()

    @property
    def neighbors(self) -> tuple[FactRecord, ...]:
        return self.semantic_neighbors


@dataclass(slots=True)
class ReconcilePlan:
    """Split of vetted candidates into reinforces, direct adds, and collisions."""

    reinforces: list[tuple[FactRecord, ProposedFact, str | None, str | None]] = field(
        default_factory=list,
    )
    direct_adds: list[tuple[ProposedFact, str | None, str | None]] = field(
        default_factory=list,
    )
    collisions: list[Collision] = field(default_factory=list)


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
        """Partition candidates into reinforces, direct adds, and LLM collisions.

        Reinforces are deterministic: exact-normalized duplicates and neighbors at
        or above ``near_duplicate_threshold`` need no LLM judgment (graphiti's fast
        path). Only the ambiguous band below that pays for reconciliation.
        """
        plan = ReconcilePlan()
        kept: list[tuple[str | None, tuple[float, ...]]] = []
        for candidate, cand_subject, cand_speaker in candidates:
            normalized = normalize_text(candidate.text)
            vector = embeddings_by_text.get(normalized)
            if vector is not None and any(
                subject == cand_subject
                and cosine(vector, other) >= self._config.near_duplicate_threshold
                for subject, other in kept
            ):
                # Paraphrase of an earlier candidate from the same response.
                continue
            if vector is not None:
                kept.append((cand_subject, vector))
            duplicate = await self._store.find_duplicate(
                guild_id,
                cand_subject,
                normalized,
            )
            if duplicate is not None:
                plan.reinforces.append((duplicate, candidate, cand_subject, cand_speaker))
                continue
            embedding = vector
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
            if not neighbors:
                plan.direct_adds.append((candidate, cand_subject, cand_speaker))
                continue
            top_record, top_score = neighbors[0]
            if top_score >= self._config.near_duplicate_threshold:
                plan.reinforces.append((top_record, candidate, cand_subject, cand_speaker))
                continue
            plan.collisions.append(
                Collision(
                    candidate=candidate,
                    subject_id=cand_subject,
                    speaker_id=cand_speaker,
                    semantic_neighbors=tuple(record for record, _ in neighbors),
                )
            )
        return plan

    async def _semantic_neighbors(
        self,
        *,
        candidate: ProposedFact,
        guild_id: str,
        cand_subject: str | None,
        batch_subject_id: str | None,
        embedding: tuple[float, ...] | None,
    ) -> tuple[tuple[FactRecord, float], ...]:
        """Score-ordered active neighbors above the collision bars."""
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
        if not hits:
            return ()
        records = await self._store.get_facts(guild_id, tuple(hit.id for hit in hits))
        active_by_id = {record.id: record for record in records if record.is_active}
        state_change = bool(STATE_CHANGE.search(candidate.text))
        return tuple(
            (active_by_id[hit.id], hit.score)
            for hit in hits
            if hit.id in active_by_id
            and (
                hit.score >= threshold
                or (
                    hit.score >= SAME_CATEGORY_COLLISION_FLOOR
                    and (state_change or active_by_id[hit.id].category.value == candidate.category)
                )
            )
        )

    async def resolve_collisions(
        self,
        collisions: list[Collision],
        *,
        guild_id: str | None = None,
    ) -> dict[int, tuple[ReconcileDecision, ...]]:
        """One batched phase-2 LLM call for all collisions, keyed by collision index.

        Neighbor fact ids are remapped to small integers for the prompt (fewer
        tokens, no id hallucination — mem0/graphiti both do this) and mapped back
        locally. A malformed response degrades safely: callers apply the
        conservative default to any collision missing from the result.
        """
        active = [(index, c) for index, c in enumerate(collisions) if c.neighbors]
        if self._llm is None or not active:
            return {}
        id_map: dict[str, str] = {}
        blocks: list[str] = []
        for candidate_index, (_original, collision) in enumerate(active):
            lines = []
            for record in collision.neighbors:
                remap = str(len(id_map))
                id_map[remap] = record.id
                lines.append(f"- [{remap}] {record.text}")
            blocks.append(
                f"CANDIDATE {candidate_index}:\n{collision.candidate.text}\n"
                f"EXISTING MEMORIES:\n" + "\n".join(lines)
            )
        output = await complete_structured(
            self._llm,
            model=ReconcileOutput,
            messages=(
                LlmMessage(role="system", content=prompts.RECONCILE_SYSTEM_PROMPT),
                LlmMessage(
                    role="user",
                    content=prompts.render_reconcile_prompt("\n\n".join(blocks)),
                ),
            ),
            max_tokens=min(4000, 500 + 250 * len(active)),
            purpose="reconcile",
            guild_id=guild_id,
        )
        if output is None:
            return {}
        results: dict[int, list[ReconcileDecision]] = {}
        for decision in output.decisions:
            if decision.candidate_index >= len(active):
                logger.warning(
                    "Dropping decision for unknown candidate %d", decision.candidate_index
                )
                continue
            original_index, collision = active[decision.candidate_index]
            known_ids = {record.id for record in collision.neighbors}
            target = id_map.get(decision.target_id or "") or None
            if decision.kind is ReconcileKind.NOOP and target not in known_ids:
                target = None  # caller falls back to the sole/strongest neighbor
            if decision.kind.value in {"update", "invalidate"} and target not in known_ids:
                logger.warning(
                    "Dropping %s decision with unknown target_id",
                    decision.kind.value,
                )
                continue
            mapped = decision.model_copy(update={"target_id": target})
            results.setdefault(original_index, []).append(mapped)
        return {index: tuple(decisions) for index, decisions in results.items()}
