"""Reconciler unit tests: deterministic reinforce tiers and batched LLM resolve."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from discord_memory.adapters.in_memory.store import InMemoryStore
from discord_memory.adapters.in_memory.vectors import InMemoryVectorIndex
from discord_memory.config import ExtractionConfig
from discord_memory.ingest.reconcile import Reconciler
from discord_memory.models.facts import FactRecord
from discord_memory.models.operations import ProposedFact, ReconcileKind
from discord_memory.ports.vectors import VectorItem
from tests.conftest import ScriptedLLM

GUILD = "g1"
USER = "u1"


def _fact(text: str, *, fact_id: str = "fct_1") -> FactRecord:
    now = datetime.now(UTC)
    return FactRecord(
        id=fact_id,
        guild_id=GUILD,
        subject_id=USER,
        text=text,
        text_normalized=text,
        created_at=now,
        updated_at=now,
        observed_at=now,
        valid_from=now,
    )


def _proposal(text: str) -> ProposedFact:
    return ProposedFact(subject_token="p0", text=text, confidence=0.9)


class _FixedEmbedder:
    """Maps exact texts to vectors; unknown texts get a zero vector."""

    def __init__(self, vectors: dict[str, tuple[float, ...]]) -> None:
        self._vectors = vectors

    async def embed(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._vectors.get(t, (0.0, 0.0)) for t in texts)


def _reconciler(
    store: InMemoryStore,
    vectors: InMemoryVectorIndex,
    llm: ScriptedLLM | None,
    embedder: _FixedEmbedder | None,
) -> Reconciler:
    return Reconciler(store, vectors, llm, embedder, ExtractionConfig())


async def _seed(
    store: InMemoryStore,
    vectors: InMemoryVectorIndex,
    fact: FactRecord,
    embedding: tuple[float, ...],
) -> None:
    await store.insert_fact(fact)
    await vectors.upsert(
        (
            VectorItem(
                id=fact.id,
                guild_id=fact.guild_id,
                subject_id=fact.subject_id,
                embedding=embedding,
            ),
        )
    )


class TestDeterministicReinforce:
    async def test_exact_duplicate_reinforces_without_llm(self) -> None:
        store, vectors = InMemoryStore(), InMemoryVectorIndex()
        fact = _fact("likes go")
        await _seed(store, vectors, fact, (1.0, 0.0))
        llm = ScriptedLLM()
        reconciler = _reconciler(store, vectors, llm, None)

        plan = await reconciler.build_plan(
            [(_proposal("likes go"), USER, USER)],
            guild_id=GUILD,
            batch_subject_id=USER,
            embeddings_by_text={},
        )

        assert [r.id for r, *_ in plan.reinforces] == [fact.id]
        assert plan.collisions == [] and plan.direct_adds == []
        assert llm.calls == []

    async def test_near_duplicate_neighbor_reinforces_without_llm(self) -> None:
        store, vectors = InMemoryStore(), InMemoryVectorIndex()
        fact = _fact("loves go")
        await _seed(store, vectors, fact, (1.0, 0.0))
        llm = ScriptedLLM()
        embedder = _FixedEmbedder({"really loves go": (0.999, 0.01)})
        reconciler = _reconciler(store, vectors, llm, embedder)

        plan = await reconciler.build_plan(
            [(_proposal("really loves go"), USER, USER)],
            guild_id=GUILD,
            batch_subject_id=USER,
            embeddings_by_text={},
        )

        assert [r.id for r, *_ in plan.reinforces] == [fact.id]
        assert plan.collisions == []
        assert llm.calls == []

    async def test_mid_band_neighbor_becomes_collision(self) -> None:
        store, vectors = InMemoryStore(), InMemoryVectorIndex()
        fact = _fact("lives in omaha")
        await _seed(store, vectors, fact, (1.0, 0.0))
        embedder = _FixedEmbedder({"moved to seattle": (1.0, 1.0)})
        reconciler = _reconciler(store, vectors, ScriptedLLM(), embedder)

        plan = await reconciler.build_plan(
            [(_proposal("moved to seattle"), USER, USER)],
            guild_id=GUILD,
            batch_subject_id=USER,
            embeddings_by_text={},
        )

        assert plan.reinforces == [] and plan.direct_adds == []
        assert len(plan.collisions) == 1
        assert plan.collisions[0].neighbors[0].id == fact.id


class TestBatchedResolve:
    async def test_one_call_for_many_collisions_with_id_remap(self) -> None:
        store, vectors = InMemoryStore(), InMemoryVectorIndex()
        first, second = (
            _fact("lives in omaha", fact_id="fct_a"),
            _fact("works as a nurse", fact_id="fct_b"),
        )
        await _seed(store, vectors, first, (1.0, 0.0))
        await _seed(store, vectors, second, (0.0, 1.0))
        embedder = _FixedEmbedder(
            {"moved to seattle": (1.0, 1.0), "promoted at the hospital": (1.0, 1.0)}
        )
        llm = ScriptedLLM(
            {
                "reconcile": json.dumps(
                    {
                        "decisions": [
                            {
                                "candidate_index": 0,
                                "kind": "invalidate",
                                "target_id": 0,
                                "reason": "moved",
                            },
                            {
                                "candidate_index": 1,
                                "kind": "update",
                                "target_id": 1,
                                "text": "works as a senior nurse",
                                "reason": "refined",
                            },
                        ]
                    }
                )
            }
        )
        reconciler = _reconciler(store, vectors, llm, embedder)
        plan = await reconciler.build_plan(
            [
                (_proposal("moved to seattle"), USER, USER),
                (_proposal("promoted at the hospital"), USER, USER),
            ],
            guild_id=GUILD,
            batch_subject_id=USER,
            embeddings_by_text={},
        )
        assert len(plan.collisions) == 2

        decisions = await reconciler.resolve_collisions(plan.collisions)

        reconcile_calls = [c for c in llm.calls if c.purpose == "reconcile"]
        assert len(reconcile_calls) == 1
        assert decisions[0][0].kind is ReconcileKind.INVALIDATE
        assert decisions[0][0].target_id == "fct_a"
        assert decisions[1][0].kind is ReconcileKind.UPDATE
        assert decisions[1][0].target_id == "fct_b"

    async def test_unknown_target_drops_update_decision(self) -> None:
        store, vectors = InMemoryStore(), InMemoryVectorIndex()
        await _seed(store, vectors, _fact("lives in omaha"), (1.0, 0.0))
        embedder = _FixedEmbedder({"moved to seattle": (1.0, 1.0)})
        llm = ScriptedLLM(
            {
                "reconcile": json.dumps(
                    {
                        "decisions": [
                            {
                                "candidate_index": 0,
                                "kind": "update",
                                "target_id": 99,
                                "text": "x",
                            }
                        ]
                    }
                )
            }
        )
        reconciler = _reconciler(store, vectors, llm, embedder)
        plan = await reconciler.build_plan(
            [(_proposal("moved to seattle"), USER, USER)],
            guild_id=GUILD,
            batch_subject_id=USER,
            embeddings_by_text={},
        )

        decisions = await reconciler.resolve_collisions(plan.collisions)

        assert decisions == {}

    async def test_noop_without_target_survives(self) -> None:
        store, vectors = InMemoryStore(), InMemoryVectorIndex()
        await _seed(store, vectors, _fact("lives in omaha"), (1.0, 0.0))
        embedder = _FixedEmbedder({"still in omaha": (1.0, 1.0)})
        llm = ScriptedLLM(
            {"reconcile": json.dumps({"decisions": [{"candidate_index": 0, "kind": "noop"}]})}
        )
        reconciler = _reconciler(store, vectors, llm, embedder)
        plan = await reconciler.build_plan(
            [(_proposal("still in omaha"), USER, USER)],
            guild_id=GUILD,
            batch_subject_id=USER,
            embeddings_by_text={},
        )

        decisions = await reconciler.resolve_collisions(plan.collisions)

        assert decisions[0][0].kind is ReconcileKind.NOOP
        assert decisions[0][0].target_id is None


@pytest.mark.parametrize("kind", ["update", "invalidate"])
def test_wire_schema_accepts_integer_target_ids(kind: str) -> None:
    from discord_memory.models.operations import ReconcileDecision

    decision = ReconcileDecision.model_validate(
        {"kind": kind, "target_id": 7, "candidate_index": 0}
    )
    assert decision.target_id == "7"
