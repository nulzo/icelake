from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from icelake.graph.relations import (
    compute_edge_weight,
    merge_edge,
    polarity_for_verb,
)
from icelake.graph.traversal import (
    aggregate_stances,
    hop_neighbors,
    jaccard_similarity,
    node_key,
)
from icelake.models.graph import NodeType, Polarity, RelationEdge
from icelake.models.retrieval import ChannelName
from icelake.scoring.fusion import (
    FusedCandidate,
    RankedChannel,
    RerankInputs,
    hybrid_rerank,
    reciprocal_rank_fusion,
)


def _edge(
    src: str = "u1",
    dst_id: str = "movies",
    dst_type=NodeType.ENTITY,
    verb: str = "likes",
    weight: float = 1.0,
) -> RelationEdge:
    return RelationEdge(
        guild_id="g",
        src_type=NodeType.USER,
        src_id=src,
        dst_type=dst_type,
        dst_id=dst_id,
        verb=verb,
        polarity=polarity_for_verb(verb),
        weight=weight,
    )


class TestRRF:
    def test_multi_channel_consensus_wins(self) -> None:
        channels = [
            RankedChannel(channel=ChannelName.VECTOR, ranked_ids=("a", "b", "c")),
            RankedChannel(channel=ChannelName.KEYWORD, ranked_ids=("b", "a")),
        ]
        fused = reciprocal_rank_fusion(channels, k=60, pool_size=10)
        assert fused[0].fact_id in {"a", "b"}
        ids = [candidate.fact_id for candidate in fused]
        assert set(ids) == {"a", "b", "c"}

    def test_pool_size_respected(self) -> None:
        channels = [
            RankedChannel(channel=ChannelName.VECTOR, ranked_ids=tuple(f"f{i}" for i in range(50))),
        ]
        fused = reciprocal_rank_fusion(channels, k=60, pool_size=10)
        assert len(fused) == 10


class TestHybridRerank:
    def test_weighted_components(self) -> None:
        candidates = [FusedCandidate("f1"), FusedCandidate("f2")]
        inputs = RerankInputs(
            semantic={"f1": 0.8, "f2": 0.4},
            lexical={"f2": 0.9},
        )
        results = hybrid_rerank(
            candidates,
            inputs,
            weight_semantic=0.6,
            weight_lexical=0.4,
            weight_entity=0.0,
            weight_strength=0.0,
        )
        scores = {fact_id: score for fact_id, score, _, _ in results}
        # f1: 0.6*0.8 = .48 ; f2: (0.6*0.4 + 0.4*0.9)/1.0 = 0.60 → f2 wins
        assert scores["f2"] > scores["f1"]

    def test_missing_components_score_zero(self) -> None:
        candidates = [_mk_candidate("only")]
        inputs = RerankInputs()
        results = hybrid_rerank(
            candidates,
            inputs,
            weight_semantic=1,
            weight_lexical=1,
            weight_entity=1,
            weight_strength=1,
        )
        assert results[0][1] == 0.0

    def test_all_zero_weights_returns_empty(self) -> None:
        result = hybrid_rerank(
            [_mk_candidate("x")],
            RerankInputs(),
            weight_semantic=0,
            weight_lexical=0,
            weight_entity=0,
            weight_strength=0,
        )
        assert result == []


def _mk_candidate(fact_id: str) -> FusedCandidate:
    return FusedCandidate(fact_id=fact_id)


class TestPolarityAndWeight:
    def test_verb_polarities(self) -> None:
        assert polarity_for_verb("likes") is Polarity.POSITIVE
        assert polarity_for_verb("dislikes") is Polarity.NEGATIVE
        assert polarity_for_verb("called_out") is Polarity.NEGATIVE
        assert polarity_for_verb("knows") is Polarity.NEUTRAL
        assert polarity_for_verb("Called Out") is Polarity.NEGATIVE

    def test_edge_weight_decays_with_age(self) -> None:
        now = datetime(2026, 8, 24, tzinfo=UTC)
        fresh = compute_edge_weight(occurrences=3, confidence=0.9, last_reinforced_at=now, now=now)
        stale = compute_edge_weight(
            occurrences=3,
            confidence=0.9,
            last_reinforced_at=now - timedelta(days=365),
            now=now,
        )
        assert fresh > stale > 0

    def test_merge_edge_accumulates(self) -> None:
        now = datetime.now(UTC)
        base = _edge(weight=1.0)
        merged = merge_edge(base, _edge(), now=now)
        assert merged.occurrences == base.occurrences + 1
        assert merged.weight >= base.weight


class TestHopTraversal:
    def adjacency(self):
        u1 = node_key("user", "u1")
        movies = node_key("entity", "movies")
        u2 = node_key("user", "u2")
        return {
            u1: [_edge(dst_id="movies"), _edge(dst_id="games", verb="dislikes")],
            movies: [_edge(src="movies", dst_id="u1", dst_type=NodeType.USER)],
            node_key("entity", "games"): [],
            u2: [_edge(dst_id="games")],
        }, {u1, movies}

    def test_one_hop_discovers_entities(self) -> None:
        adjacency, seeds = self.adjacency()
        del seeds
        neighbors = hop_neighbors(node_key("user", "u1"), adjacency, depth=2, fan_out_per_node=10)
        ids = {(n.node_type.value, n.node_id) for n in neighbors}
        assert ("entity", "movies") in ids
        assert ("entity", "games") in ids

    def test_paths_recorded(self) -> None:
        adjacency, _ = self.adjacency()
        neighbors = hop_neighbors(node_key("user", "u1"), adjacency, depth=2, fan_out_per_node=10)
        movies = next(n for n in neighbors if n.node_id == "movies")
        assert "likes" in movies.relation_path


class TestStancesAndSimilarity:
    def test_aggregate_stances_groups_polarity(self) -> None:
        edges = (
            _edge(src="u1", verb="likes"),
            _edge(src="u2", verb="dislikes"),
            _edge(src="u3", verb="knows"),
        )
        summary = aggregate_stances("movies", "Movies", edges)
        assert len(summary.positive) == 1
        assert len(summary.negative) == 1
        assert len(summary.other) == 1
        assert summary.total_evidence == 3

    def test_jaccard(self) -> None:
        a = frozenset({"movies", "rust"})
        b = frozenset({"movies"})
        c = frozenset()
        assert jaccard_similarity(a, b) == pytest.approx(1 / 2)
        assert jaccard_similarity(a, c) == 0.0
