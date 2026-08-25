"""Quality-gate contract tests — the accuracy armor (PLAN.md §3.3)."""

from __future__ import annotations

import pytest

from discord_memory.ingest.gates import (
    batch_worth_extracting,
    confidence_gate,
    ephemeral_share_gate,
    normalize_text,
    plagiarism_gate,
    text_hygiene_gate,
)


class TestTextHygiene:
    def test_clean_fact_passes(self) -> None:
        assert text_hygiene_gate("alice prefers mechanical keyboards").allowed

    def test_empty_rejected(self) -> None:
        assert not text_hygiene_gate("   ").allowed

    def test_too_long_rejected(self) -> None:
        assert not text_hygiene_gate("x" * 401).allowed

    def test_too_short_rejected(self) -> None:
        assert not text_hygiene_gate("ok lol").allowed

    def test_refusal_rejected(self) -> None:
        assert not text_hygiene_gate("I'm sorry, I cannot help with that").allowed

    def test_llm_meta_rejected(self) -> None:
        text = "operations: empty operations because chain of thought is unavailable"
        assert not text_hygiene_gate(text).allowed

    def test_snowflake_rejected(self) -> None:
        assert not text_hygiene_gate("user 123456789012345678 likes tea").allowed

    def test_mention_tag_rejected(self) -> None:
        assert not text_hygiene_gate("bob mentioned <@1234567890123456> today").allowed

    @pytest.mark.parametrize(
        "text",
        ["what does bob like?", "Who is alice", "is this true"],
    )
    def test_questions_rejected(self, text: str) -> None:
        assert not text_hygiene_gate(text).allowed

    def test_rant_denial_rejected(self) -> None:
        assert not text_hygiene_gate("no no no that's wrong and i don't agree").allowed


class TestEphemeralShare:
    def test_bare_link_rejected(self) -> None:
        assert not ephemeral_share_gate("https://example.com/thing").allowed

    def test_media_share_rejected(self) -> None:
        assert not ephemeral_share_gate("posted a funny gif").allowed

    def test_link_share_rejected(self) -> None:
        assert not ephemeral_share_gate("shared https://youtube.com/x").allowed

    def test_meaningful_link_context_passes(self) -> None:
        assert ephemeral_share_gate(
            "recommends the rust programming language book from no starch press",
        ).allowed


class TestConfidenceGate:
    def test_above_floor_passes(self) -> None:
        assert confidence_gate(0.9, 0.55, "alice lives in berlin").allowed

    def test_below_floor_rejected(self) -> None:
        assert not confidence_gate(0.3, 0.55, "alice lives in berlin").allowed

    def test_ephemeral_claim_needs_high_confidence_and_durable(self) -> None:
        weak = "heard that alice might work somewhere"
        assert not confidence_gate(0.6, 0.55, weak).allowed
        durable = "heard that alice works at acme as an engineer"
        assert confidence_gate(0.8, 0.55, durable).allowed


class TestPlagiarism:
    def test_exact_copy_rejected(self) -> None:
        source = "i really love playing valorant every evening"
        assert not plagiarism_gate(source, (source,)).allowed

    def test_near_copy_rejected(self) -> None:
        source = "i have been learning the rust programming language for a year"
        candidate = "i've been learning the rust programming language for a year"
        assert not plagiarism_gate(candidate, (source,)).allowed

    def test_synthesized_fact_passes(self) -> None:
        source = "yeah i've been grinding rust for like a year now honestly"
        fact = "user has been programming in Rust for approximately one year"
        assert plagiarism_gate(fact, (source,)).allowed


class TestBatchGate:
    def test_noise_batch_skipped(self) -> None:
        assert not batch_worth_extracting(("lol", "ok", "haha nice"))

    def test_durable_marker_triggers_extraction(self) -> None:
        assert batch_worth_extracting(("my name is klim btw",))

    def test_long_enough_content_triggers(self) -> None:
        long_text = "we should definitely schedule the next campaign session for friday night"
        assert batch_worth_extracting((long_text,))


def test_normalize_text() -> None:
    assert normalize_text("  Hello   WORLD \n") == "hello world"
