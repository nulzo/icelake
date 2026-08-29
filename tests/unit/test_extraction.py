"""Extraction stage unit tests: parsing, vetting, roster verification (§3.1/§3.3)."""

from __future__ import annotations

import json

import pytest

from icelake.config import ExtractionConfig
from icelake.errors import StructuredOutputError
from icelake.ingest.extraction import FactExtractor
from icelake.ingest.roster import Roster
from icelake.models.facts import FactCategory
from icelake.models.graph import EntityKind
from icelake.models.operations import ExtractionOutput, ProposedEntity, ProposedFact
from tests.conftest import ScriptedLLM


@pytest.fixture()
def roster() -> Roster:
    roster = Roster()
    roster.add("u-alice", "alice")
    roster.add("u-bob", "bob")
    return roster


def _extractor(responses: dict) -> FactExtractor:
    llm = ScriptedLLM(responses)
    return FactExtractor(llm, ExtractionConfig()), llm


class TestParsing:
    def test_clean_json(self) -> None:
        output = ExtractionOutput.model_validate(
            {"operations": [{"subject_token": "p0", "text": "t"}]},
        )
        assert output.operations[0].category is FactCategory.GENERAL

    def test_category_coercion_unknown_falls_back(self) -> None:
        proposal = ProposedFact(subject_token="p0", text="x", category="weird")
        assert proposal.category is FactCategory.GENERAL

    def test_category_is_typed_enum(self) -> None:
        fact = ProposedFact(subject_token="p0", text="x y z w", category="goals")
        assert fact.category is FactCategory.GOALS

    def test_entity_kind_coercion(self) -> None:
        entity = ProposedEntity(name="Paris", kind="city")
        assert entity.kind is EntityKind.CONCEPT


class TestExtractVetting:
    async def test_valid_fact_is_vetted(self, roster: Roster) -> None:
        payload = {
            "operations": [
                {
                    "subject_token": "p0",
                    "text": "alice prefers mechanical keyboards for coding",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                }
            ]
        }
        extractor, _ = _extractor({"extraction": json.dumps(payload)})
        result = await extractor.extract(
            roster=roster,
            messages=(("alice", "honestly after weeks of use this keyboard is incredible"),),
            existing_memories_block="",
        )
        assert len(result.vetted) == 1
        assert result.vetted[0].subject_id == "u-alice"

    async def test_third_party_speaker_recorded(self, roster: Roster) -> None:
        payload = {
            "operations": [
                {
                    "subject_token": "p1",
                    "speaker_token": "p0",
                    "text": "bob enjoys hiking in the mountains nearby",
                    "confidence": 0.85,
                    "source_message_indexes": [1],
                }
            ]
        }
        extractor, _ = _extractor({"extraction": json.dumps(payload)})
        result = await extractor.extract(
            roster=roster,
            messages=(("alice", "we went up the trail last weekend and bob was so fast"),),
            existing_memories_block="",
        )
        assert result.vetted[0].subject_id == "u-bob"
        assert result.vetted[0].speaker_id == "u-alice"

    async def test_server_token_maps_to_none_subject(self, roster: Roster) -> None:
        payload = {
            "operations": [
                {
                    "subject_token": "server",
                    "text": "the community loves strategy game tournaments",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                }
            ]
        }
        extractor, _ = _extractor({"extraction": json.dumps(payload)})
        result = await extractor.extract(
            roster=roster,
            messages=(("alice", "tournament season is starting again soon"),),
            existing_memories_block="",
        )
        assert result.vetted[0].subject_id is None

    @pytest.mark.parametrize("bad_payload", ["not json at all", '[{"subject_token": "p0"}]'])
    async def test_unparseable_responses_raise(
        self,
        roster: Roster,
        bad_payload: str,
    ) -> None:
        extractor, _ = _extractor({"extraction": bad_payload})
        with pytest.raises(StructuredOutputError):
            await extractor.extract(
                roster=roster,
                messages=(("alice", "some substantive message content here"),),
                existing_memories_block="",
            )

    async def test_non_list_operations_is_empty_not_failure(self, roster: Roster) -> None:
        extractor, _ = _extractor({"extraction": '{"operations": "not-a-list"}'})
        result = await extractor.extract(
            roster=roster,
            messages=(("alice", "some substantive message content here"),),
            existing_memories_block="",
        )
        assert not result.has_candidates

    async def test_system_prompt_includes_extraction_instructions(self, roster: Roster) -> None:
        payload = {
            "operations": [
                {
                    "subject_token": "p0",
                    "text": "alice enjoys weekend trail runs",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                }
            ]
        }
        extractor, llm = _extractor({"extraction": json.dumps(payload)})
        await extractor.extract(
            roster=roster,
            messages=(("alice", "i ran ten miles on the trail this weekend"),),
            existing_memories_block="",
        )
        from icelake.prompts.extraction import EXTRACTION_INSTRUCTIONS

        system = llm.calls[0].messages[0]
        assert system.role == "system"
        assert "ALWAYS emit a fact when NEW MESSAGES restate" in system.content
        assert EXTRACTION_INSTRUCTIONS in system.content
        # Shared events/plans are community-wide even when one person announces
        # them — the community pass drops user-anchored candidates, so narrowing
        # this rule silently loses server-scope facts.
        assert 'subject_token="server" for community-wide facts' in EXTRACTION_INSTRUCTIONS
        assert "shared" in EXTRACTION_INSTRUCTIONS

    async def test_max_tokens_comes_from_constructor(self, roster: Roster) -> None:
        payload = {
            "operations": [
                {
                    "subject_token": "p0",
                    "text": "alice enjoys weekend trail runs",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                }
            ]
        }
        llm = ScriptedLLM({"extraction": json.dumps(payload)})
        extractor = FactExtractor(llm, ExtractionConfig(), max_tokens=424)
        await extractor.extract(
            roster=roster,
            messages=(("alice", "i ran ten miles on the trail this weekend"),),
            existing_memories_block="",
        )
        assert llm.calls[0].max_tokens == 424

    async def test_gate_rejections_reported(self, roster: Roster) -> None:
        payload = {
            "operations": [
                {
                    "subject_token": "p7",
                    "text": "stranger likes vintage synthesizers a lot",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                },
                {
                    "subject_token": "p0",
                    "text": "what does alice even like doing?",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                },
                {
                    "subject_token": "p0",
                    "text": "posted a funny gif",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                },
                {
                    "subject_token": "p0",
                    "text": "alice likes vintage synthesizers a lot",
                    "confidence": 0.2,
                    "source_message_indexes": [1],
                },
            ]
        }
        extractor, _ = _extractor({"extraction": json.dumps(payload)})
        result = await extractor.extract(
            roster=roster,
            messages=(("alice", "assorted chat lines for context"),),
            existing_memories_block="",
        )
        assert not result.has_candidates
        reasons = {reason for _, reason in result.rejected}
        assert "unknown_subject_token" in reasons
        assert "question" in reasons
        assert "media_share" in reasons
        assert "below_min_confidence" in reasons

    async def test_plagiarism_of_source_message_rejected(self, roster: Roster) -> None:
        source_line = "i have been learning the rust programming language all year"
        payload = {
            "operations": [
                {
                    "subject_token": "p0",
                    "text": source_line,
                    "confidence": 0.95,
                    "source_message_indexes": [1],
                }
            ]
        }
        extractor, _ = _extractor({"extraction": json.dumps(payload)})
        result = await extractor.extract(
            roster=roster,
            messages=(("alice", source_line),),
            existing_memories_block="",
        )
        assert not result.has_candidates
        assert any(reason == "raw_copy" or "copy" in reason for _, reason in result.rejected)

    async def test_max_candidates_cap_enforced(self, roster: Roster) -> None:
        ops = [
            {
                "subject_token": "p0",
                "text": f"alice hobby number {i} is collecting things",
                "confidence": 0.8,
                "source_message_indexes": [1],
            }
            for i in range(20)
        ]
        extractor, _ = _extractor({"extraction": json.dumps({"operations": ops})})
        result = await extractor.extract(
            roster=roster,
            messages=(("alice", "long enough content line to pass the gates"),),
            existing_memories_block="",
        )
        assert len(result.vetted) <= ExtractionConfig().max_candidates_per_batch

    async def test_first_pass_requests_strict_schema(self, roster: Roster) -> None:
        payload = {
            "operations": [
                {
                    "subject_token": "p0",
                    "text": "alice enjoys weekend trail runs",
                    "confidence": 0.9,
                    "source_message_indexes": [1],
                }
            ]
        }
        extractor, llm = _extractor({"extraction": json.dumps(payload)})
        await extractor.extract(
            roster=roster,
            messages=(("alice", "i ran ten miles on the trail this weekend"),),
            existing_memories_block="",
        )
        assert llm.calls[0].response_schema is not None
        props = llm.calls[0].response_schema.get("properties", {})
        assert "operations" in props


class TestNoLlmDegradation:
    async def test_extractor_without_llm_returns_empty(self, roster: Roster) -> None:
        extractor = FactExtractor(None, ExtractionConfig())
        result = await extractor.extract(
            roster=roster,
            messages=(("alice", "anything"),),
            existing_memories_block="",
        )
        assert not result.has_candidates
