"""Harness tests use explicitly synthetic conversations, not accuracy evidence."""

from datetime import UTC, datetime

import pytest
from evals.conversations import ConversationCase, Provenance, evaluate_case, load_cases
from pydantic import ValidationError

from icelake import DiscordMemory
from tests.conftest import ScriptedLLM, extraction_response, make_config


def case():
    return ConversationCase.model_validate(
        {
            "case_id": "synthetic-harness-regression",
            "provenance": "synthetic",
            "permission_reference": "repository-owned fixture",
            "messages": [
                {
                    "message_id": "m1",
                    "guild_id": "g",
                    "channel_id": "private",
                    "author_id": "123456789012345678",
                    "content": "I spend weekends repairing old watches with my grandfather.",
                    "created_at": datetime.now(UTC),
                    "author_display_name": "Alice",
                }
            ],
            "probes": [
                {
                    "after_message": 1,
                    "question": "What is Alice's hobby?",
                    "audience": {"channel_id": "private", "requester_id": "123456789012345678"},
                    "expected": [
                        {
                            "subject_id": "123456789012345678",
                            "text_contains": "antique watches",
                            "source_message_ids": ["m1"],
                        }
                    ],
                },
                {
                    "after_message": 1,
                    "question": "What is Alice's hobby?",
                    "audience": {"channel_id": "public", "requester_id": "123456789012345678"},
                    "must_abstain": True,
                    "forbidden_text": ["watches"],
                },
            ],
        }
    )


async def test_harness_runs_real_pipeline_and_checks_attribution_and_private_abstention():
    def factory(clock):
        llm = ScriptedLLM(
            {
                "extraction": extraction_response(
                    [
                        {
                            "subject_token": "p0",
                            "text": "Alice restores antique watches as a hobby",
                            "category": "interests",
                            "confidence": 0.9,
                            "source_message_indexes": [1],
                        }
                    ]
                )
            }
        )
        return DiscordMemory(make_config(), llm=llm, clock=clock)

    result = await evaluate_case(case(), factory)
    assert result.provenance is Provenance.SYNTHETIC
    assert result.passed
    assert result.probes[0].grounded_returned == 1
    assert result.probes[1].returned == 0


@pytest.mark.parametrize("mutation", ["future_evidence", "duplicate_message", "no_probes"])
def test_annotations_reject_invalid_timelines(mutation):
    raw = case().model_dump(mode="json")
    if mutation == "future_evidence":
        raw["probes"][0]["expected"][0]["source_message_ids"] = ["unknown"]
    elif mutation == "duplicate_message":
        raw["messages"] *= 2
    else:
        raw["probes"] = []
    with pytest.raises(ValidationError):
        ConversationCase.model_validate(raw)


def test_empty_dataset_is_not_a_passing_evaluation(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="nonempty"):
        load_cases(path)
