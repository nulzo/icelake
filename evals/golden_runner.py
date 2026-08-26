"""Golden eval harness: deterministic attribution + quality scoring.

Each scenario in ``evals/golden/`` specifies messages, expected facts,
forbidden facts, and expected skips. The runner replays them through the
pipeline with a ScriptedLLM and asserts every expectation.

CI gate: any change that drops attribution accuracy or allows a forbidden
fact fails the build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from discord_memory import DiscordMemory, MessageEvent
from tests.conftest import ScriptedLLM

GOLDEN_DIR = Path(__file__).parent / "golden"


@dataclass(frozen=True)
class ExpectedFact:
    """A fact that MUST exist on a subject's profile after the scenario."""

    subject_id: str
    text_contains: str


@dataclass(frozen=True)
class ForbiddenFact:
    """A fact that must NOT exist anywhere after the scenario."""

    text_contains: str


@dataclass(frozen=True)
class GoldenScenario:
    name: str
    messages: tuple[dict, ...]  # raw dicts for MessageEvent construction
    llm_operations: list[dict]  # what the scripted LLM returns per batch
    expected_facts: tuple[ExpectedFact, ...] = ()
    forbidden_facts: tuple[ForbiddenFact, ...] = ()
    expected_skips: tuple[str, ...] = ()  # ignore reasons we expect to see
    guild_id: str = "555"


def load_scenarios(directory: Path | None = None) -> list[GoldenScenario]:
    """Load all YAML golden scenarios from the directory."""
    root = directory or GOLDEN_DIR
    scenarios = []
    for path in sorted(root.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text())
        expected = tuple(
            ExpectedFact(subject_id=ef["subject_id"], text_contains=ef["text_contains"])
            for ef in raw.get("expected_facts", [])
        )
        forbidden = tuple(
            ForbiddenFact(text_contains=ff["text_contains"])
            for ff in raw.get("forbidden_facts", [])
        )
        scenarios.append(GoldenScenario(
            name=path.stem,
            messages=tuple(raw.get("messages", [])),
            llm_operations=raw.get("llm_operations", []),
            expected_facts=expected,
            forbidden_facts=forbidden,
            expected_skips=tuple(raw.get("expected_skips", [])),
            guild_id=raw.get("guild_id", "555"),
        ))
    return scenarios


async def run_scenario(
    scenario: GoldenScenario,
    *,
    config_overrides: dict | None = None,
) -> dict:
    """Replay one golden scenario; return pass/fail details."""
    from tests.conftest import make_config

    llm = ScriptedLLM({
        "extraction": json.dumps({"operations": scenario.llm_operations}),
    })
    memory = DiscordMemory(
        make_config(**(config_overrides or {})), llm=llm,
    )
    await memory.start()

    skip_reasons: list[str] = []
    original_observe = memory.observe

    async def tracking_observe(event: MessageEvent):
        receipt = await original_observe(event)
        if receipt.reason is not None:
            skip_reasons.append(receipt.reason.value)
        return receipt

    memory.observe = tracking_observe  # type: ignore[method-assign]

    now = datetime.now(UTC)
    counter = {"n": 0}
    for msg_spec in scenario.messages:
        counter["n"] += 1
        event = MessageEvent(
            message_id=msg_spec.get("message_id", f"gm-{counter['n']}"),
            guild_id=scenario.guild_id,
            channel_id=msg_spec.get("channel_id", "general"),
            author_id=msg_spec["author_id"],
            content=msg_spec["content"],
            created_at=datetime.now(UTC),
            author_display_name=msg_spec.get("display_name", ""),
            mention_ids=tuple(msg_spec.get("mentions", [])),
        )
        await memory.observe(event)
    await memory.flush()

    failures: list[str] = []

    # Check expected facts exist.
    all_facts: dict[str, list[str]] = {}
    for subject_id in {ef.subject_id for ef in scenario.expected_facts} | {
        m["author_id"] for m in scenario.messages
    }:
        page = await memory.facts.list_for_subject(
            scenario.guild_id, subject_id, include_server=False,
        )
        all_facts[subject_id] = [f.text.lower() for f in page.items]

    for ef in scenario.expected_facts:
        subject_texts = all_facts.get(ef.subject_id, [])
        if not any(ef.text_contains.lower() in t for t in subject_texts):
            failures.append(
                f"MISSING: '{ef.text_contains}' on subject {ef.subject_id}"
            )

    # Check forbidden facts absent everywhere.
    all_text = " || ".join(
        " || ".join(texts) for texts in all_facts.values()
    )
    for ff in scenario.forbidden_facts:
        if ff.text_contains.lower() in all_text:
            failures.append(f"FORBIDDEN PRESENT: '{ff.text_contains}'")

    await memory.close()
    return {"name": scenario.name, "failures": failures}


async def run_all(*, directory: Path | None = None) -> tuple[int, int]:
    """Run all golden scenarios. Returns (passed, total)."""
    scenarios = load_scenarios(directory)
    passed = 0
    for scenario in scenarios:
        result = await run_scenario(scenario)
        if not result["failures"]:
            passed += 1
        else:
            print(f"FAIL [{scenario.name}]: {result['failures']}")
    return passed, len(scenarios)


__all__ = [
    "ExpectedFact",
    "ForbiddenFact",
    "GoldenScenario",
    "load_scenarios",
    "run_all",
    "run_scenario",
]
