"""Annotated conversation replay against a configured, unscripted memory pipeline.

No dataset is bundled as 'real'. Supply consented, redacted transcripts with
human-labeled probes. Generated fixtures must declare synthetic provenance.
Scoring measures retrieval/attribution, not answer-model prose quality.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from time import perf_counter

from pydantic import Field, model_validator

from icelake import DiscordMemory, MessageEvent, RecallQuery, Scope
from icelake.models.common import FrozenModel
from icelake.permissions import Audience
from icelake.ports.clock import FixedClock


class Provenance(StrEnum):
    CONSENTED_REDACTED = "consented_redacted"
    SYNTHETIC = "synthetic"


class ExpectedMemory(FrozenModel):
    subject_id: str | None
    text_contains: str = Field(min_length=1)
    source_message_ids: tuple[str, ...] = Field(min_length=1)


class Probe(FrozenModel):
    after_message: int = Field(ge=1)
    question: str = Field(min_length=1)
    audience: Audience
    subject_ids: tuple[str, ...] = ()
    expected: tuple[ExpectedMemory, ...] = ()
    forbidden_text: tuple[str, ...] = ()
    must_abstain: bool = False
    top_k: int = Field(default=8, ge=1, le=100)


class ConversationCase(FrozenModel):
    case_id: str = Field(min_length=1)
    provenance: Provenance
    permission_reference: str = Field(min_length=1)
    messages: tuple[MessageEvent, ...] = Field(min_length=1)
    probes: tuple[Probe, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_timeline(self) -> ConversationCase:
        guilds = {m.guild_id for m in self.messages}
        ids = {m.message_id for m in self.messages}
        if len(guilds) != 1 or len(ids) != len(self.messages):
            raise ValueError("one guild and unique message IDs are required per case")
        if any(
            a.created_at > b.created_at
            for a, b in zip(self.messages, self.messages[1:], strict=False)
        ):
            raise ValueError("messages must be chronological")
        for probe in self.probes:
            if probe.after_message > len(self.messages) or (probe.must_abstain and probe.expected):
                raise ValueError("invalid probe checkpoint or abstention label")
            prefix = {m.message_id for m in self.messages[: probe.after_message]}
            if any(not prefix.issuperset(e.source_message_ids) for e in probe.expected):
                raise ValueError("expected evidence must precede the probe")
        return self


class ProbeResult(FrozenModel):
    checkpoint: int
    expected: int
    matched: int
    returned: int
    relevant_returned: int
    grounded_returned: int
    forbidden_matches: int
    abstention_passed: bool
    latency_ms: float


class ConversationResult(FrozenModel):
    case_id: str
    provenance: Provenance
    extraction_model: str
    probes: tuple[ProbeResult, ...]
    passed: bool
    meter: dict[str, object]


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


async def evaluate_case(
    case: ConversationCase,
    memory_factory: Callable[[FixedClock], DiscordMemory],
    *,
    timeout_seconds: float = 300,
) -> ConversationResult:
    """Replay one isolated case with real configured extraction; close even on failure.

    Supply a factory creating a fresh store per case. Never point this runner at
    a production database. Annotation text is not supplied to the extraction LLM.
    """
    clock = FixedClock(case.messages[0].created_at)
    memory = memory_factory(clock)
    if memory.config.workers.enabled:
        raise ValueError("disable background workers during isolated evaluation")
    results: list[ProbeResult] = []
    async with asyncio.timeout(timeout_seconds), memory:
        health = await memory.ops.health()
        if any(c.component == "llm" and c.status.value != "ok" for c in health.components):
            raise ValueError("conversation evaluation requires an extraction provider")
        if memory.active_guilds:
            raise ValueError("evaluation requires an empty isolated store")
        for index, message in enumerate(case.messages, start=1):
            clock.advance((message.created_at - clock.now()).total_seconds())
            await memory.observe(message)
            pending = [p for p in case.probes if p.after_message == index]
            if pending:
                await memory.flush()
                health = await memory.ops.health()
                if health.pending_messages or health.dead_letters:
                    raise RuntimeError("ingest failed or remains pending; evaluation is incomplete")
                for probe in pending:
                    results.append(await _probe(memory, case, probe))
        return ConversationResult(
            case_id=case.case_id,
            provenance=case.provenance,
            extraction_model=memory.config.llm.model or "injected-provider",
            probes=tuple(results),
            passed=all(
                r.matched == r.expected
                and not r.forbidden_matches
                and r.abstention_passed
                and r.grounded_returned == r.returned
                for r in results
            ),
            meter=memory.ops.meter_snapshot().as_dict(),
        )


async def _probe(memory: DiscordMemory, case: ConversationCase, probe: Probe) -> ProbeResult:
    started = perf_counter()
    result = await memory.recall(
        RecallQuery(
            guild_id=case.messages[0].guild_id,
            audience=probe.audience,
            text=probe.question,
            subject_ids=probe.subject_ids,
            scope=Scope.SUBJECTS if probe.subject_ids else Scope.GUILD,
            top_k=probe.top_k,
            max_per_subject=probe.top_k,
        )
    )
    records = tuple(scored.fact for scored in result.facts)
    matches = [
        {
            r.id
            for r in records
            if r.subject_id == expected.subject_id
            and _normalize(expected.text_contains) in _normalize(r.text)
            and set(expected.source_message_ids).intersection(c.message_id for c in r.citations)
        }
        for expected in probe.expected
    ]
    known = {m.message_id: m for m in case.messages[: probe.after_message]}
    grounded = sum(
        bool(r.citations)
        and all(
            c.message_id in known
            and known[c.message_id].channel_id == c.channel_id
            and known[c.message_id].author_id == c.author_id
            and c.guild_id == r.guild_id
            for c in r.citations
        )
        for r in records
    )
    return ProbeResult(
        checkpoint=probe.after_message,
        expected=len(probe.expected),
        matched=sum(bool(m) for m in matches),
        returned=len(records),
        relevant_returned=len(set().union(*matches)),
        grounded_returned=grounded,
        forbidden_matches=sum(
            any(_normalize(f) in _normalize(r.text) for r in records) for f in probe.forbidden_text
        ),
        abstention_passed=not probe.must_abstain or not records,
        latency_ms=(perf_counter() - started) * 1000,
    )


def load_cases(path: Path) -> tuple[ConversationCase, ...]:
    """Load JSONL annotations; reject empty or duplicate-ID datasets."""
    cases = tuple(
        ConversationCase.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    )
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError("dataset must contain unique, nonempty conversation cases")
    return cases


def save_results(path: Path, results: tuple[ConversationResult, ...]) -> None:
    """Write numeric results and case IDs, excluding raw conversations and credentials."""
    path.write_text(json.dumps([r.model_dump(mode="json") for r in results], indent=2) + "\n")
