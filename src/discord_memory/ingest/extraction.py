"""Extraction stage: LLM call, strict parsing, gates, roster verification (§3.1/§3.3)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pydantic import ValidationError

from discord_memory.config import ExtractionConfig
from discord_memory.ingest.gates import (
    GateDecision,
    confidence_gate,
    ephemeral_share_gate,
    plagiarism_gate,
    text_hygiene_gate,
)
from discord_memory.ingest.roster import Roster
from discord_memory.models.facts import FactCategory
from discord_memory.models.operations import ExtractionOutput, ProposedFact
from discord_memory.ports.llm import ChatLLM, ChatRequest, LlmMessage
from discord_memory.prompts import extraction as prompts

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VettedFact:
    """A proposed fact that survived every gate and the roster verification."""

    proposal: ProposedFact
    subject_id: str | None
    speaker_id: str | None


class ExtractionResult:
    """Outcome of one extraction pass over a claimed batch."""

    def __init__(self) -> None:
        self.vetted: list[VettedFact] = []
        self.rejected: list[tuple[str, str]] = []

    @property
    def has_candidates(self) -> bool:
        return bool(self.vetted)


def _parse_json_object(text: str) -> dict[str, object]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in response")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


class FactExtractor:
    """Runs the extraction LLM call and hardens its output into vetted candidates."""

    def __init__(self, llm: ChatLLM | None, config: ExtractionConfig) -> None:
        self._llm = llm
        self._config = config

    async def extract(
        self,
        *,
        roster: Roster,
        messages: tuple[tuple[str, str], ...],
        existing_memories_block: str,
    ) -> ExtractionResult:
        result = ExtractionResult()
        if self._llm is None or not messages:
            return result
        user_prompt = prompts.render_extraction_prompt(
            roster_block=roster.render(),
            messages_block=prompts.render_messages_block(messages),
            existing_memories_block=existing_memories_block,
        )
        response = await self._llm.complete(
            ChatRequest(
                messages=(
                    LlmMessage(role="system", content=prompts.EXTRACTION_SYSTEM_PROMPT),
                    LlmMessage(role="user", content=user_prompt),
                ),
                json_mode=True,
                max_tokens=1800,
                purpose="extraction",
            )
        )
        try:
            payload = _parse_json_object(response.text)
            output = ExtractionOutput.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            logger.warning("Extraction parse failed: %s", exc)
            return result

        for operation in output.operations[: self._config.max_candidates_per_batch]:
            decision = self._vet(operation, roster, messages)
            if isinstance(decision, GateDecision):
                result.rejected.append((operation.text[:80], decision.reason))
                continue
            result.vetted.append(decision)
        return result

    def _vet(
        self,
        operation: ProposedFact,
        roster: Roster,
        messages: tuple[tuple[str, str], ...],
    ) -> VettedFact | GateDecision:
        """Run all gates; returns the vetted fact or the failing gate's decision."""
        hygiene = text_hygiene_gate(operation.text)
        if not hygiene.allowed:
            return hygiene
        share = ephemeral_share_gate(operation.text)
        if not share.allowed:
            return share
        source_texts = tuple(
            content
            for index, content in enumerate((content for _, content in messages), 1)
            if index in set(operation.source_message_indexes)
        )
        plagiarism = plagiarism_gate(operation.text, source_texts)
        if not plagiarism.allowed:
            return plagiarism
        conf = confidence_gate(operation.confidence, self._config.min_confidence, operation.text)
        if not conf.allowed:
            return conf

        token = operation.subject_token.strip()
        if not roster.knows(token):
            return GateDecision(False, "unknown_subject_token")
        subject_id = None if token == "server" else roster.user_id_for(token)

        speaker_id = None
        if operation.speaker_token:
            speaker_token = operation.speaker_token.strip()
            if speaker_token != "server" and not roster.knows(speaker_token):
                return GateDecision(False, "unknown_speaker_token")
            speaker_id = roster.user_id_for(speaker_token)
        return VettedFact(
            proposal=operation,
            subject_id=subject_id,
            speaker_id=speaker_id,
        )


def category_of(proposal: ProposedFact) -> FactCategory:
    try:
        return FactCategory(proposal.category)
    except ValueError:
        return FactCategory.GENERAL
