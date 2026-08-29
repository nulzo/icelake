"""Extraction stage: LLM call, strict parsing, gates, roster verification (§3.1/§3.3)."""

from __future__ import annotations

from dataclasses import dataclass

from icelake.config import ExtractionConfig
from icelake.errors import StructuredOutputError
from icelake.ingest.gates import (
    GateDecision,
    confidence_gate,
    ephemeral_share_gate,
    plagiarism_gate,
    text_hygiene_gate,
)
from icelake.ingest.roster import Roster
from icelake.models.admin import MeterPurpose
from icelake.models.operations import ExtractionOutput, ProposedFact
from icelake.ports.llm import ChatLLM, LlmMessage, MessageRole
from icelake.prompts import extraction as prompts
from icelake.structured import complete_structured


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


class FactExtractor:
    """Runs the extraction LLM call and hardens its output into vetted candidates."""

    def __init__(
        self,
        llm: ChatLLM | None,
        config: ExtractionConfig,
        *,
        max_tokens: int = 1800,
    ) -> None:
        self._llm = llm
        self._config = config
        self._max_tokens = max_tokens

    async def extract(
        self,
        *,
        roster: Roster,
        messages: tuple[tuple[str, str], ...],
        existing_memories_block: str,
        guild_id: str | None = None,
    ) -> ExtractionResult:
        result = ExtractionResult()
        if self._llm is None or not messages:
            return result
        user_prompt = prompts.render_extraction_prompt(
            roster_block=roster.render(),
            messages_block=prompts.render_messages_block(messages),
            existing_memories_block=existing_memories_block,
        )
        output = await complete_structured(
            self._llm,
            model=ExtractionOutput,
            messages=(
                LlmMessage(
                    role=MessageRole.SYSTEM,
                    content=(
                        f"{prompts.EXTRACTION_SYSTEM_PROMPT}\n\n{prompts.EXTRACTION_INSTRUCTIONS}"
                    ),
                ),
                LlmMessage(role=MessageRole.USER, content=user_prompt),
            ),
            max_tokens=self._max_tokens,
            purpose=MeterPurpose.EXTRACTION,
            guild_id=guild_id,
        )
        if output is None:
            raise StructuredOutputError("extraction output invalid after one repair")

        for operation in output.operations[: self._config.max_candidates_per_batch]:
            decision = self._vet(operation, roster, messages)
            if isinstance(decision, GateDecision):
                result.rejected.append((operation.text, decision.reason))
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
