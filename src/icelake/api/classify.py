"""Natural-language memory-command classifier (API.md §7, opt-in helper).

Regex gate first (cheap); only plausible commands hit the extraction-model LLM.
Returns a typed intent; execution is always the consumer's decision.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field

from icelake.models.admin import MeterPurpose
from icelake.models.common import FrozenModel
from icelake.ports.llm import ChatLLM, LlmMessage, MessageRole
from icelake.structured import complete_structured

_COMMAND_PATTERN = re.compile(
    r"\b(remember|forget|don'?t remember|update|what do you know|what do you remember"
    r"|recall|save this|purge your memories|never remember)\b",
    re.IGNORECASE,
)

_CLASSIFY_PROMPT = """\
Classify this Discord message as a memory-management command.

MESSAGE: {text}

Respond ONLY with JSON:
{{"action": "remember|forget|update|query|none",
  "target_text": "<the fact text being remembered/forgotten/updated, empty otherwise>",
  "confidence": 0.0-1.0}}"""


class CommandAction(StrEnum):
    REMEMBER = "remember"
    FORGET = "forget"
    UPDATE = "update"
    QUERY = "query"
    NONE = "none"


class UserMemoryCommand(FrozenModel):
    """Typed classification of a possible in-chat memory command."""

    action: CommandAction = CommandAction.NONE
    target_text: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class CommandClassifier:
    """Regex-gated, LLM-backed intent classification."""

    def __init__(self, llm: ChatLLM | None) -> None:
        self._llm = llm

    async def classify(self, text: str) -> UserMemoryCommand:
        if not _COMMAND_PATTERN.search(text):
            return UserMemoryCommand(action=CommandAction.NONE)
        if self._llm is None:
            lowered = text.lower().strip()
            if lowered.startswith("remember ") or " remember that " in lowered:
                return UserMemoryCommand(
                    action=CommandAction.REMEMBER,
                    target_text=text.split("remember", 1)[1].strip(" ,.:"),
                    confidence=0.6,
                )
            if lowered.startswith("forget ") or " forget that " in lowered:
                return UserMemoryCommand(action=CommandAction.FORGET, confidence=0.6)
            return UserMemoryCommand(action=CommandAction.NONE, confidence=0.3)

        command = await complete_structured(
            self._llm,
            model=UserMemoryCommand,
            messages=(
                LlmMessage(
                    role=MessageRole.SYSTEM, content="You classify user commands precisely."
                ),
                LlmMessage(role=MessageRole.USER, content=_CLASSIFY_PROMPT.format(text=text)),
            ),
            max_tokens=160,
            purpose=MeterPurpose.CLASSIFY_COMMAND,
        )
        return command if command is not None else UserMemoryCommand()


__all__ = ["CommandAction", "CommandClassifier", "UserMemoryCommand"]
