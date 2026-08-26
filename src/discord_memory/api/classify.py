"""Natural-language memory-command classifier (API.md §7, opt-in helper).

Regex gate first (cheap); only plausible commands hit the extraction-model LLM.
Returns a typed intent; execution is always the consumer's decision.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import Field

from discord_memory.models.common import FrozenModel
from discord_memory.ports.llm import ChatLLM, ChatRequest, LlmMessage

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


class CommandAction(Enum):
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
            return UserMemoryCommand(action=CommandAction.NONE, confidence=0.3)

        response = await self._llm.complete(
            ChatRequest(
                messages=(
                    LlmMessage(role="system", content="You classify user commands precisely."),
                    LlmMessage(role="user", content=_CLASSIFY_PROMPT.format(text=text)),
                ),
                json_mode=True,
                max_tokens=160,
                purpose="classify_command",
            )
        )
        return _parse(response.text)


def _parse(text: str) -> UserMemoryCommand:
    import json

    try:
        start = text.find("{")
        end = text.rfind("}")
        payload = json.loads(text[start : end + 1])
        action = str(payload.get("action", "none")).lower()
        try:
            action_enum = CommandAction(action)
        except ValueError:
            action_enum = CommandAction.NONE
        return UserMemoryCommand(
            action=action_enum,
            target_text=str(payload.get("target_text", "")),
            confidence=float(payload.get("confidence", 0.0)),
        )
    except (ValueError, TypeError):
        return UserMemoryCommand(action=CommandAction.NONE)


__all__ = ["CommandAction", "CommandClassifier", "UserMemoryCommand"]
