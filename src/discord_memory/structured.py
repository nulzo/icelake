"""Typed structured completions: the single way JSON crosses the LLM boundary.

Every JSON-producing call site goes through :func:`complete_structured`. The
Pydantic model is both the wire contract (sent as strict ``json_schema`` on the
first pass — constrained decoding on capable providers) and the validator. One
retry feeds the validation error back to the model; if it still fails, the call
returns ``None`` and the caller applies its domain default (skip the batch,
default to ADD, treat as no command). Per-call-site parsers are forbidden.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ValidationError

from discord_memory._json import parse_json_object
from discord_memory.ports.llm import ChatLLM, ChatRequest, LlmMessage

logger = logging.getLogger(__name__)


async def complete_structured[T: BaseModel](
    llm: ChatLLM,
    *,
    model: type[T],
    messages: tuple[LlmMessage, ...],
    max_tokens: int,
    purpose: str,
    guild_id: str | None = None,
) -> T | None:
    """Complete against a Pydantic model; ``None`` if invalid after one repair."""
    schema = model.model_json_schema()

    async def call(turns: tuple[LlmMessage, ...]) -> str:
        response = await llm.complete(
            ChatRequest(
                messages=turns,
                max_tokens=max_tokens,
                purpose=purpose,
                response_schema=schema,
                guild_id=guild_id,
            )
        )
        return response.text

    first = await call(messages)
    try:
        return model.model_validate(parse_json_object(first))
    except (ValueError, ValidationError) as error:
        feedback = (
            *messages,
            LlmMessage(role="assistant", content=first),
            LlmMessage(
                role="user",
                content=f"Your response failed validation ({error}). "
                "Re-emit ONLY the corrected JSON conforming to the schema.",
            ),
        )
        second = await call(feedback)
        try:
            output = model.model_validate(parse_json_object(second))
        except (ValueError, ValidationError):
            logger.warning("%s: structured output invalid after one repair; dropping", purpose)
            return None
        logger.info("%s: structured output repaired after one retry", purpose)
        return output


__all__ = ["complete_structured"]
