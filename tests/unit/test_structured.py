"""complete_structured: schema on the first pass, one feedback repair, then None."""

from __future__ import annotations

from pydantic import BaseModel

from discord_memory.ports.llm import ChatRequest, LlmMessage
from discord_memory.structured import complete_structured
from tests.conftest import ScriptedLLM


class Widget(BaseModel):
    name: str
    count: int = 0


_MESSAGES = (LlmMessage(role="user", content="give me a widget"),)


async def complete(llm: ScriptedLLM) -> Widget | None:
    return await complete_structured(
        llm, model=Widget, messages=_MESSAGES, max_tokens=100, purpose="widget"
    )


class TestCompleteStructured:
    async def test_first_pass_sends_schema_and_validates(self) -> None:
        llm = ScriptedLLM({"widget": '{"name": "gear", "count": 2}'})
        output = await complete(llm)
        assert output == Widget(name="gear", count=2)
        assert llm.calls[0].response_schema == Widget.model_json_schema()

    async def test_invalid_output_retries_once_with_feedback(self) -> None:
        state = {"calls": 0}

        def handler(_request: ChatRequest) -> str:
            state["calls"] += 1
            return "not json" if state["calls"] == 1 else '{"name": "fixed"}'

        llm = ScriptedLLM({"widget": handler})
        output = await complete(llm)
        assert output == Widget(name="fixed")
        assert len(llm.calls) == 2
        retry_messages = llm.calls[1].messages
        assert retry_messages[-2].role == "assistant"
        assert "failed validation" in retry_messages[-1].content

    async def test_double_failure_returns_none(self) -> None:
        llm = ScriptedLLM({"widget": "garbage"})
        assert await complete(llm) is None
        assert len(llm.calls) == 2
