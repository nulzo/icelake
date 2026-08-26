"""LLM JSON extraction with staged repair (ported from bot utils/llm_json.py).

Ladder: fence strip -> balanced-object scan (string/escape-aware) -> truncation
repair (closes open strings + bracket stack) -> candidate slicing. A truncated
extraction response should lose nothing recoverable.
"""

from __future__ import annotations

import json
from typing import Final

_FENCES: Final = (("```json", "```"), ("```JSON", "```"), ("```", "```"))


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    for opening, closing in _FENCES:
        if stripped.startswith(opening):
            inner = stripped[len(opening) :]
            end = inner.rfind(closing)
            return (inner[:end] if end != -1 else inner).strip()
    return stripped


def _scan_balanced(text: str, start: int) -> str | None:
    """Extract one balanced JSON object starting at ``text[start] == '{'``."""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _close_open_constructs(candidate: str) -> str:
    """Repair truncated JSON: close an open string literal, then brackets."""
    in_string = False
    escaped = False
    stack: list[str] = []
    for char in candidate:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"} and stack and stack[-1] == char:
            stack.pop()
    repaired = candidate
    if escaped:
        repaired += '"'
    if in_string:
        repaired += '"'
    while stack:
        repaired += stack.pop()
    return repaired


def parse_json_object(text: str) -> dict[str, object]:
    """Parse the first JSON object in ``text`` with staged repair.

    Raises ``ValueError`` only when no object can be recovered at all.
    """
    cleaned = _strip_fence(text)

    direct_start = cleaned.find("{")
    direct_end = cleaned.rfind("}")
    if direct_start != -1 and direct_end > direct_start:
        try:
            parsed = json.loads(cleaned[direct_start : direct_end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    scanner_start = max(direct_start, 0)
    while scanner_start != -1:
        balanced = _scan_balanced(cleaned, scanner_start)
        if balanced is not None:
            try:
                parsed = json.loads(balanced)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
        scanner_start = cleaned.find("{", scanner_start + 1)

    if direct_start != -1:
        truncated = _close_open_constructs(cleaned[direct_start:])
        try:
            parsed = json.loads(truncated)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError("no recoverable JSON object in response")


__all__ = ["parse_json_object"]
