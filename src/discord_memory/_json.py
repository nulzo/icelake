"""LLM JSON extraction with staged repair (ported from bot utils/llm_json.py).

Ladder: fence strip -> balanced-object scan (string/escape-aware) -> truncation
repair (closes open strings + bracket stack) -> candidate slicing. A truncated
extraction response should lose nothing recoverable.
"""

from __future__ import annotations

import json
import re
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


def coerce_extraction_payload(payload: dict[str, object]) -> dict[str, object]:
    """Normalize alternate LLM fact schemas into the canonical operations shape.

    Live models (Gemini/DeepSeek/etc.) sometimes answer in Graphiti-style
    triples ({"facts": [{"subject": ..., "predicate": ..., "object": ...}]})
    or use "memories" instead of "operations". Strictly rejecting those threw
    away entire batches — total learning loss. We accept the common variants,
    synthesize text for bare triples, and leave anything unrecognizable to the
    validation gates downstream.
    """
    if "operations" in payload and isinstance(payload["operations"], list):
        return payload
    for key in ("facts", "memories", "items", "results"):
        raw = payload.get(key)
        if isinstance(raw, list):
            return {"operations": [_coerce_item(i) for i in raw]}
    # A single triple-shaped object at top level.
    coerced = _coerce_item(payload)
    if coerced is not None:
        return {"operations": [coerced]}
    return payload


def _coerce_item(item: object) -> dict[str, object] | None:
    """Coerce one candidate record into an operations-style dict, or None."""
    if not isinstance(item, dict):
        return None
    out: dict[str, object] = dict(item)

    # Token normalization: "<p0>", 0, "p0" all become "p0".
    for token_key in ("subject_token", "speaker_token"):
        value = out.get(token_key)
        if isinstance(value, int):
            out[token_key] = f"p{value}"
        elif isinstance(value, str):
            stripped = value.strip().strip("<>").lower()
            if stripped.isdigit():
                out[token_key] = f"p{stripped}"
            elif re.fullmatch(r"p\d+", stripped):
                out[token_key] = stripped

    # Graphiti-triple shape: {subject, predicate, object} with no text.
    has_text = bool(str(out.get("text", "")).strip())
    if not has_text:
        subj = out.pop("subject", None) or out.pop("subject_entity", None)
        pred = out.pop("predicate", None) or out.pop("verb", None) or out.pop("relationship", None)
        obj = out.pop("object", None) or out.pop("object_entity", None)
        if isinstance(subj, dict):
            subj = subj.get("name") or subj.get("id")
        if isinstance(obj, dict):
            obj = obj.get("name") or obj.get("id")
        if isinstance(pred, str) and subj and obj:
            verb = str(pred).strip().replace("_", " ")
            out["text"] = f"{subj} {verb.lower()} {obj}"
            out.setdefault(
                "relations",
                [
                    {
                        "verb": str(pred).strip(),
                        "from_token": out.get("subject_token"),
                        "to_entity": obj if not str(obj).isdigit() else None,
                        "to_token": obj if str(obj).isdigit() else None,
                    }
                ],
            )
            text_value = out["text"]
            if isinstance(text_value, str):
                out["text"] = text_value.strip()

    # Drop triple bookkeeping keys we cannot map.
    for junk in ("id", "uuid", "entity_types", "edge_type_map", "created_at"):
        out.pop(junk, None)
    return out


__all__ += ["coerce_extraction_payload"]
