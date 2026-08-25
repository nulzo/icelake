"""Anti-fabrication gates: pure functions deciding whether a proposed fact may be
stored (PLAN.md §3.3). Ported from the bot's memory_quality.py hardening.

Every gate is side-effect-free and takes explicit context so it is trivially unit-
testable and fuzzable. Gates are the second-to-last line of defense before storage;
the roster verification gate runs in the pipeline.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

SNOWFLAKE = re.compile(r"\b\d{15,25}\b")
MENTION_TAG = re.compile(r"<@!?&?\d+>")
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
QUESTION_START = re.compile(
    r"^(do|does|did|is|are|was|were|who|what|when|where|why|how|can|could|should)\b",
    re.IGNORECASE,
)
RANT_DENIAL = re.compile(r"\b(no no no|i don'?t|stop it)\b", re.IGNORECASE)
SHARE_VERB = re.compile(r"\b(shared|posted|sent|uploaded|linked|dropped)\b", re.IGNORECASE)
MEDIA_TARGET = re.compile(
    r"\b(link|video|gif|image|photo|meme|file|youtube|tiktok|tenor)s?\b",
    re.IGNORECASE,
)
LLM_META_MARKERS = (
    "operations",
    "chain of thought",
    "let me think",
    "final answer:",
    "empty operations",
    "as requested",
    "here is the json",
    "based on the messages",
)
REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i apologize",
    "i'm sorry",
    "as an ai",
    "i'm unable",
    "against my guidelines",
)
EPHEMERAL_CLAIMS = (
    "expressed a belief",
    "heard that",
    "claims that",
    "read online",
    "someone said",
    "rumor",
)
DURABLE_MARKERS = (
    "my name is",
    "call me",
    "i live",
    "i work",
    "i study",
    "allergic",
    "birthday",
    "pronouns",
    "partner",
    "spouse",
    "husband",
    "wife",
    "moved to",
    "graduated",
    "works at",
    "studies",
)

MAX_FACT_CHARS = 400
MIN_FACT_CHARS = 12
PLAGIARISM_RATIO = 0.88


def normalize_text(text: str) -> str:
    """Lowercase with collapsed whitespace — the dedup key basis."""
    return " ".join(text.strip().lower().split())


def strip_urls(text: str) -> str:
    return URL.sub(" ", text)


class GateDecision:
    """Outcome of the gate chain for one proposed fact."""

    __slots__ = ("allowed", "reason")

    def __init__(self, allowed: bool, reason: str = "") -> None:
        self.allowed = allowed
        self.reason = reason

    def __repr__(self) -> str:
        state = "allow" if self.allowed else f"reject({self.reason})"
        return f"GateDecision.{state}"


ALLOW = GateDecision(True)


def _reject(reason: str) -> GateDecision:
    return GateDecision(False, reason)


def text_hygiene_gate(text: str) -> GateDecision:
    """Structural hygiene on synthesized fact text."""
    if not text or not text.strip():
        return _reject("empty")
    if len(text) > MAX_FACT_CHARS:
        return _reject("too_long")
    if len(normalize_text(text)) < MIN_FACT_CHARS:
        return _reject("too_short")
    lowered = text.lower()
    if any(marker in lowered for marker in REFUSAL_MARKERS):
        return _reject("refusal")
    meta_hits = sum(1 for marker in LLM_META_MARKERS if marker in lowered)
    if meta_hits >= 2 or (meta_hits >= 1 and len(text) > 200):
        return _reject("llm_meta")
    if SNOWFLAKE.search(text):
        return _reject("snowflake_in_text")
    if MENTION_TAG.search(text):
        return _reject("mention_tag_in_text")
    if text.strip().endswith("?") or QUESTION_START.match(text.strip()):
        return _reject("question")
    if RANT_DENIAL.search(lowered):
        return _reject("rant_denial")
    return ALLOW


def ephemeral_share_gate(text: str) -> GateDecision:
    """Bare link/media shares carry no durable meaning."""
    stripped_urls = strip_urls(text).strip()
    if not stripped_urls:
        return _reject("bare_link")
    lowered = text.lower()
    if SHARE_VERB.search(lowered) and MEDIA_TARGET.search(lowered):
        return _reject("media_share")
    if SHARE_VERB.search(lowered) and URL.search(text):
        return _reject("link_share")
    return ALLOW


def confidence_gate(confidence: float, min_confidence: float, text: str) -> GateDecision:
    """Confidence floor; ephemeral-claim phrasing needs a higher bar unless durable."""
    if confidence >= min_confidence and not any(m in text.lower() for m in EPHEMERAL_CLAIMS):
        return ALLOW
    if any(marker in text.lower() for marker in EPHEMERAL_CLAIMS):
        if confidence >= 0.75 and any(m in text.lower() for m in DURABLE_MARKERS):
            return ALLOW
        return _reject("ephemeral_claim")
    return _reject("below_min_confidence")


def plagiarism_gate(fact_text: str, source_texts: tuple[str, ...]) -> GateDecision:
    """Reject near-verbatim copies of chat lines — memories must be synthesized."""
    normalized_fact = normalize_text(strip_urls(fact_text))
    for source in source_texts:
        normalized_source = normalize_text(source)
        if normalized_fact == normalized_source:
            return _reject("raw_copy")
        if len(normalized_fact) > 20 and len(normalized_source) > 20:
            ratio = SequenceMatcher(None, normalized_fact, normalized_source).ratio()
            if ratio >= PLAGIARISM_RATIO:
                return _reject("near_copy")
    return ALLOW


def batch_worth_extracting(messages: tuple[str, ...], *, min_chars: int = 28) -> bool:
    """Noise gate: skip the extraction LLM when a batch has no substantive content."""
    joined = normalize_text(" ".join(messages))
    if len(joined) < 8:
        return False
    has_durable = any(marker in joined for marker in DURABLE_MARKERS)
    if has_durable:
        return True
    word_count = len(joined.split())
    longest = max((len(normalize_text(m)) for m in messages), default=0)
    return word_count >= 12 or longest >= min_chars
