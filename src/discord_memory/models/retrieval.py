"""Retrieval boundary models: queries, scored facts, citations, prompt context."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from collections.abc import Callable

from pydantic import Field

from discord_memory.models.common import FrozenModel, TokenUsage
from discord_memory.models.facts import FactRecord
from discord_memory.models.identity import Resolution


class Scope(StrEnum):
    """Candidate-space restriction, enforced in store queries (PLAN.md §5.1)."""

    SUBJECTS = "subjects"
    GUILD = "guild"
    SERVER = "server"


class ChannelName(StrEnum):
    """Recall channels. ``DEFAULT`` is the benchmarked core set; ``DISCOVERY`` adds hops."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    LINKS = "links"
    BASELINE = "baseline"
    ENTITY = "entity"
    GRAPH_HOP = "graph_hop"


ChannelSet = frozenset[ChannelName]
"""Immutable channel selection: ``frozenset`` of :class:`ChannelName`."""

CHANNELS_DEFAULT: ChannelSet = frozenset(
    {
        ChannelName.VECTOR,
        ChannelName.KEYWORD,
        ChannelName.LINKS,
        ChannelName.BASELINE,
        ChannelName.ENTITY,
    }
)
CHANNELS_DISCOVERY: ChannelSet = CHANNELS_DEFAULT | {ChannelName.GRAPH_HOP}
CHANNELS_ALL: ChannelSet = frozenset(ChannelName)


def channels(*names: ChannelName) -> ChannelSet:
    """Build a channel selection from explicit names."""
    return frozenset(names)


class ScoreComponents(FrozenModel):
    """Calibrated [0,1] breakdown behind a final score — ranking is debuggable."""

    semantic: float = 0.0
    lexical: float = 0.0
    entity: float = 0.0
    strength: float = 0.0


class ScoredFact(FrozenModel):
    """A fact plus its calibrated score and provenance within one recall call."""

    fact: FactRecord
    score: float = Field(ge=0.0, le=1.0)
    components: ScoreComponents = ScoreComponents()
    matched_channels: tuple[ChannelName, ...] = ()
    hop_path: tuple[str, ...] = ()


class RecallQuery(FrozenModel):
    """Explicit retrieval request (API.md §6.2). Scope enforced store-side."""

    guild_id: str
    text: str | None = None
    subject_ids: tuple[str, ...] = ()
    pair_ids: tuple[str, str] | None = None
    entity_hint: str | None = None
    scope: Scope = Scope.SUBJECTS
    exclude_ids: tuple[str, ...] = ()
    top_k: int = 8
    max_per_subject: int = 4
    min_score: float = 0.0
    token_budget: int = 600
    channels: ChannelSet | None = None
    as_of: datetime | None = None   # time-travel: what was known at this instant?


class Citation(FrozenModel):
    """Citation binding for an injected fact (``mem:N`` → jump link)."""

    ref: str
    fact_id: str
    url: str
    snippet: str = ""
    subject_id: str | None = None
    subject_name: str = ""


class RecallWarning(StrEnum):
    IDENTITY_AMBIGUOUS = "identity_ambiguous"
    BUDGET_TRIMMED = "budget_trimmed"
    DEGRADED_CHANNEL = "degraded_channel"
    SUBJECT_UNRESOLVED = "subject_unresolved"


class RecallResult(FrozenModel):
    """Structured recall output; transport-free."""

    facts: tuple[ScoredFact, ...] = ()
    citations: tuple[Citation, ...] = ()
    resolutions: tuple[Resolution, ...] = ()
    warnings: tuple[RecallWarning, ...] = ()
    degraded_channels: tuple[str, ...] = ()
    usage: TokenUsage = TokenUsage()


class PromptContext(FrozenModel):
    """Everything a consumer needs for one LLM turn (API.md §6.1)."""

    injection_block: str
    facts: tuple[ScoredFact, ...] = ()
    citations: tuple[Citation, ...] = ()
    resolutions: tuple[Resolution, ...] = ()
    asker_summary: str | None = None
    usage: TokenUsage = TokenUsage()
    warnings: tuple[RecallWarning, ...] = ()

    def apply_citations(self, text: str) -> str:
        """Resolve echoed ``[mem:N]`` tags to markdown links; strip unknown residue."""
        import re

        by_ref = {c.ref.removeprefix("mem:"): c for c in self.citations}

        def replace(match: re.Match[str]) -> str:
            citation = by_ref.get(match.group(1).removeprefix("mem:"))
            if citation is None:
                return ""
            if citation.url:
                return f"[[{citation.ref}]]({citation.url})"
            return f"[{citation.ref}]"

        return re.sub(r"\[(mem:\d+)\]", replace, text)


CitationResolver = Callable[[str], Citation | None]


def render_citation_tag(index: int) -> str:
    """Prompt-facing tag for the fact injected at position ``index`` (1-based)."""
    return f"[mem:{index}]"


__all__ = [
    "CHANNELS_ALL",
    "CHANNELS_DEFAULT",
    "CHANNELS_DISCOVERY",
    "ChannelName",
    "ChannelSet",
    "Citation",
    "PromptContext",
    "RecallQuery",
    "RecallResult",
    "RecallWarning",
    "Resolution",
    "Scope",
    "ScoreComponents",
    "ScoredFact",
    "channels",
    "render_citation_tag",
]
