"""Retrieval boundary models: queries, scored facts, citations, prompt context."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum

from pydantic import Field

from icelake.models.common import FrozenModel, TokenUsage
from icelake.models.facts import FactRecord
from icelake.models.identity import Resolution


class Scope(StrEnum):
    """Candidate-space restriction, enforced in store queries."""

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


def discovery_pairs(
    asker_id: str,
    mentioned_ids: tuple[str, ...] = (),
    thread_participant_ids: tuple[str, ...] = (),
) -> tuple[tuple[str, str], ...]:
    """Unique unordered pairs among the asker and every related user.

    Related = ``mentioned_ids`` plus ``thread_participant_ids``, asker excluded,
    first-seen order. Empty when the turn is solo — callers then stay on
    ``CHANNELS_DEFAULT`` (no graph-hop tax). All pairs, not just asker-other,
    so m users share entity facts with one another (Alice-Bob, Alice-Carol,
    Bob-Carol) in one recall.
    """
    related = tuple(
        dict.fromkeys(
            uid for uid in (*mentioned_ids, *thread_participant_ids) if uid and uid != asker_id
        )
    )
    if not related:
        return ()
    members = (asker_id, *related)
    return tuple(
        (members[i], members[j]) for i in range(len(members)) for j in range(i + 1, len(members))
    )


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
    #: Raw L2 cross-encoder score when a reranker ran; ``None`` on first-stage only.
    rerank_score: float | None = None


class RecallQuery(FrozenModel):
    """Explicit retrieval request (API.md §6.2). Scope enforced store-side."""

    guild_id: str
    text: str | None = None
    subject_ids: tuple[str, ...] = ()
    pair_ids: tuple[tuple[str, str], ...] = ()
    entity_hint: str | None = None
    scope: Scope = Scope.SUBJECTS
    exclude_ids: tuple[str, ...] = ()
    top_k: int = 8
    max_per_subject: int = 4
    min_score: float = 0.0
    channels: ChannelSet | None = None
    as_of: datetime | None = None  # time-travel: what was known at this instant?


class Citation(FrozenModel):
    """Citation binding for an injected fact (``mem:N`` → jump link).

    Rich object resolved from a closed ID set; parsing/validation is owned by
    :class:`icelake.citations.Citations` (or ``PromptContext.apply_citations``).
    """

    ref: str
    fact_id: str
    url: str
    snippet: str = ""
    subject_id: str | None = None
    subject_name: str = ""
    #: Discord snowflakes for the primary source message (when known).
    message_id: str | None = None
    channel_id: str | None = None
    #: Final ranking score for this fact within the recall call.
    score: float | None = None
    #: L2 cross-encoder score when a reranker ran.
    rerank_score: float | None = None


class UsedCitation(Citation):
    """A citation the generator actually used, resolved from a closed ID set.

    ``claim`` is the matched text span (echoed tag or structured claim) that
    motivated the citation; empty when the consumer only passed an ID.
    """

    claim: str = ""


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
    degraded_channels: tuple[ChannelName, ...] = ()
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

    def resolve_used(self, text: str) -> tuple[UsedCitation, ...]:
        """Resolve generator output to the used subset of the closed citation set.

        Accepts echoed ``[mem:N]`` tags and a structured ``{"claims": [{"fact_id":
        ...}]}`` JSON block. Unknown IDs are dropped (never invented). Order is
        first-appearance in the text; duplicates collapse to one used citation.
        """
        used: list[UsedCitation] = []
        seen: set[str] = set()

        def add(citation: Citation, claim: str = "") -> None:
            if citation.ref in seen:
                return
            seen.add(citation.ref)
            used.append(UsedCitation(**citation.model_dump(), claim=claim))

        for match in re.finditer(r"\[(?:mem:)?(\d+)\]", text):
            citation = self._citation_by_ref(match.group(1))
            if citation is not None:
                add(citation, claim=match.group(0))

        for fact_id in _structured_claims(text):
            citation = self._citation_by_fact_id(fact_id)
            if citation is not None:
                add(citation, claim=fact_id)

        return tuple(used)

    def apply_citations(self, text: str) -> str:
        """Discord helper: weave used citations into markdown; strip residue.

        Delegates to :class:`icelake.citations.Citations` — the single parsing
        boundary. Echoed ``[mem:N]`` tags become jump links; unknown or
        citation-shaped tokens the model invented (``[mem:99]``, bare ``[5]``,
        self-written ``[1](url)`` links) are removed.
        """
        from icelake.citations import Citations

        return Citations(self.citations).apply(text)

    def _citation_by_ref(self, ref: str) -> Citation | None:
        key = ref.removeprefix("mem:")
        for citation in self.citations:
            if citation.ref.removeprefix("mem:") == key:
                return citation
        return None

    def _citation_by_fact_id(self, fact_id: str) -> Citation | None:
        for citation in self.citations:
            if citation.fact_id == fact_id:
                return citation
        return None


def _structured_claims(text: str) -> tuple[str, ...]:
    """Extract ``fact_id`` values from a structured claims JSON block, if any."""
    import json

    match = re.search(r"\{[^{}]*\"claims\"[^{}]*\[[^\]]*\][^{}]*\}", text, re.DOTALL)
    if match is None:
        return ()
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return ()
    claims = payload.get("claims")
    if not isinstance(claims, list):
        return ()
    out: list[str] = []
    for item in claims:
        if isinstance(item, dict) and isinstance(item.get("fact_id"), str):
            out.append(item["fact_id"])
    return tuple(out)


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
    "UsedCitation",
    "channels",
    "discovery_pairs",
    "render_citation_tag",
]
