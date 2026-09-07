"""Retrieval boundary models: queries, scored facts, citations, prompt context."""

from __future__ import annotations

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

    Pure provenance data — the closed set a reply may cite. Registration and
    rendering are owned by :class:`icelake.citations.Citations`; claim mapping
    is owned by :func:`icelake.attribution.attribute`. The answer model never
    sees or writes these refs.
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
    "discovery_pairs",
]
