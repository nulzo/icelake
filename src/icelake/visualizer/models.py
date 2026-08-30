"""Wire schema for the graph explorer snapshot (HTML payload)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from icelake.models.common import FrozenModel, utc_now


class VizNodeType(StrEnum):
    USER = "user"
    ENTITY = "entity"
    SERVER = "server"


class VizEdgeKind(StrEnum):
    RELATION = "relation"
    IDENTITY = "identity"


class VizAlias(FrozenModel):
    alias: str
    source: str
    weight: float = 0.0


class VizCitation(FrozenModel):
    message_url: str = ""
    author_name: str = ""
    content_snippet: str = ""


class VizFact(FrozenModel):
    """Compact fact for the inspector; ids are resolvable back to ``dm_facts``."""

    id: str
    text: str
    category: str
    tier: str
    subject_id: str | None = None
    confidence: float = 1.0
    occurrences: int = 1
    active: bool = True
    entity_slugs: tuple[str, ...] = ()
    related_user_ids: tuple[str, ...] = ()
    citations: tuple[VizCitation, ...] = ()


class VizNode(FrozenModel):
    id: str
    type: VizNodeType
    label: str
    user_id: str | None = None
    entity_slug: str | None = None
    entity_kind: str | None = None
    linked_user_id: str | None = None
    aliases: tuple[VizAlias, ...] = ()
    fact_ids: tuple[str, ...] = ()
    search_text: str = ""
    summary: str = ""


class VizEdge(FrozenModel):
    id: str
    source: str
    target: str
    verb: str
    polarity: str
    weight: float = 0.0
    occurrences: int = 1
    confidence: float = 0.5
    evidence_fact_ids: tuple[str, ...] = ()
    kind: VizEdgeKind = VizEdgeKind.RELATION


class VizStats(FrozenModel):
    total_facts: int = 0
    active_facts: int = 0
    user_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    pending_messages: int = 0


class GraphSnapshot(FrozenModel):
    """Self-contained guild graph for the explorer. No incidence (``dm_links``) edges."""

    guild_id: str
    generated_at: datetime = Field(default_factory=utc_now)
    center: str | None = None
    depth: int | None = None
    stats: VizStats
    nodes: tuple[VizNode, ...] = ()
    edges: tuple[VizEdge, ...] = ()
    facts: tuple[VizFact, ...] = ()
