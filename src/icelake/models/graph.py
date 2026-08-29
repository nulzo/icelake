"""Knowledge-graph boundary models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from icelake.models.common import FrozenModel


class NodeType(StrEnum):
    USER = "user"
    ENTITY = "entity"


class EdgeKind(StrEnum):
    """Incidence kinds (fact↔node)."""

    SUBJECT_OF = "subject_of"
    SPEAKER_OF = "speaker_of"
    ABOUT_USER = "about_user"
    MENTIONED_WITH = "mentioned_with"
    ABOUT_ENTITY = "about_entity"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RelationVerb(StrEnum):
    """Known relation verbs with a polarity mapping (``polarity_for_verb``).

    The vocabulary is open by design: extraction may store verbs outside this
    set, and unknown verbs are NEUTRAL. Members are plain strings, so they
    work anywhere a verb string is accepted::

        ProposedRelation(verb=RelationVerb.CALLED_OUT, from_token=a, to_token=b)
        [e for e in edges if e.verb == RelationVerb.LIKES]
    """

    LIKES = "likes"
    LOVES = "loves"
    ENJOYS = "enjoys"
    PREFERS = "prefers"
    FRIENDS_WITH = "friends_with"
    FRIEND_OF = "friend_of"
    SUPPORTS = "supports"
    ADMIRES = "admires"
    COLLABORATES_WITH = "collaborates_with"
    TEAMMATE_OF = "teammate_of"
    FAN_OF = "fan_of"
    DISLIKES = "dislikes"
    HATES = "hates"
    CALLED_OUT = "called_out"
    ARGUES_WITH = "argues_with"
    DISTRUSTS = "distrusts"
    FEUDS_WITH = "feuds_with"
    DISAPPROVES_OF = "disapproves_of"
    AVOIDS = "avoids"
    KNOWS = "knows"


class RelationEdge(FrozenModel):
    """Typed node→node edge (layer 3): ``X -likes-> movies``, ``X -called_out-> Y``."""

    guild_id: str
    src_type: NodeType
    src_id: str
    dst_type: NodeType
    dst_id: str
    verb: str
    polarity: Polarity = Polarity.NEUTRAL
    weight: float = Field(default=0.0, ge=0.0)
    occurrences: int = Field(default=1, ge=1)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_fact_ids: tuple[str, ...] = ()
    valid_from: datetime | None = None
    valid_until: datetime | None = None


class StanceSummary(FrozenModel):
    """Aggregated positions on one entity node (Q5)."""

    entity_slug: str
    entity_name: str = ""
    positive: tuple[RelationEdge, ...] = ()
    negative: tuple[RelationEdge, ...] = ()
    other: tuple[RelationEdge, ...] = ()
    total_evidence: int = 0


class NeighborInfo(FrozenModel):
    """One hop-discovery result with its relation path for honest phrasing."""

    node_type: NodeType
    node_id: str
    strength: float = 0.0
    relation_path: tuple[str, ...] = ()
    evidence_fact_ids: tuple[str, ...] = ()


class EntityKind(StrEnum):
    """Closed vocabulary for named-entity nodes."""

    PERSON = "person"
    PLACE = "place"
    CONCEPT = "concept"
    ORG = "org"


class EntityRecord(FrozenModel):
    """Named-entity junction node accumulating stances from many members."""

    guild_id: str
    slug: str
    name: str
    kind: EntityKind = EntityKind.CONCEPT
    aliases: tuple[str, ...] = ()
    fact_count: int = 0
    linked_user_id: str | None = None
    summary: str = ""


class LinkRow(FrozenModel):
    """Incidence row (layer 2): one fact touching one node."""

    guild_id: str
    memory_id: str
    node_type: NodeType
    node_id: str
    kind: EdgeKind
    created_at: datetime | None = None


class SimilarUser(FrozenModel):
    """One ``graph.similar_users`` hit: a member and their Jaccard overlap score."""

    user_id: str
    score: float = Field(ge=0.0, le=1.0)


__all__ = [
    "EdgeKind",
    "EntityKind",
    "EntityRecord",
    "LinkRow",
    "NeighborInfo",
    "NodeType",
    "Polarity",
    "RelationEdge",
    "RelationVerb",
    "SimilarUser",
    "StanceSummary",
]
