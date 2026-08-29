"""Pure document<->model mapping for the MongoDB adapter.

Kept free of any driver imports so the conversion layer is unit-testable without a
running MongoDB server.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from icelake.models.facts import (
    Attribution,
    AttributionType,
    FactCategory,
    FactRecord,
    MemoryTier,
    ProfileSummary,
    SourceRef,
)
from icelake.models.graph import (
    EdgeKind,
    EntityKind,
    EntityRecord,
    LinkRow,
    NodeType,
    Polarity,
    RelationEdge,
)
from icelake.models.identity import AliasRecord, AliasSource


def _dt_out(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _dt_in(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def fact_to_doc(record: FactRecord) -> dict[str, Any]:
    attribution = {
        "type": record.attribution.type.value,
        "speaker_id": record.attribution.speaker_id,
        "speaker_name": record.attribution.speaker_name,
        "actor_id": record.attribution.actor_id,
    }
    return {
        "_id": record.id,
        "guild_id": record.guild_id,
        "subject_id": record.subject_id,
        "text": record.text,
        "text_normalized": record.text_normalized,
        "category": record.category.value,
        "confidence": record.confidence,
        "tier": record.tier.value,
        "scope": record.scope,
        "attribution": attribution,
        "occurrences": record.occurrences,
        "strength": record.strength,
        "last_reinforced_at": _dt_out(record.last_reinforced_at),
        "created_at": _dt_out(record.created_at),
        "updated_at": _dt_out(record.updated_at),
        "observed_at": _dt_out(record.observed_at),
        "valid_from": _dt_out(record.valid_from),
        "valid_until": _dt_out(record.valid_until),
        "supersedes_id": record.supersedes_id,
        "superseded_by_id": record.superseded_by_id,
        "citations": [c.model_dump(mode="json") for c in record.citations],
        "related_user_ids": list(record.related_user_ids),
        "entity_slugs": list(record.entity_slugs),
        "tags": list(record.tags),
        "expires_at": _dt_out(record.expires_at),
        "version": record.version,
    }


def fact_from_doc(doc: dict[str, Any]) -> FactRecord:
    attribution = doc.get("attribution") or {}
    return FactRecord(
        id=doc["_id"],
        guild_id=doc["guild_id"],
        subject_id=doc.get("subject_id"),
        text=doc["text"],
        text_normalized=doc.get("text_normalized", ""),
        category=FactCategory(doc.get("category", "general")),
        confidence=float(doc.get("confidence", 1.0)),
        tier=MemoryTier(doc.get("tier", "short_term")),
        scope=doc.get("scope", "user"),
        attribution=Attribution(
            type=AttributionType(attribution.get("type", "self")),
            speaker_id=attribution.get("speaker_id"),
            speaker_name=attribution.get("speaker_name"),
            actor_id=attribution.get("actor_id"),
        ),
        occurrences=int(doc.get("occurrences", 1)),
        strength=float(doc.get("strength", 1.0)),
        last_reinforced_at=_dt_in(doc.get("last_reinforced_at")),
        created_at=_dt_in(doc.get("created_at")),
        updated_at=_dt_in(doc.get("updated_at")),
        observed_at=_dt_in(doc.get("observed_at")),
        valid_from=_dt_in(doc.get("valid_from")),
        valid_until=_dt_in(doc.get("valid_until")),
        supersedes_id=doc.get("supersedes_id"),
        superseded_by_id=doc.get("superseded_by_id"),
        citations=tuple(SourceRef.model_validate(c) for c in doc.get("citations", [])),
        related_user_ids=tuple(doc.get("related_user_ids", [])),
        entity_slugs=tuple(doc.get("entity_slugs", [])),
        tags=tuple(doc.get("tags", [])),
        expires_at=_dt_in(doc.get("expires_at")),
        version=int(doc.get("version", 1)),
    )


def alias_to_doc(
    guild_id: str,
    alias_norm: str,
    user_id: str,
    source: AliasSource,
    weight: float,
    updated_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "guild_id": guild_id,
        "alias_norm": alias_norm,
        "user_id": user_id,
        "source": source.value,
        "weight": weight,
        "updated_at": _dt_out(updated_at),
    }


def alias_from_doc(doc: dict[str, Any]) -> AliasRecord:
    return AliasRecord(
        guild_id=doc["guild_id"],
        alias_norm=doc["alias_norm"],
        user_id=doc["user_id"],
        source=AliasSource(doc["source"]),
        weight=float(doc.get("weight", 0.5)),
        updated_at=_dt_in(doc.get("updated_at")),
    )


def link_to_doc(row: LinkRow) -> dict[str, Any]:
    return {
        "memory_id": row.memory_id,
        "guild_id": row.guild_id,
        "node_type": row.node_type.value,
        "node_id": row.node_id,
        "kind": row.kind.value,
        "created_at": _dt_out(row.created_at),
    }


def link_from_doc(doc: dict[str, Any]) -> LinkRow:
    return LinkRow(
        memory_id=doc["memory_id"],
        guild_id=doc["guild_id"],
        node_type=NodeType(doc["node_type"]),
        node_id=doc["node_id"],
        kind=EdgeKind(doc["kind"]),
        created_at=_dt_in(doc.get("created_at")),
    )


def relation_to_doc(edge: RelationEdge) -> dict[str, Any]:
    return {
        "guild_id": edge.guild_id,
        "src_type": edge.src_type.value,
        "src_id": edge.src_id,
        "dst_type": edge.dst_type.value,
        "dst_id": edge.dst_id,
        "verb": edge.verb,
        "polarity": edge.polarity.value,
        "weight": edge.weight,
        "occurrences": edge.occurrences,
        "confidence": edge.confidence,
        "evidence_ids": list(edge.evidence_fact_ids),
        "valid_from": _dt_out(edge.valid_from),
        "valid_until": _dt_out(edge.valid_until),
    }


def relation_from_doc(doc: dict[str, Any]) -> RelationEdge:
    return RelationEdge(
        guild_id=doc["guild_id"],
        src_type=NodeType(doc["src_type"]),
        src_id=doc["src_id"],
        dst_type=NodeType(doc["dst_type"]),
        dst_id=doc["dst_id"],
        verb=doc["verb"],
        polarity=Polarity(doc["polarity"]),
        weight=float(doc.get("weight", 0.0)),
        occurrences=int(doc.get("occurrences", 1)),
        confidence=float(doc.get("confidence", 0.5)),
        evidence_fact_ids=tuple(doc.get("evidence_ids", [])),
        valid_from=_dt_in(doc.get("valid_from")),
        valid_until=_dt_in(doc.get("valid_until")),
    )


RELATION_ID_FIELDS = ("guild_id", "src_type", "src_id", "dst_type", "dst_id", "verb")


def relation_business_id(edge: RelationEdge) -> dict[str, Any]:
    """Deterministic filter key for the active edge between a pair+verb."""
    return {
        "guild_id": edge.guild_id,
        "src_type": edge.src_type.value,
        "src_id": edge.src_id,
        "dst_type": edge.dst_type.value,
        "dst_id": edge.dst_id,
        "verb": edge.verb,
    }


def entity_to_doc(record: EntityRecord) -> dict[str, Any]:
    return {
        "guild_id": record.guild_id,
        "slug": record.slug,
        "name": record.name,
        "kind": record.kind,
        "aliases": list(record.aliases),
        "fact_count": record.fact_count,
        "linked_user_id": record.linked_user_id,
        "summary": record.summary,
    }


def entity_from_doc(doc: dict[str, Any]) -> EntityRecord:
    kind: EntityKind = doc.get("kind", "concept")
    return EntityRecord(
        guild_id=doc["guild_id"],
        slug=doc["slug"],
        name=doc["name"],
        kind=kind,
        aliases=tuple(doc.get("aliases", [])),
        fact_count=int(doc.get("fact_count", 0)),
        linked_user_id=doc.get("linked_user_id"),
        summary=doc.get("summary", ""),
    )


def summary_to_doc(summary: ProfileSummary) -> dict[str, Any]:
    return {
        "guild_id": summary.guild_id,
        "subject_key": summary.subject_id or "__server__",
        "text": summary.text,
        "generated_at": _dt_out(summary.generated_at),
        "source_fact_count": summary.source_fact_count,
    }


def summary_from_doc(doc: dict[str, Any]) -> ProfileSummary:
    key = doc.get("subject_key", "__server__")
    return ProfileSummary(
        guild_id=doc["guild_id"],
        subject_id=None if key == "__server__" else key,
        text=doc["text"],
        generated_at=_dt_in(doc.get("generated_at")),
        source_fact_count=int(doc.get("source_fact_count", 0)),
    )
