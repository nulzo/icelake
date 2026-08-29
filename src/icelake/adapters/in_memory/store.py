"""Thread-safe in-memory MemoryStore. The reference backend and conformance baseline.

Implements every port method with dicts guarded by an asyncio lock; used by the test
suite, evals, and small single-process deployments that want zero dependencies.
"""

from __future__ import annotations

import asyncio
from contextlib import AbstractAsyncContextManager
from datetime import datetime

from icelake.lifecycle.prune import select_prune_victims_by_anchor
from icelake.models.admin import GuildStats, PurgeReport
from icelake.models.common import Page
from icelake.models.facts import (
    FactCategory,
    FactHistoryEntry,
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
from icelake.ports.store import NodeRef


class InMemoryStore:
    """Dict-backed implementation of :class:`~icelake.ports.MemoryStore`."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._facts: dict[tuple[str, str], FactRecord] = {}
        self._history: dict[str, list[FactHistoryEntry]] = {}
        self._aliases: dict[tuple[str, str], AliasRecord] = {}
        self._links: dict[str, list[LinkRow]] = {}
        self._relations: dict[tuple[str, str, str, str, str], RelationEdge] = {}
        self._entities: dict[tuple[str, str], EntityRecord] = {}
        self._entity_aliases: dict[tuple[str, str], str] = {}
        self._summaries: dict[tuple[str, str | None], ProfileSummary] = {}
        self._cursors: dict[tuple[str, str], str] = {}
        self._opt_outs: set[tuple[str, str]] = set()
        self.closed = False

    async def setup(self) -> None:
        pass

    def transaction(self) -> AbstractAsyncContextManager[None]:
        import contextlib

        return contextlib.nullcontext()

    async def close(self) -> None:
        self.closed = True

    async def ping(self) -> bool:
        return not self.closed

    # -- aliases ---------------------------------------------------------------

    async def upsert_alias(
        self,
        guild_id: str,
        alias_norm: str,
        user_id: str,
        source: AliasSource,
        weight: float,
    ) -> None:
        key = (guild_id, f"{alias_norm}\x00{user_id}")
        existing = self._aliases.get(key)
        if existing and existing.weight >= weight:
            return
        self._aliases[key] = AliasRecord(
            guild_id=guild_id,
            alias_norm=alias_norm,
            user_id=user_id,
            source=source,
            weight=weight,
            updated_at=datetime.now(),
        )

    async def resolve_alias_candidates(
        self,
        guild_id: str,
        alias_norm: str,
        limit: int = 8,
    ) -> tuple[AliasRecord, ...]:
        matches = [
            record
            for (gid, key_composite), record in self._aliases.items()
            if gid == guild_id and key_composite.split("\x00")[0] == alias_norm
        ]
        return tuple(sorted(matches, key=lambda r: -r.weight)[:limit])

    async def prefix_alias_candidates(
        self,
        guild_id: str,
        prefix_norm: str,
        limit: int = 8,
    ) -> tuple[AliasRecord, ...]:
        seen: dict[str, AliasRecord] = {}
        for (_gid, _composite), record in self._aliases.items():
            if record.guild_id != guild_id or not record.alias_norm.startswith(prefix_norm):
                continue
            current = seen.get(record.user_id)
            if current is None or record.weight > current.weight:
                seen[record.user_id] = record
        return tuple(sorted(seen.values(), key=lambda r: -r.weight)[:limit])

    async def aliases_for_user(
        self,
        guild_id: str,
        user_id: str,
    ) -> tuple[AliasRecord, ...]:
        return tuple(
            record
            for (gid, _), record in self._aliases.items()
            if gid == guild_id and record.user_id == user_id
        )

    async def delete_aliases_for_user(self, guild_id: str, user_id: str) -> int:
        doomed = [
            key
            for key, record in self._aliases.items()
            if key[0] == guild_id and record.user_id == user_id
        ]
        for key in doomed:
            del self._aliases[key]
        return len(doomed)

    # -- facts -------------------------------------------------------------------

    async def insert_fact(self, record: FactRecord) -> None:
        self._facts[(record.guild_id, record.id)] = record

    def _fact(self, guild_id: str, fact_id: str) -> FactRecord | None:
        return self._facts.get((guild_id, fact_id))

    async def get_fact(self, guild_id: str, fact_id: str) -> FactRecord | None:
        return self._fact(guild_id, fact_id)

    async def get_facts(
        self,
        guild_id: str,
        fact_ids: tuple[str, ...],
    ) -> tuple[FactRecord, ...]:
        found = [self._fact(guild_id, fid) for fid in fact_ids]
        return tuple(f for f in found if f is not None)

    async def find_duplicate(
        self,
        guild_id: str,
        subject_id: str | None,
        text_normalized: str,
    ) -> FactRecord | None:
        for record in self._facts.values():
            if record.guild_id != guild_id or record.subject_id != subject_id:
                continue
            if record.text_normalized == text_normalized and record.is_active:
                return record
        return None

    async def reinforce_fact(
        self,
        guild_id: str,
        fact_id: str,
        *,
        occurrences_delta: int,
        strength: float,
        last_reinforced_at: datetime,
        expires_at: datetime | None,
        tier: str,
        confidence: float,
        extra_citations: tuple[SourceRef, ...] = (),
    ) -> FactRecord | None:
        record = self._fact(guild_id, fact_id)
        if record is None:
            return None
        merged_citations = record.citations + tuple(extra_citations)
        updated = record.model_copy(
            update={
                "occurrences": record.occurrences + occurrences_delta,
                "strength": strength,
                "last_reinforced_at": last_reinforced_at,
                "expires_at": expires_at,
                "tier": MemoryTier(tier),
                "confidence": confidence,
                "citations": merged_citations[:8],
                "updated_at": last_reinforced_at,
                "version": record.version + 1,
            }
        )
        self._facts[(guild_id, fact_id)] = updated
        return updated

    async def transition_fact(
        self,
        guild_id: str,
        fact_id: str,
        *,
        valid_until: datetime | None = None,
        superseded_by_id: str | None = None,
        updated_at: datetime,
    ) -> FactRecord | None:
        record = self._fact(guild_id, fact_id)
        if record is None:
            return None
        update: dict[str, object] = {"updated_at": updated_at}
        if valid_until is not None:
            update["valid_until"] = valid_until
        if superseded_by_id is not None:
            update["superseded_by_id"] = superseded_by_id
        new_record = record.model_copy(
            update={
                **update,
                "version": record.version + 1,
            }
        )
        self._facts[(guild_id, fact_id)] = new_record
        return new_record

    async def update_fact_fields(
        self,
        guild_id: str,
        fact_id: str,
        *,
        text: str | None = None,
        text_normalized: str | None = None,
        category: FactCategory | None = None,
        confidence: float | None = None,
        tier: str | None = None,
        expires_at: datetime | None = None,
        updated_at: datetime,
    ) -> FactRecord | None:
        record = self._fact(guild_id, fact_id)
        if record is None:
            return None
        update: dict[str, object] = {"updated_at": updated_at}
        if text is not None:
            update["text"] = text
        if text_normalized is not None:
            update["text_normalized"] = text_normalized
        if category is not None:
            update["category"] = category
        if confidence is not None:
            update["confidence"] = confidence
        if tier is not None:
            update["tier"] = tier
        if expires_at is not None:
            update["expires_at"] = expires_at
        new_record = record.model_copy(
            update={
                **update,
                "version": record.version + 1,
            }
        )
        self._facts[(guild_id, fact_id)] = new_record
        return new_record

    @staticmethod
    def _active(record: FactRecord) -> bool:
        return record.valid_until is None and record.superseded_by_id is None

    async def list_facts(
        self,
        guild_id: str,
        *,
        subject_id: str | None,
        include_server: bool = False,
        active_only: bool = True,
        limit: int = 50,
        cursor: str | None = None,
    ) -> Page[FactRecord]:
        selected = []
        for record in self._facts.values():
            if record.guild_id != guild_id:
                continue
            matches_subject = record.subject_id == subject_id or (
                subject_id is not None and include_server and record.is_server_fact
            )
            if not matches_subject:
                continue
            if active_only and not self._active(record):
                continue
            selected.append(record)
        selected.sort(key=lambda r: r.id)
        start = 0
        if cursor:
            start = next((i for i, r in enumerate(selected) if r.id > cursor), len(selected))
        window = selected[start : start + limit]
        next_cursor = window[-1].id if len(window) == limit else None
        return Page(items=tuple(window), next_cursor=next_cursor)

    async def top_strength_facts(
        self,
        guild_id: str,
        *,
        subject_ids: tuple[str, ...] | None,
        server_only: bool = False,
        limit: int = 10,
    ) -> tuple[FactRecord, ...]:
        pool: list[FactRecord] = []
        for record in self._facts.values():
            if record.guild_id != guild_id or not self._active(record):
                continue
            if server_only and not record.is_server_fact:
                continue
            if not server_only and subject_ids is not None:
                touched = record.subject_id in subject_ids or any(
                    uid in record.related_user_ids for uid in subject_ids
                )
                if not touched:
                    continue
            pool.append(record)
        pool.sort(key=lambda r: (-r.strength, -r.confidence, r.id))
        return tuple(pool[:limit])

    async def search_facts_text(
        self,
        guild_id: str,
        query: str,
        *,
        subject_ids: tuple[str, ...] | None = None,
        server_only: bool = False,
        limit: int = 20,
        as_of: datetime | None = None,
    ) -> tuple[tuple[FactRecord, float], ...]:
        terms = [t for t in query.lower().split() if t]
        scored: list[tuple[FactRecord, float]] = []
        for record in self._facts.values():
            if record.guild_id != guild_id:
                continue
            if as_of is not None:
                start = record.valid_from
                end = record.valid_until
                if start is not None and start > as_of:
                    continue
                if end is not None and end <= as_of:
                    continue
            elif not self._active(record):
                continue
            if server_only and not record.is_server_fact:
                continue
            if not server_only and subject_ids is not None:
                touched = record.subject_id in subject_ids or any(
                    uid in record.related_user_ids for uid in subject_ids
                )
                if not touched:
                    continue
            slug_text = " ".join(record.entity_slugs)
            haystack = f"{record.text} {slug_text}".lower()
            hits = sum(1 for term in terms if term in haystack)
            if hits:
                scored.append((record, min(1.0, hits / max(1, len(terms)))))
        scored.sort(key=lambda pair: -pair[1])
        return tuple(scored[:limit])

    async def append_history(
        self,
        guild_id: str,
        fact_id: str,
        entry: FactHistoryEntry,
    ) -> None:
        self._history.setdefault(fact_id, []).append(entry)

    async def get_history(
        self,
        guild_id: str,
        fact_id: str,
    ) -> tuple[FactHistoryEntry, ...]:
        return tuple(self._history.get(fact_id, ()))

    # -- links ---------------------------------------------------------------

    async def add_links(self, rows: tuple[LinkRow, ...]) -> None:
        for row in rows:
            bucket = self._links.setdefault(row.memory_id, [])
            if all(
                existing.node_id != row.node_id or existing.kind != row.kind for existing in bucket
            ):
                bucket.append(row)

    async def links_for_node(
        self,
        guild_id: str,
        node_type: NodeType,
        node_id: str,
        *,
        kinds: tuple[EdgeKind, ...] | None = None,
        active_only: bool = True,
        limit: int = 100,
        as_of: datetime | None = None,
    ) -> tuple[tuple[LinkRow, FactRecord], ...]:
        results: list[tuple[LinkRow, FactRecord]] = []
        for rows in self._links.values():
            for row in rows:
                if row.guild_id != guild_id or row.node_type != node_type:
                    continue
                if row.node_id != node_id:
                    continue
                if kinds and row.kind not in kinds:
                    continue
                record = self._facts.get((guild_id, row.memory_id))
                if record is None:
                    continue
                if as_of is not None:
                    start = record.valid_from
                    end = record.valid_until
                    if start is not None and start > as_of:
                        continue
                    if end is not None and end <= as_of:
                        continue
                elif active_only and not self._active(record):
                    continue
                results.append((row, record))
        return tuple(results[:limit])

    async def nodes_for_fact(self, guild_id: str, memory_id: str) -> tuple[LinkRow, ...]:
        return tuple(self._links.get(memory_id, ()))

    # -- relations -------------------------------------------------------------

    @staticmethod
    def _edge_key(edge: RelationEdge) -> tuple[str, str, str, str, str]:
        return (edge.guild_id, edge.src_id, edge.dst_id, edge.verb, edge.polarity.value)

    async def upsert_relation(self, edge: RelationEdge) -> RelationEdge:
        key = self._edge_key(edge)
        existing = self._relations.get(key)
        if existing is None or existing.valid_until is not None:
            self._relations[key] = edge
            return edge
        evidence = dict.fromkeys(existing.evidence_fact_ids + edge.evidence_fact_ids)
        merged = existing.model_copy(
            update={
                "occurrences": existing.occurrences + 1,
                "weight": max(existing.weight, edge.weight) * 0.5
                + edge.weight * 0.5
                + 0.1 * (existing.occurrences + 1) ** -0.5,
                "confidence": max(existing.confidence, edge.confidence),
                "evidence_fact_ids": tuple(evidence)[-8:],
            }
        )
        self._relations[key] = merged
        return merged

    async def edges_between(
        self,
        guild_id: str,
        src: NodeRef,
        dst: NodeRef,
    ) -> tuple[RelationEdge, ...]:
        return tuple(
            edge
            for edge in self._relations.values()
            if edge.guild_id == guild_id
            and edge.src_id == src[1]
            and edge.src_type == src[0]
            and edge.dst_id == dst[1]
            and edge.dst_type == dst[0]
            and edge.valid_until is None
        )

    async def incident_edges(
        self,
        guild_id: str,
        node: NodeRef,
        *,
        limit: int = 50,
    ) -> tuple[RelationEdge, ...]:
        hits = [
            edge
            for edge in self._relations.values()
            if edge.guild_id == guild_id
            and edge.valid_until is None
            and (node[0], node[1]) in {(edge.src_type, edge.src_id), (edge.dst_type, edge.dst_id)}
        ]
        hits.sort(key=lambda e: -e.weight)
        return tuple(hits[:limit])

    async def drop_evidence_from_edges(
        self,
        guild_id: str,
        fact_id: str,
        until: datetime,
    ) -> int:
        changed = 0
        for key, edge in list(self._relations.items()):
            if edge.guild_id != guild_id or fact_id not in edge.evidence_fact_ids:
                continue
            remaining = tuple(fid for fid in edge.evidence_fact_ids if fid != fact_id)
            if remaining:
                self._relations[key] = edge.model_copy(
                    update={
                        "evidence_fact_ids": remaining,
                        "weight": edge.weight * 0.8,
                    }
                )
            else:
                self._relations[key] = edge.model_copy(update={"valid_until": until})
            changed += 1
        return changed

    async def entity_stance_edges(
        self,
        guild_id: str,
        entity_slug: str,
        *,
        polarity: Polarity | None = None,
        limit: int = 25,
    ) -> tuple[RelationEdge, ...]:
        hits = [
            edge
            for edge in self._relations.values()
            if edge.guild_id == guild_id
            and edge.valid_until is None
            and edge.dst_type is NodeType.ENTITY
            and edge.dst_id == entity_slug
            and (polarity is None or edge.polarity is polarity)
        ]
        hits.sort(key=lambda e: -e.weight)
        return tuple(hits[:limit])

    # -- entities ----------------------------------------------------------------

    async def upsert_entity(
        self,
        guild_id: str,
        slug: str,
        name: str,
        kind: EntityKind,
        aliases: tuple[str, ...] = (),
    ) -> EntityRecord:
        key = (guild_id, slug)
        existing = self._entities.get(key)
        merged_aliases = dict.fromkeys((existing.aliases if existing else ()) + aliases)
        record = EntityRecord(
            guild_id=guild_id,
            slug=slug,
            name=name,
            kind=kind,
            aliases=tuple(merged_aliases),
            fact_count=existing.fact_count if existing else 0,
            linked_user_id=existing.linked_user_id if existing else None,
            summary=existing.summary if existing else "",
        )
        self._entities[key] = record
        self._entity_aliases.setdefault((guild_id, name.lower()), slug)
        for alias in merged_aliases:
            self._entity_aliases.setdefault((guild_id, alias.lower()), slug)
        return record

    async def bump_entity_facts(self, guild_id: str, slug: str, delta: int = 1) -> None:
        key = (guild_id, slug)
        record = self._entities.get(key)
        if record:
            self._entities[key] = record.model_copy(
                update={
                    "fact_count": record.fact_count + delta,
                }
            )

    async def get_entity(self, guild_id: str, slug: str) -> EntityRecord | None:
        return self._entities.get((guild_id, slug))

    async def resolve_entity_alias(self, guild_id: str, alias_norm: str) -> str | None:
        return self._entity_aliases.get((guild_id, alias_norm.lower()))

    async def merge_entities(
        self,
        guild_id: str,
        from_slugs: tuple[str, ...],
        to_slug: str,
    ) -> int:
        moved = 0
        target = self._entities.get((guild_id, to_slug))
        for slug in from_slugs:
            source = self._entities.pop((guild_id, slug), None)
            if source is None:
                continue
            moved += 1
            if target is not None:
                target = target.model_copy(
                    update={
                        "fact_count": target.fact_count + source.fact_count,
                        "aliases": dict.fromkeys(target.aliases + source.aliases + (slug,)),
                    }
                )
        if target is not None:
            self._entities[(guild_id, to_slug)] = target
        return moved

    # -- summaries --------------------------------------------------------------

    async def get_cursor(self, guild_id: str, key: str) -> str | None:
        return self._cursors.get((guild_id, key))

    async def set_cursor(self, guild_id: str, key: str, value: str) -> None:
        self._cursors[(guild_id, key)] = value

    async def get_summary(
        self,
        guild_id: str,
        subject_id: str | None,
    ) -> ProfileSummary | None:
        return self._summaries.get((guild_id, subject_id))

    async def put_summary(self, summary: ProfileSummary) -> None:
        self._summaries[(summary.guild_id, summary.subject_id)] = summary

    async def delete_summary(self, guild_id: str, subject_id: str | None) -> int:
        return 1 if self._summaries.pop((guild_id, subject_id), None) else 0

    # -- governance ----------------------------------------------------------------

    async def set_opt_out(self, guild_id: str, user_id: str, opted_out: bool) -> None:
        key = (guild_id, user_id)
        if opted_out:
            self._opt_outs.add(key)
        else:
            self._opt_outs.discard(key)

    async def get_opt_out(self, guild_id: str, user_id: str) -> bool:
        return (guild_id, user_id) in self._opt_outs

    async def purge_user_data(self, guild_id: str, user_id: str, dry_run: bool) -> PurgeReport:
        facts_removed = sum(
            1
            for (gid, _), record in self._facts.items()
            if gid == guild_id and record.subject_id == user_id
        )
        links_removed = 0
        edges_removed = 0
        for rows in self._links.values():
            links_removed += sum(
                1
                for row in rows
                if row.guild_id == guild_id
                and (
                    (row.node_type is NodeType.USER and row.node_id == user_id)
                    or any(
                        rec is not None and rec.subject_id == user_id
                        for rec in [self._facts.get((guild_id, row.memory_id))]
                    )
                )
            )
        for key, edge in list(self._relations.items()):
            if edge.guild_id != guild_id:
                continue
            touches = (edge.src_type is NodeType.USER and edge.src_id == user_id) or (
                edge.dst_type is NodeType.USER and edge.dst_id == user_id
            )
            if touches:
                edges_removed += 1
                if not dry_run:
                    del self._relations[key]
        report = PurgeReport(
            guild_id=guild_id,
            subject_id=user_id,
            dry_run=dry_run,
            facts_removed=facts_removed,
            links_removed=links_removed,
            edges_removed=edges_removed,
            aliases_removed=len(await self.aliases_for_user(guild_id, user_id)),
            summaries_removed=1 if (guild_id, user_id) in self._summaries else 0,
            vectors_removed=facts_removed,
        )
        if dry_run:
            return report
        for (gid, _), record in list(self._facts.items()):
            if gid == guild_id and record.subject_id == user_id:
                self._facts.pop((gid, record.id))
        await self.delete_aliases_for_user(guild_id, user_id)
        self._summaries.pop((guild_id, user_id), None)
        self._opt_outs.discard((guild_id, user_id))
        return report

    async def import_guild(
        self,
        facts: tuple[FactRecord, ...],
        entities: tuple[EntityRecord, ...],
        relations: tuple[RelationEdge, ...],
    ) -> int:
        """Bulk restore from a MemoryExport."""
        for record in facts:
            await self.insert_fact(record)
        for entity in entities:
            key = (entity.guild_id, entity.slug)
            if key not in self._entities:
                self._entities[key] = entity
        return len(facts)

    async def sweep_expired(self, guild_id: str, now: datetime) -> int:
        changed = 0
        for key, record in list(self._facts.items()):
            if (
                record.guild_id == guild_id
                and record.expires_at is not None
                and record.expires_at <= now
                and record.is_active
            ):
                self._facts[key] = record.model_copy(update={"valid_until": now})
                changed += 1
        return changed

    async def prune_to_caps(
        self,
        guild_id: str,
        *,
        max_per_user: int,
        max_server: int,
        now: datetime,
    ) -> int:
        victims = select_prune_victims_by_anchor(
            tuple(record for record in self._facts.values() if record.guild_id == guild_id),
            max_per_user=max_per_user,
            max_server=max_server,
        )
        for victim in victims:
            self._facts[(guild_id, victim.id)] = victim.model_copy(update={"valid_until": now})
        return len(victims)

    async def apply_forgetting(
        self,
        guild_id: str,
        *,
        now: datetime,
        retention_floor: float,
    ) -> int:
        from icelake.lifecycle.strength import retention, should_forget

        forgotten = 0
        for key, record in list(self._facts.items()):
            if record.guild_id != guild_id or not record.is_active:
                continue
            last = record.last_reinforced_at or record.created_at
            if last is None:
                continue
            value = retention(
                last_reinforced_at=last,
                now=now,
                strength=record.strength,
            )
            manual = record.attribution.type.value == "manual"
            if should_forget(
                retention_value=value,
                tier=record.tier.value,
                manual=manual,
                forget_retention_floor=retention_floor,
            ):
                self._facts[key] = record.model_copy(update={"valid_until": now})
                forgotten += 1
        return forgotten

    async def touch_facts(
        self,
        guild_id: str,
        fact_ids: tuple[str, ...],
        *,
        accessed_at: datetime,
    ) -> int:
        touched = 0
        for fact_id in fact_ids:
            record = self._facts.get((guild_id, fact_id))
            if record is None:
                continue
            self._facts[(guild_id, fact_id)] = record.model_copy(
                update={
                    "last_reinforced_at": accessed_at,
                    "updated_at": accessed_at,
                }
            )
            touched += 1
        return touched

    async def export_guild(
        self, guild_id: str
    ) -> tuple[
        tuple[FactRecord, ...],
        tuple[EntityRecord, ...],
        tuple[RelationEdge, ...],
    ]:
        facts = tuple(r for r in self._facts.values() if r.guild_id == guild_id)
        entities = tuple(e for (gid, _), e in self._entities.items() if gid == guild_id)
        relations = tuple(e for e in self._relations.values() if e.guild_id == guild_id)
        return facts, entities, relations

    async def guild_stats(self, guild_id: str) -> GuildStats:
        facts = [r for r in self._facts.values() if r.guild_id == guild_id]
        by_tier: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        users: set[str] = set()
        for record in facts:
            by_tier[record.tier.value] = by_tier.get(record.tier.value, 0) + 1
            scope = "server" if record.is_server_fact else "user"
            by_scope[scope] = by_scope.get(scope, 0) + 1
            if record.subject_id:
                users.add(record.subject_id)
        relations = [e for e in self._relations.values() if e.guild_id == guild_id]
        entities = [e for (gid, _), e in self._entities.items() if gid == guild_id]
        return GuildStats(
            guild_id=guild_id,
            total_facts=len(facts),
            active_facts=sum(1 for r in facts if self._active(r)),
            by_tier=by_tier,
            by_scope=by_scope,
            user_count=len(users),
            entity_count=len(entities),
            relation_count=len(relations),
        )
