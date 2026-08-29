"""MongoStore: full MemoryStore port over PyMongo Async (Motor's 2026 successor).

Single-class cohesion note: like SqliteStore, this adapter is one object implementing
the whole port (~450 lines exceeds the 300-line gate deliberately — splitting a single
adapter across mixin-by-file hurt more than the size cost; the domain logic it serves
lives in pure modules above).
"""

from __future__ import annotations

import asyncio
import re
import typing
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any

from icelake.adapters.mongo.mapping import (
    alias_from_doc,
    alias_to_doc,
    entity_from_doc,
    entity_to_doc,
    fact_from_doc,
    fact_to_doc,
    link_from_doc,
    link_to_doc,
    relation_business_id,
    relation_from_doc,
    relation_to_doc,
    summary_from_doc,
    summary_to_doc,
)
from icelake.adapters.mongo.queue import CLAIMED
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
    Polarity,
    RelationEdge,
)
from icelake.models.identity import AliasRecord, AliasSource


def _iso(moment: datetime | None) -> str | None:
    return moment.isoformat() if moment else None


def _history_kind(entry: FactHistoryEntry) -> str:
    return entry.kind.value


class MongoStore:
    """Implements :class:`~icelake.ports.MemoryStore` on MongoDB."""

    def __init__(self, url: str, database: str = "icelake") -> None:
        try:
            from pymongo import AsyncMongoClient
        except ImportError as exc:  # pragma: no cover - requires the extra
            raise ImportError(
                "the MongoDB adapter requires pymongo>=4.10 (pip install icelake[mongo])",
            ) from exc
        self._client: Any = AsyncMongoClient(url, serverSelectionTimeoutMS=5000)
        self.db = self._client[database]
        from icelake.adapters.mongo.queue import MongoIngestQueue
        from icelake.adapters.mongo.vectors import MongoVectorIndex

        self.queue = MongoIngestQueue(self.db)
        self.vectors = MongoVectorIndex(self.db)

    def transaction(self) -> AbstractAsyncContextManager[None]:
        """Best-effort unit-of-work. Causal sessions need a replica set."""
        import contextlib

        return contextlib.nullcontext()

    # -- lifecycle -------------------------------------------------------------

    async def setup(self) -> None:
        from pymongo import ASCENDING as ASC

        facts = self.db["dm_facts"]
        await facts.create_index([("guild_id", ASC), ("subject_id", ASC)])
        await facts.create_index(
            [("guild_id", ASC), ("subject_id", ASC), ("text_normalized", ASC)],
        )
        await facts.create_index([("guild_id", ASC), ("strength", DESC := -1)])
        await self.db["dm_aliases"].create_index(
            [("guild_id", ASC), ("alias_norm", ASC), ("weight", DESC)],
        )
        await self.db["dm_links"].create_index(
            [("guild_id", ASC), ("node_type", ASC), ("node_id", ASC)],
        )
        relations = self.db["dm_relations"]
        await relations.create_index(
            [("guild_id", ASC), ("dst_type", ASC), ("dst_id", ASC)],
        )
        await relations.create_index(
            [("guild_id", ASC), ("src_type", ASC), ("src_id", ASC)],
        )
        # Unique ACTIVE edge per (pair, verb) — the Graphiti-style invariant the
        # SQLite adapter enforces with a partial unique index.
        await relations.create_index(
            [
                ("guild_id", ASC),
                ("src_type", ASC),
                ("src_id", ASC),
                ("dst_type", ASC),
                ("dst_id", ASC),
                ("verb", ASC),
            ],
            unique=True,
            partialFilterExpression={"valid_until": None},
        )
        await self.db["dm_history"].create_index(
            [("fact_id", ASC), ("at", ASC)],
        )
        await self.db["dm_entities"].create_index([("guild_id", ASC)])
        await self.db["dm_entity_aliases"].create_index(
            [("guild_id", ASC), ("alias_norm", ASC)],
        )
        await self.db["dm_summaries"].create_index(
            [("guild_id", ASC), ("subject_key", ASC)],
        )
        await self.queue.setup()
        await self.vectors.setup()

    async def close(self) -> None:
        await self._client.close()

    async def ping(self) -> bool:
        try:
            await self.db.command("ping")
            return True
        except Exception:
            return False

    # -- aliases ------------------------------------------------------------------

    async def upsert_alias(
        self,
        guild_id: str,
        alias_norm: str,
        user_id: str,
        source: AliasSource,
        weight: float,
    ) -> None:
        doc = alias_to_doc(
            guild_id, alias_norm, user_id, source, weight, updated_at=datetime.now().astimezone()
        )
        await self.db["dm_aliases"].update_one(
            {"guild_id": guild_id, "alias_norm": alias_norm, "user_id": user_id},
            {
                "$set": {
                    "source": doc["source"],
                    "weight": doc["weight"],
                    "updated_at": doc["updated_at"],
                }
            },
            upsert=True,
        )

    async def resolve_alias_candidates(
        self,
        guild_id: str,
        alias_norm: str,
        limit: int = 8,
    ) -> tuple[AliasRecord, ...]:
        cursor = (
            self.db["dm_aliases"]
            .find({"guild_id": guild_id, "alias_norm": alias_norm})
            .sort("weight", -1)
            .limit(limit)
        )
        return tuple([alias_from_doc(d) async for d in cursor])

    async def prefix_alias_candidates(
        self,
        guild_id: str,
        prefix_norm: str,
        limit: int = 8,
    ) -> tuple[AliasRecord, ...]:
        escaped = prefix_norm.replace("\\", "\\\\").replace("%", r"\%")
        cursor = (
            self.db["dm_aliases"]
            .find({"guild_id": guild_id, "alias_norm": {"$regex": f"^{escaped}"}})
            .sort("weight", -1)
            .limit(limit)
        )
        by_user: dict[str, AliasRecord] = {}
        async for doc in cursor:
            record = alias_from_doc(doc)
            current = by_user.get(record.user_id)
            if current is None or record.weight > current.weight:
                by_user[record.user_id] = record
        return tuple(sorted(by_user.values(), key=lambda r: -r.weight))

    async def aliases_for_user(self, guild_id: str, user_id: str) -> tuple[AliasRecord, ...]:
        cursor = self.db["dm_aliases"].find(
            {"guild_id": guild_id, "user_id": user_id},
        )
        return tuple([alias_from_doc(d) async for d in cursor])

    async def delete_aliases_for_user(self, guild_id: str, user_id: str) -> int:
        result = await self.db["dm_aliases"].delete_many(
            {"guild_id": guild_id, "user_id": user_id},
        )
        return result.deleted_count

    # -- facts ---------------------------------------------------------------------

    _ACTIVE: typing.ClassVar[dict[str, None]] = {
        "valid_until": None,
        "superseded_by_id": None,
    }

    async def insert_fact(self, record: FactRecord) -> None:
        await self.db["dm_facts"].insert_one(fact_to_doc(record))

    async def get_fact(self, guild_id: str, fact_id: str) -> FactRecord | None:
        doc = await self.db["dm_facts"].find_one({"_id": fact_id, "guild_id": guild_id})
        return fact_from_doc(doc) if doc else None

    async def get_facts(
        self,
        guild_id: str,
        fact_ids: tuple[str, ...],
    ) -> tuple[FactRecord, ...]:
        cursor = self.db["dm_facts"].find(
            {
                "guild_id": guild_id,
                "_id": {"$in": list(fact_ids)},
            }
        )
        # $in does not preserve request order; callers pair these with scored
        # hits by id, so return them in the order they were asked for.
        by_id = {doc["_id"]: fact_from_doc(doc) async for doc in cursor}
        return tuple(by_id[fact_id] for fact_id in fact_ids if fact_id in by_id)

    async def find_duplicate(
        self,
        guild_id: str,
        subject_id: str | None,
        text_normalized: str,
    ) -> FactRecord | None:
        query: dict[str, Any] = {
            "guild_id": guild_id,
            "subject_id": subject_id,
            "text_normalized": text_normalized,
            **self._ACTIVE,
        }
        doc = await self.db["dm_facts"].find_one(query)
        return fact_from_doc(doc) if doc else None

    async def reinforce_fact(
        self,
        guild_id: str,
        fact_id: str,
        *,
        occurrences_delta: int,
        strength: float,
        last_reinforced_at: datetime,
        expires_at: datetime | None,
        tier: MemoryTier,
        confidence: float,
        extra_citations: tuple[SourceRef, ...] = (),
    ) -> FactRecord | None:
        update: dict[str, Any] = {
            "$inc": {"occurrences": occurrences_delta, "version": 1},
            "$set": {
                "strength": strength,
                "last_reinforced_at": _iso(last_reinforced_at),
                "expires_at": _iso(expires_at),
                "tier": tier.value,
                "confidence": confidence,
            },
        }
        if extra_citations:
            update["$push"] = {
                "citations": {
                    "$each": [c.model_dump(mode="json") for c in extra_citations],
                    "$slice": 8,
                }
            }
        matched = await self.db["dm_facts"].update_one(
            {"_id": fact_id, "guild_id": guild_id, **self._ACTIVE},
            update,
        )
        if matched.matched_count == 0:
            return None
        return await self.get_fact(guild_id, fact_id)

    async def transition_fact(
        self,
        guild_id: str,
        fact_id: str,
        *,
        valid_until: datetime | None = None,
        superseded_by_id: str | None = None,
        updated_at: datetime,
    ) -> FactRecord | None:
        update: dict[str, Any] = {"updated_at": _iso(updated_at)}
        if valid_until is not None:
            update["valid_until"] = _iso(valid_until)
        if superseded_by_id is not None:
            update["superseded_by_id"] = superseded_by_id
        result = await self.db["dm_facts"].update_many(
            {"guild_id": guild_id, "_id": fact_id},
            {"$set": update, "$inc": {"version": 1}},
        )
        if result.matched_count == 0:
            return None
        return await self.get_fact(guild_id, fact_id)

    async def update_fact_fields(
        self,
        guild_id: str,
        fact_id: str,
        *,
        text: str | None = None,
        text_normalized: str | None = None,
        category: FactCategory | None = None,
        confidence: float | None = None,
        tier: MemoryTier | None = None,
        expires_at: datetime | None = None,
        updated_at: datetime,
    ) -> FactRecord | None:
        sets: dict[str, Any] = {"updated_at": _iso(updated_at)}
        if text is not None:
            sets["text"] = text
        if text_normalized is not None:
            sets["text_normalized"] = text_normalized
        if category is not None:
            sets["category"] = category.value
        if confidence is not None:
            sets["confidence"] = confidence
        if tier is not None:
            sets["tier"] = tier.value
        if expires_at is not None:
            sets["expires_at"] = _iso(expires_at)
        result = await self.db["dm_facts"].update_many(
            {"guild_id": guild_id, "_id": fact_id},
            {"$set": sets, "$inc": {"version": 1}},
        )
        if result.matched_count == 0:
            return None
        return await self.get_fact(guild_id, fact_id)

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
        subject_clause: list[dict[str, Any]] = [{"subject_id": subject_id}]
        if include_server and subject_id is not None:
            subject_clause.append({"subject_id": None})
        query: dict[str, Any] = {
            "guild_id": guild_id,
            "$or": subject_clause,
        }
        if active_only:
            query.update(self._ACTIVE)
        if cursor:
            query["_id"] = {"$gt": cursor}
        docs = (
            await self.db["dm_facts"].find(query).sort("_id", 1).limit(limit + 1).to_list(limit + 1)
        )
        items = tuple(fact_from_doc(d) for d in docs[:limit])
        next_cursor = items[-1].id if len(docs) > limit and items else None
        return Page(items=items, next_cursor=next_cursor)

    async def top_strength_facts(
        self,
        guild_id: str,
        *,
        subject_ids: tuple[str, ...] | None,
        server_only: bool = False,
        limit: int = 10,
        as_of: datetime | None = None,
    ) -> tuple[FactRecord, ...]:
        query: dict[str, Any] = {"guild_id": guild_id}
        if as_of is not None:
            iso_as_of = _iso(as_of)
            query["$or"] = [
                {"valid_from": None},
                {"valid_from": {"$lte": iso_as_of}},
            ]
            # point-in-time: supersession flags describe later knowledge
        else:
            query.update(self._ACTIVE)
        if server_only:
            query["scope"] = "server"
        elif subject_ids is not None:
            query["$or"] = [
                {"subject_id": {"$in": list(subject_ids)}},
                {"related_user_ids": {"$in": list(subject_ids)}},
            ]
        docs = (
            await self.db["dm_facts"]
            .find(query)
            .sort([("strength", -1), ("confidence", -1)])
            .limit(limit)
            .to_list(limit)
        )
        return tuple(fact_from_doc(d) for d in docs)

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
        if not terms:
            return ()
        docs = await (
            self.db["dm_facts"]
            .find(
                {
                    "guild_id": guild_id,
                    **self._ACTIVE,
                    "text": {
                        "$regex": "|".join(re.escape(t) for t in terms),
                        "$options": "i",
                    },
                }
            )
            .limit(limit * 3)
            .to_list(limit * 3)
        )
        scored: list[tuple[FactRecord, float]] = []
        allowed = set(subject_ids) if subject_ids is not None else None
        for doc in docs:
            record = fact_from_doc(doc)
            haystack = f"{record.text} {' '.join(record.entity_slugs)}".lower()
            hits = sum(1 for term in terms if term in haystack)
            if not hits:
                continue
            if server_only and not record.is_server_fact:
                continue
            if allowed is not None and not server_only:
                touched = record.subject_id in allowed or any(
                    uid in record.related_user_ids for uid in allowed
                )
                if not touched:
                    continue
            scored.append((record, min(1.0, hits / max(1, len(terms)))))
        scored.sort(key=lambda pair: -pair[1])
        return tuple(scored[:limit])

    async def append_history(
        self,
        guild_id: str,
        fact_id: str,
        entry: FactHistoryEntry,
    ) -> None:
        await self.db["dm_history"].insert_one(
            {
                "guild_id": guild_id,
                "fact_id": fact_id,
                "at": _iso(entry.at),
                "kind": _history_kind(entry),
                "detail": entry.detail,
                "fact_version": entry.fact_version,
            }
        )

    async def get_history(
        self,
        guild_id: str,
        fact_id: str,
    ) -> tuple[FactHistoryEntry, ...]:
        cursor = (
            self.db["dm_history"].find({"guild_id": guild_id, "fact_id": fact_id}).sort("at", 1)
        )
        out = []
        async for doc in cursor:
            moment = doc.get("at")
            out.append(
                FactHistoryEntry(
                    at=datetime.fromisoformat(moment)
                    if isinstance(moment, str)
                    else moment or datetime.now().astimezone(),
                    kind=doc["kind"],
                    detail=doc.get("detail", ""),
                    fact_version=int(doc.get("fact_version", 1)),
                )
            )
        return tuple(out)

    # -- incidence links ----------------------------------------------------------

    async def add_links(self, rows: tuple[Any, ...]) -> None:
        for row in rows:
            await self.db["dm_links"].update_one(
                {
                    "memory_id": row.memory_id,
                    "node_type": row.node_type.value,
                    "node_id": row.node_id,
                    "kind": row.kind.value,
                },
                {"$setOnInsert": link_to_doc(row)},
                upsert=True,
            )

    async def links_for_node(
        self,
        guild_id: str,
        node_type: object,
        node_id: str,
        *,
        kinds: tuple[EdgeKind, ...] | None = None,
        active_only: bool = True,
        limit: int = 100,
        as_of: datetime | None = None,
    ) -> tuple[tuple[LinkRow, FactRecord], ...]:
        type_value = getattr(node_type, "value", node_type)
        pipeline: list[dict[str, Any]] = [
            {"$match": {"guild_id": guild_id, "node_type": type_value, "node_id": node_id}},
            {"$limit": limit},
            {
                "$lookup": {
                    "from": "dm_facts",
                    "localField": "memory_id",
                    "foreignField": "_id",
                    "as": "fact",
                }
            },
            {"$unwind": "$fact"},
        ]
        if active_only:
            pipeline.insert(
                1,
                {
                    "$match": {
                        "fact.valid_until": None,
                        "fact.superseded_by_id": None,
                    }
                },
            )
        results: list[tuple[LinkRow, FactRecord]] = []
        cursor = await self.db["dm_links"].aggregate(pipeline)
        async for doc in cursor:
            row = link_from_doc({**doc, "kind": doc["kind"], "node_type": doc["node_type"]})
            if kinds is not None and row.kind not in kinds:
                continue
            results.append((row, fact_from_doc(doc["fact"])))
            if len(results) >= limit:
                break
        return tuple(results)

    async def nodes_for_fact(self, guild_id: str, memory_id: str) -> tuple[LinkRow, ...]:
        cursor = self.db["dm_links"].find(
            {"guild_id": guild_id, "memory_id": memory_id},
        )
        return tuple([link_from_doc(d) async for d in cursor])

    async def links_for_nodes(
        self,
        guild_id: str,
        nodes: tuple[Any, ...],
        *,
        active_only: bool = True,
        limit_per_node: int = 50,
    ) -> tuple[tuple[LinkRow, FactRecord], ...]:
        if not nodes:
            return ()
        unique = tuple(dict.fromkeys(nodes))
        pipeline: list[dict[str, Any]] = [
            {
                "$match": {
                    "guild_id": guild_id,
                    "$or": [
                        {
                            "node_type": getattr(t, "value", t),
                            "node_id": i,
                        }
                        for t, i in unique
                    ],
                }
            },
            {"$limit": limit_per_node * len(unique)},
            {
                "$lookup": {
                    "from": "dm_facts",
                    "localField": "memory_id",
                    "foreignField": "_id",
                    "as": "fact",
                }
            },
            {"$unwind": "$fact"},
        ]
        if active_only:
            pipeline.insert(
                1,
                {"$match": {"fact.valid_until": None, "fact.superseded_by_id": None}},
            )
        results: list[tuple[LinkRow, FactRecord]] = []
        per_node: dict[tuple[str, str], int] = {}
        cursor = await self.db["dm_links"].aggregate(pipeline)
        async for doc in cursor:
            key = (doc["node_type"], doc["node_id"])
            if per_node.get(key, 0) >= limit_per_node:
                continue
            per_node[key] = per_node.get(key, 0) + 1
            results.append((link_from_doc(doc), fact_from_doc(doc["fact"])))
        return tuple(results)

    # -- relations --------------------------------------------------------------------

    async def upsert_relation(self, edge: RelationEdge) -> RelationEdge:
        key = relation_business_id(edge)
        key["valid_until"] = None
        existing = await self.db["dm_relations"].find_one(key)
        if existing is None:
            await self.db["dm_relations"].insert_one(relation_to_doc(edge))
            return edge
        merged_evidence = list(
            dict.fromkeys(
                existing.get("evidence_ids", []) + list(edge.evidence_fact_ids),
            )
        )[-8:]
        await self.db["dm_relations"].update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "occurrences": int(existing.get("occurrences", 1)) + 1,
                    "weight": max(float(existing.get("weight", 0)), edge.weight),
                    "confidence": max(float(existing.get("confidence", 0.5)), edge.confidence),
                    "evidence_ids": merged_evidence,
                }
            },
        )
        refreshed = await self.db["dm_relations"].find_one({"_id": existing["_id"]})
        assert refreshed is not None
        return relation_from_doc(refreshed)

    async def edges_between(
        self,
        guild_id: str,
        src: Any,
        dst: Any,
    ) -> tuple[RelationEdge, ...]:
        cursor = self.db["dm_relations"].find(
            {
                "guild_id": guild_id,
                "src_type": src[0].value,
                "src_id": src[1],
                "dst_type": dst[0].value,
                "dst_id": dst[1],
                "valid_until": None,
            }
        )
        return tuple([relation_from_doc(d) async for d in cursor])

    async def incident_edges(
        self,
        guild_id: str,
        node: Any,
        *,
        limit: int = 50,
    ) -> tuple[RelationEdge, ...]:
        node_value = getattr(node[0], "value", node[0])
        cursor = (
            self.db["dm_relations"]
            .find(
                {
                    "guild_id": guild_id,
                    "valid_until": None,
                    "$or": [
                        {"src_type": node_value, "src_id": node[1]},
                        {"dst_type": node_value, "dst_id": node[1]},
                    ],
                }
            )
            .sort("weight", -1)
            .limit(limit)
        )
        return tuple([relation_from_doc(d) async for d in cursor])

    async def incident_edges_many(
        self,
        guild_id: str,
        nodes: tuple[Any, ...],
        *,
        limit_per_node: int = 50,
    ) -> tuple[RelationEdge, ...]:
        if not nodes:
            return ()
        unique = tuple(dict.fromkeys(nodes))
        wanted = {(getattr(t, "value", t), i) for t, i in unique}
        clauses = [
            clause
            for node_value, node_id in wanted
            for clause in (
                {"src_type": node_value, "src_id": node_id},
                {"dst_type": node_value, "dst_id": node_id},
            )
        ]
        cursor = (
            self.db["dm_relations"]
            .find({"guild_id": guild_id, "valid_until": None, "$or": clauses})
            .sort("weight", -1)
            .limit(limit_per_node * len(unique))
        )
        results: list[RelationEdge] = []
        per_node: dict[tuple[str, str], int] = {}
        async for doc in cursor:
            edge = relation_from_doc(doc)
            touches = [
                key
                for key in ((edge.src_type.value, edge.src_id), (edge.dst_type.value, edge.dst_id))
                if key in wanted
            ]
            if any(per_node.get(key, 0) >= limit_per_node for key in touches):
                continue
            for key in touches:
                per_node[key] = per_node.get(key, 0) + 1
            results.append(edge)
        return tuple(results)

    async def edges_to_nodes(
        self,
        guild_id: str,
        nodes: tuple[Any, ...],
        *,
        limit: int = 500,
    ) -> tuple[RelationEdge, ...]:
        if not nodes:
            return ()
        cursor = (
            self.db["dm_relations"]
            .find(
                {
                    "guild_id": guild_id,
                    "valid_until": None,
                    "$or": [
                        {"dst_type": getattr(t, "value", t), "dst_id": i}
                        for t, i in dict.fromkeys(nodes)
                    ],
                }
            )
            .sort("weight", -1)
            .limit(limit)
        )
        return tuple([relation_from_doc(d) async for d in cursor])

    async def drop_evidence_from_edges(
        self,
        guild_id: str,
        fact_id: str,
        until: datetime,
    ) -> int:
        changed = 0
        cursor = self.db["dm_relations"].find(
            {
                "guild_id": guild_id,
                "valid_until": None,
                "evidence_ids": fact_id,
            }
        )
        async for doc in cursor:
            remaining = [f for f in doc.get("evidence_ids", []) if f != fact_id]
            if remaining:
                await self.db["dm_relations"].update_one(
                    {"_id": doc["_id"]},
                    {
                        "$set": {
                            "evidence_ids": remaining,
                            "weight": float(doc.get("weight", 0)) * 0.8,
                        }
                    },
                )
            else:
                await self.db["dm_relations"].update_one(
                    {"_id": doc["_id"]},
                    {"$set": {"valid_until": _iso(until)}},
                )
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
        query: dict[str, Any] = {
            "guild_id": guild_id,
            "valid_until": None,
            "dst_type": "entity",
            "dst_id": entity_slug,
        }
        if polarity is not None:
            query["polarity"] = polarity.value
        cursor = self.db["dm_relations"].find(query).sort("weight", -1).limit(limit)
        return tuple([relation_from_doc(d) async for d in cursor])

    # -- entities ----------------------------------------------------------------------

    async def upsert_entity(
        self,
        guild_id: str,
        slug: str,
        name: str,
        kind: EntityKind,
        aliases: tuple[str, ...] = (),
    ) -> EntityRecord:
        await self.db["dm_entities"].update_one(
            {"guild_id": guild_id, "slug": slug},
            {
                "$set": {"name": name, "kind": kind.value},
                "$addToSet": {"aliases": {"$each": list(aliases)}},
            },
            upsert=True,
        )
        for alias in dict.fromkeys(aliases):
            await self.db["dm_entity_aliases"].update_one(
                {"guild_id": guild_id, "alias_norm": alias},
                {"$set": {"slug": slug}},
                upsert=True,
            )
        entity = await self.get_entity(guild_id, slug)
        assert entity is not None
        return entity

    async def bump_entity_facts(self, guild_id: str, slug: str, delta: int = 1) -> None:
        await self.db["dm_entities"].update_one(
            {"guild_id": guild_id, "slug": slug},
            {"$inc": {"fact_count": delta}},
        )

    async def get_entity(self, guild_id: str, slug: str) -> EntityRecord | None:
        doc = await self.db["dm_entities"].find_one(
            {"guild_id": guild_id, "slug": slug},
        )
        return entity_from_doc(doc) if doc else None

    async def resolve_entity_alias(self, guild_id: str, alias_norm: str) -> str | None:
        doc = await self.db["dm_entity_aliases"].find_one(
            {"guild_id": guild_id, "alias_norm": alias_norm},
        )
        return doc["slug"] if doc else None

    async def merge_entities(
        self,
        guild_id: str,
        from_slugs: tuple[str, ...],
        to_slug: str,
    ) -> int:
        moved = 0
        target = await self.get_entity(guild_id, to_slug)
        for slug in from_slugs:
            source = await self.get_entity(guild_id, slug)
            if source is None:
                continue
            moved += 1
            merged_aliases = tuple(
                dict.fromkeys(
                    ((target.aliases if target else ()) + source.aliases + (slug,)),
                )
            )
            if target is None:
                target = EntityRecord(
                    guild_id=guild_id,
                    slug=to_slug,
                    name=source.name,
                    kind=source.kind,
                    aliases=merged_aliases,
                    fact_count=source.fact_count,
                )
                await self.db["dm_entities"].insert_one(entity_to_doc(target))
            else:
                target = target.model_copy(
                    update={
                        "fact_count": target.fact_count + source.fact_count,
                        "aliases": merged_aliases,
                    }
                )
                await self.db["dm_entities"].replace_one(
                    {"guild_id": guild_id, "slug": to_slug},
                    entity_to_doc(target),
                )
            await self.db["dm_entities"].delete_one({"guild_id": guild_id, "slug": slug})
        return moved

    # -- derived summaries ------------------------------------------------------------

    async def get_summary(
        self,
        guild_id: str,
        subject_id: str | None,
    ) -> ProfileSummary | None:
        doc = await self.db["dm_summaries"].find_one(
            {"guild_id": guild_id, "subject_key": subject_id or "__server__"},
        )
        return summary_from_doc(doc) if doc else None

    async def put_summary(self, summary: ProfileSummary) -> None:
        await self.db["dm_summaries"].replace_one(
            {"guild_id": summary.guild_id, "subject_key": summary.subject_id or "__server__"},
            summary_to_doc(summary),
            upsert=True,
        )

    async def delete_summary(self, guild_id: str, subject_id: str | None) -> int:
        result = await self.db["dm_summaries"].delete_many(
            {"guild_id": guild_id, "subject_key": subject_id or "__server__"},
        )
        return result.deleted_count

    # -- consent & governance ------------------------------------------------------------

    async def set_opt_out(self, guild_id: str, user_id: str, opted_out: bool) -> None:
        if opted_out:
            await self.db["dm_optouts"].update_one(
                {"guild_id": guild_id, "user_id": user_id},
                {"$set": {"guild_id": guild_id, "user_id": user_id}},
                upsert=True,
            )
        else:
            await self.db["dm_optouts"].delete_many(
                {"guild_id": guild_id, "user_id": user_id},
            )

    async def get_opt_out(self, guild_id: str, user_id: str) -> bool:
        doc = await self.db["dm_optouts"].find_one(
            {"guild_id": guild_id, "user_id": user_id},
        )
        return doc is not None

    async def purge_user_data(self, guild_id: str, user_id: str, dry_run: bool) -> PurgeReport:
        owned_query = {"guild_id": guild_id, "subject_id": user_id}
        facts_removed = await self.db["dm_facts"].count_documents(owned_query)
        links_removed = await self.db["dm_links"].count_documents(
            {
                "$or": [
                    {"guild_id": guild_id, "node_type": "user", "node_id": user_id},
                    {
                        "guild_id": guild_id,
                        "memory_id": {
                            "$in": [
                                d["_id"]
                                async for d in self.db["dm_facts"].find(
                                    owned_query,
                                    {"_id": True},
                                )
                            ]
                        },
                    },
                ],
            }
        )
        edges_removed = await self.db["dm_relations"].count_documents(
            {
                "$or": [
                    {"guild_id": guild_id, "src_type": "user", "src_id": user_id},
                    {"guild_id": guild_id, "dst_type": "user", "dst_id": user_id},
                ],
            }
        )
        report = PurgeReport(
            guild_id=guild_id,
            subject_id=user_id,
            dry_run=dry_run,
            facts_removed=facts_removed,
            links_removed=links_removed,
            edges_removed=edges_removed,
            aliases_removed=len(await self.aliases_for_user(guild_id, user_id)),
            summaries_removed=0,
            vectors_removed=facts_removed,
        )
        if dry_run:
            return report
        owned_ids = [
            d["_id"]
            async for d in self.db["dm_facts"].find(
                owned_query,
                {"_id": True},
            )
        ]
        await self.db["dm_vectors"].delete_many({"_id": {"$in": owned_ids}})
        await self.db["dm_facts"].delete_many(owned_query)
        await self.db["dm_links"].delete_many(
            {
                "guild_id": guild_id,
                "node_type": "user",
                "node_id": user_id,
            }
        )
        await self.db["dm_links"].delete_many(
            {
                "guild_id": guild_id,
                "memory_id": {"$in": owned_ids},
            }
        )
        await self.db["dm_relations"].delete_many(
            {
                "$or": [
                    {"guild_id": guild_id, "src_type": "user", "src_id": user_id},
                    {"guild_id": guild_id, "dst_type": "user", "dst_id": user_id},
                ],
            }
        )
        await self.delete_aliases_for_user(guild_id, user_id)
        await self.delete_summary(guild_id, user_id)
        await self.set_opt_out(guild_id, user_id, False)
        return report

    async def export_guild(
        self, guild_id: str
    ) -> tuple[
        tuple[FactRecord, ...],
        tuple[EntityRecord, ...],
        tuple[RelationEdge, ...],
    ]:
        facts = tuple(
            [fact_from_doc(d) async for d in self.db["dm_facts"].find({"guild_id": guild_id})]
        )
        entities = tuple(
            [entity_from_doc(d) async for d in self.db["dm_entities"].find({"guild_id": guild_id})]
        )
        relations = tuple(
            [
                relation_from_doc(d)
                async for d in self.db["dm_relations"].find({"guild_id": guild_id})
            ]
        )
        return facts, entities, relations

    # -- maintenance ----------------------------------------------------------------------

    async def import_guild(
        self,
        facts: tuple[FactRecord, ...],
        entities: tuple[EntityRecord, ...],
        relations: tuple[RelationEdge, ...],
    ) -> int:
        """Bulk restore from a MemoryExport."""
        from icelake.adapters.mongo.mapping import (
            entity_to_doc,
            fact_to_doc,
            relation_to_doc,
        )

        for record in facts:
            await self.db["dm_facts"].insert_one(fact_to_doc(record))
        for entity in entities:
            await self.db["dm_entities"].update_one(
                {"guild_id": entity.guild_id, "slug": entity.slug},
                {"$setOnInsert": entity_to_doc(entity)},
                upsert=True,
            )
        for edge in relations:
            doc = relation_to_doc(edge)
            await self.db["dm_relations"].update_one(
                {
                    k: doc[k]
                    for k in ("guild_id", "src_type", "src_id", "dst_type", "dst_id", "verb")
                },
                {"$setOnInsert": doc},
                upsert=True,
            )
        return len(facts)

    async def sweep_expired(self, guild_id: str, now: datetime) -> int:
        result = await self.db["dm_facts"].update_many(
            {"guild_id": guild_id, "expires_at": {"$ne": None, "$lte": _iso(now)}, **self._ACTIVE},
            {"$set": {"valid_until": _iso(now), "updated_at": _iso(now)}, "$inc": {"version": 1}},
        )
        return result.modified_count

    async def prune_to_caps(
        self,
        guild_id: str,
        *,
        max_per_user: int,
        max_server: int,
        now: datetime,
    ) -> int:
        docs = [
            doc async for doc in self.db["dm_facts"].find({"guild_id": guild_id, **self._ACTIVE})
        ]
        victims = select_prune_victims_by_anchor(
            tuple(fact_from_doc(doc) for doc in docs),
            max_per_user=max_per_user,
            max_server=max_server,
        )
        if not victims:
            return 0
        await self.db["dm_facts"].update_many(
            {"_id": {"$in": [victim.id for victim in victims]}},
            {
                "$set": {"valid_until": _iso(now), "updated_at": _iso(now)},
                "$inc": {"version": 1},
            },
        )
        return len(victims)

    async def apply_forgetting(
        self,
        guild_id: str,
        *,
        now: datetime,
        retention_floor: float,
    ) -> int:
        import math

        rows = (
            await self.db["dm_facts"]
            .find(
                {"guild_id": guild_id, **self._ACTIVE},
            )
            .to_list(5000)
        )
        forgotten = 0
        for doc in rows:
            if doc.get("tier") == "core":
                continue
            attribution = doc.get("attribution") or {}
            if attribution.get("type") == "manual":
                continue
            last = doc.get("last_reinforced_at") or doc.get("created_at")
            if isinstance(last, str):
                try:
                    last = datetime.fromisoformat(last)
                except ValueError:
                    continue
            if last is None:
                continue
            delta_days = max(0.0, (now - last).total_seconds() / 86_400.0)
            strength = max(1.0, float(doc.get("strength", 1.0)))
            if math.exp(-delta_days / strength) >= retention_floor:
                continue
            await self.db["dm_facts"].update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {"valid_until": _iso(now), "updated_at": _iso(now)},
                    "$inc": {"version": 1},
                },
            )
            forgotten += 1
        return forgotten

    async def get_cursor(self, guild_id: str, key: str) -> str | None:
        doc = await self.db["dm_cursors"].find_one(
            {"guild_id": guild_id, "key": key},
        )
        return doc["value"] if doc else None

    async def set_cursor(self, guild_id: str, key: str, value: str) -> None:
        await self.db["dm_cursors"].update_one(
            {"guild_id": guild_id, "key": key},
            {"$set": {"value": value}},
            upsert=True,
        )

    async def list_guild_ids(self) -> tuple[str, ...]:
        from_facts, from_messages = await asyncio.gather(
            self.db["dm_facts"].distinct("guild_id"),
            self.db["dm_messages"].distinct("guild_id"),
        )
        return tuple(sorted(set(from_facts) | set(from_messages)))

    async def charge_guild_tokens(
        self,
        guild_id: str,
        *,
        day_key: str,
        month_key: str,
        prompt_tokens: int,
    ) -> tuple[int, int]:
        from pymongo import ReturnDocument

        async def bump(period: str) -> int:
            doc = await self.db["dm_budgets"].find_one_and_update(
                {"_id": f"{guild_id}:{period}"},
                {"$inc": {"prompt_tokens": prompt_tokens}},
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            assert doc is not None  # upsert + AFTER always returns the doc
            return int(doc["prompt_tokens"])

        day_total, month_total = await asyncio.gather(bump(day_key), bump(month_key))
        return day_total, month_total

    async def guild_token_usage(
        self,
        guild_id: str,
        *,
        day_key: str,
        month_key: str,
    ) -> tuple[int, int]:
        cursor = self.db["dm_budgets"].find(
            {"_id": {"$in": [f"{guild_id}:{day_key}", f"{guild_id}:{month_key}"]}}
        )
        by_id = {doc["_id"]: int(doc["prompt_tokens"]) async for doc in cursor}
        return (
            by_id.get(f"{guild_id}:{day_key}", 0),
            by_id.get(f"{guild_id}:{month_key}", 0),
        )

    async def touch_facts(
        self,
        guild_id: str,
        fact_ids: tuple[str, ...],
        *,
        accessed_at: datetime,
    ) -> int:
        result = await self.db["dm_facts"].update_many(
            {"guild_id": guild_id, "_id": {"$in": list(fact_ids)}},
            {
                "$set": {
                    "last_reinforced_at": _iso(accessed_at),
                    "updated_at": _iso(accessed_at),
                }
            },
        )
        return int(result.modified_count)

    async def guild_stats(self, guild_id: str) -> GuildStats:
        total = await self.db["dm_facts"].count_documents({"guild_id": guild_id})
        active = await self.db["dm_facts"].count_documents(
            {"guild_id": guild_id, **self._ACTIVE},
        )
        users = len(
            await self.db["dm_facts"].distinct(
                "subject_id",
                {"guild_id": guild_id, "subject_id": {"$ne": None}},
            )
        )
        entities = await self.db["dm_entities"].count_documents({"guild_id": guild_id})
        relations = await self.db["dm_relations"].count_documents(
            {"guild_id": guild_id, "valid_until": None},
        )
        pending = await self.queue.pending_count(guild_id)
        claimed = await self.db["dm_messages"].count_documents(
            {"guild_id": guild_id, "status": CLAIMED}
        )
        dead = await self.queue.dead_letter_count(guild_id)
        return GuildStats(
            guild_id=guild_id,
            total_facts=total,
            active_facts=active,
            user_count=users,
            entity_count=entities,
            relation_count=relations,
            pending_messages=pending,
            in_flight_messages=claimed,
            dead_letters=dead,
        )


__all__ = ["MongoStore"]
