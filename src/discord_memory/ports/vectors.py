"""Vector index port. Backends: in-memory brute force, SQLite, Postgres (pgvector)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from discord_memory.models.common import FrozenModel


class VectorItem(FrozenModel):
    """One indexed fact embedding with scope metadata for pre-filtering."""

    id: str
    guild_id: str
    subject_id: str | None
    embedding: tuple[float, ...]


class VectorHit(FrozenModel):
    """Candidate hit; scores are cosine similarity in [0, 1]."""

    id: str
    score: float


@runtime_checkable
class VectorIndex(Protocol):
    """ANN or brute-force nearest-neighbor search scoped to a guild.

    Implementations pre-filter by guild/scope server-side where possible; validity and
    consent filtering happen after joining back to facts.
    """

    async def setup(self) -> None: ...

    async def upsert(self, items: tuple[VectorItem, ...]) -> None: ...

    async def delete(self, ids: tuple[str, ...]) -> int: ...

    async def search(
        self,
        embedding: tuple[float, ...],
        *,
        guild_id: str,
        subject_ids: tuple[str, ...] | None = None,
        server_only: bool = False,
        limit: int = 20,
        candidate_cap: int = 500,
    ) -> tuple[VectorHit, ...]: ...

    async def count(self, guild_id: str) -> int: ...
