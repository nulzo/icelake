"""Vector index port. Backends: in-memory brute force, SQLite, Postgres (pgvector)."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from icelake.models.common import FrozenModel


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity clipped to [0, 1]."""
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


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
