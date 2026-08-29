"""In-memory adapter package exports."""

from __future__ import annotations

from icelake.adapters.in_memory.queue import InMemoryIngestQueue
from icelake.adapters.in_memory.store import InMemoryStore
from icelake.adapters.in_memory.vectors import InMemoryVectorIndex
from icelake.ports.vectors import cosine

__all__ = ["InMemoryIngestQueue", "InMemoryStore", "InMemoryVectorIndex", "cosine"]
