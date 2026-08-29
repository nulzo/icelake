"""SQLite-backed LlmCache. Bounded: oldest rows are pruned past the cap."""

from __future__ import annotations

from datetime import UTC, datetime

from icelake.adapters.sqlite.connection import SqliteConnection
from icelake.ports.llm import ChatResponse

_MAX_ROWS = 1000


class SqliteLlmCache:
    def __init__(self, db: SqliteConnection) -> None:
        self._db = db

    async def get(self, key: str) -> ChatResponse | None:
        row = await self._db.query_one(
            "SELECT response, model FROM dm_llm_cache WHERE key = ?",
            (key,),
        )
        if row is None:
            return None
        # Zero tokens: a replay costs nothing, and MeteredLLM records what it sees.
        return ChatResponse(text=row["response"], model=row["model"])

    async def put(self, key: str, response: ChatResponse) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO dm_llm_cache (key, response, model, created_at)"
            " VALUES (?, ?, ?, ?)",
            (key, response.text, response.model, datetime.now(UTC).isoformat()),
        )
        await self._db.execute(
            "DELETE FROM dm_llm_cache WHERE key NOT IN"
            " (SELECT key FROM dm_llm_cache ORDER BY created_at DESC LIMIT ?)",
            (_MAX_ROWS,),
        )


__all__ = ["SqliteLlmCache"]
