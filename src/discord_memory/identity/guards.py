"""Bot guard and consent enforcement (ported from memory_bot_guard, hardened).

Bot accounts are never memory subjects. Opted-out users are never observed or
recalled. ``user_id=None`` (server scope) is always allowed.
"""

from __future__ import annotations

from discord_memory.ports.store import MemoryStore


class BotGuard:
    """Tracks bot accounts that must never receive stored memories."""

    def __init__(self) -> None:
        self._bot_ids: set[str] = set()

    def register(self, user_id: str) -> None:
        self._bot_ids.add(str(user_id))

    def register_many(self, user_ids: tuple[str, ...]) -> None:
        self._bot_ids.update(str(uid) for uid in user_ids)

    def note_author(self, author_id: str | None, *, is_bot: bool) -> None:
        if is_bot and author_id:
            self._bot_ids.add(str(author_id))

    def is_bot(self, user_id: str | None) -> bool:
        return user_id is not None and str(user_id) in self._bot_ids


class ConsentPolicy:
    """Opt-out checks against the store."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def is_blocked(self, guild_id: str, user_id: str) -> bool:
        return await self._store.get_opt_out(guild_id, user_id)


class SubjectGate:
    """Combined admission check for making a subject out of a user id."""

    def __init__(self, guard: BotGuard, consent: ConsentPolicy) -> None:
        self._guard = guard
        self._consent = consent

    async def allows(self, guild_id: str, user_id: str | None) -> bool:
        if user_id is None:
            return True
        if self._guard.is_bot(user_id):
            return False
        return not await self._consent.is_blocked(guild_id, user_id)
