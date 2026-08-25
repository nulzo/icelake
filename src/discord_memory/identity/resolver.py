"""Identity resolution ladder: identifier → hardened user ID (PLAN.md §3.2).

Order: snowflake passthrough → exact alias → prefix alias. Ambiguity never guesses:
when the runner-up is within ``AMBIGUITY_RATIO`` of the leader the resolution is
flagged ambiguous and callers must not attribute.
"""

from __future__ import annotations

from discord_memory.identity.aliases import (
    combined_confidence,
    is_valid_alias,
    normalize_alias,
)
from discord_memory.models.identity import (
    AliasRecord,
    AliasSource,
    Resolution,
    ResolvedCandidate,
)
from discord_memory.ports.store import MemoryStore

AMBIGUITY_RATIO = 0.85
MIN_PREFIX_LENGTH = 3


def looks_like_snowflake(identifier: str) -> bool:
    """17-20 digit strings are treated as Discord IDs, verbatim."""
    stripped = identifier.strip()
    return stripped.isdigit() and 15 <= len(stripped) <= 25


class IdentityResolver:
    """Read-side resolver over the alias index in the store."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def resolve(self, guild_id: str, identifier: str) -> Resolution:
        normalized = normalize_alias(identifier)
        if not normalized:
            return Resolution(identifier=identifier)

        if looks_like_snowflake(identifier.strip()):
            candidate = ResolvedCandidate(
                user_id=identifier.strip(),
                matched_alias=identifier.strip(),
                source=AliasSource.MENTION,
                weight=1.0,
                confidence=1.0,
            )
            return Resolution(
                identifier=identifier,
                resolved=candidate,
                candidates=(candidate,),
            )

        if not is_valid_alias(normalized):
            return Resolution(identifier=identifier)

        exact = await self._store.resolve_alias_candidates(guild_id, normalized)
        if exact:
            return self._decide(identifier, normalized, exact)
        if len(normalized) >= MIN_PREFIX_LENGTH:
            fuzzy = await self._store.prefix_alias_candidates(guild_id, normalized)
            if fuzzy:
                return self._decide(identifier, normalized, fuzzy)
        return Resolution(identifier=identifier)

    def _decide(
        self,
        identifier: str,
        normalized: str,
        records: tuple[AliasRecord, ...],
    ) -> Resolution:
        best = max(records, key=lambda r: (r.source.rank, r.weight))
        top_weight = max(r.weight for r in records)
        candidates = tuple(
            ResolvedCandidate(
                user_id=record.user_id,
                matched_alias=record.alias_norm,
                source=record.source,
                weight=record.weight,
                confidence=combined_confidence(record.source.rank, record.weight, top_weight),
            )
            for record in records[:5]
        )
        rivals = [c for c in candidates if c.user_id != best.user_id]
        ambiguous = any(c.weight >= AMBIGUITY_RATIO * best.weight for c in rivals)
        resolved = None if ambiguous else candidates[0]
        return Resolution(
            identifier=identifier,
            resolved=resolved,
            candidates=candidates,
            ambiguous=ambiguous,
        )
