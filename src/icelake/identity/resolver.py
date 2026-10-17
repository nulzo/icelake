"""Identity resolution ladder: identifier → hardened user ID.

Order: snowflake passthrough → exact alias → prefix alias. Ambiguity never guesses:
when the runner-up is within ``AMBIGUITY_RATIO`` of the leader the resolution is
flagged ambiguous and callers must not attribute.
"""

from __future__ import annotations

from icelake.identity.aliases import (
    combined_confidence,
    is_valid_alias,
    normalize_alias,
)
from icelake.models.identity import (
    AliasRecord,
    AliasSource,
    Resolution,
    ResolvedCandidate,
)
from icelake.ports.store import MemoryStore

AMBIGUITY_RATIO = 0.85
MIN_PREFIX_LENGTH = 3
_ALIAS_CANDIDATE_LIMIT = 8
_PRESENTATION_LIMIT = 5


def looks_like_snowflake(identifier: str) -> bool:
    """17-20 digit strings are treated as Discord IDs, verbatim."""
    stripped = identifier.strip()
    return stripped.isdigit() and 15 <= len(stripped) <= 25


class IdentityResolver:
    """Read-side resolver over the alias index in the store."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def resolve(self, guild_id: str, identifier: str) -> Resolution:
        """Resolve a name conservatively, leaving ambiguous or truncated matches unresolved."""
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

        exact = await self._store.resolve_alias_candidates(
            guild_id, normalized, limit=_ALIAS_CANDIDATE_LIMIT
        )
        if exact:
            return self._decide(identifier, normalized, exact, fuzzy=False)
        if len(normalized) >= MIN_PREFIX_LENGTH:
            fuzzy = await self._store.prefix_alias_candidates(
                guild_id, normalized, limit=_ALIAS_CANDIDATE_LIMIT
            )
            if fuzzy:
                return self._decide(identifier, normalized, fuzzy, fuzzy=True)
        return Resolution(identifier=identifier)

    def _decide(
        self,
        identifier: str,
        normalized: str,
        records: tuple[AliasRecord, ...],
        *,
        fuzzy: bool,
    ) -> Resolution:
        ordered = sorted(records, key=lambda r: (r.source.rank, r.weight), reverse=True)
        best = ordered[0]
        top_weight = max(r.weight for r in records)
        ambiguous = len(records) >= _ALIAS_CANDIDATE_LIMIT or any(
            record.user_id != best.user_id
            and (fuzzy or record.source.rank == best.source.rank)
            and record.weight >= AMBIGUITY_RATIO * best.weight
            for record in records
        )
        per_user = {record.user_id: record for record in reversed(ordered)}
        candidates = tuple(
            ResolvedCandidate(
                user_id=record.user_id,
                matched_alias=record.alias_norm,
                source=record.source,
                weight=record.weight,
                confidence=combined_confidence(record.source.rank, record.weight, top_weight),
            )
            for record in sorted(
                per_user.values(), key=lambda r: (r.source.rank, r.weight), reverse=True
            )[:_PRESENTATION_LIMIT]
        )
        return Resolution(
            identifier=identifier,
            resolved=None if ambiguous else candidates[0],
            candidates=candidates,
            ambiguous=ambiguous,
        )
