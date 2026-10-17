"""Replaceable claim-support verification, independent of citation rendering."""

from typing import Protocol

from pydantic import Field

from icelake.models.common import FrozenModel


class CitationCandidate(FrozenModel):
    """One bounded claim/evidence pair. IDs are minted by the caller."""

    id: int = Field(ge=0)
    claim: str = Field(min_length=1, max_length=4000)
    evidence: str = Field(min_length=1, max_length=4000)


class CitationVerification(FrozenModel):
    """Only positively supported pairs; omission means abstention."""

    supported_ids: tuple[int, ...] = Field(default=(), max_length=36)


class CitationVerifier(Protocol):
    """Verify complete claims, including entity, negation, time and attribution.

    Implementations must abstain on ambiguity or partial support. Similarity
    alone is insufficient. The caller rejects IDs outside the submitted set.
    """

    async def verify(self, candidates: tuple[CitationCandidate, ...]) -> CitationVerification: ...
