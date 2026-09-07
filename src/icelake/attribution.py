"""Claim-level attribution: map reply spans to the closed source set.

This is the generate-then-attribute pattern production grounded systems use
(Anthropic's span-level citations, the ALCE/post-hoc-attribution literature):
the answer model writes plain prose and never sees citation syntax; a separate
cheap structured call maps verbatim claim spans to source IDs. Generators
hallucinate citations as confidently as facts, so the two skills never share
one call.

Reliability comes from deterministic validation, not from parsing model text:

- The attributor can only reference the closed set (numbered source list).
- Every returned ``claim`` must be a **verbatim substring** of the reply
  (``str.find`` — no regex, no fuzzy matching); anything else is dropped.
- Source numbers outside the list are dropped.

A source that was retrieved but supports no claim is never cited — over-citing
and source-list dumps are structurally impossible.
"""

from __future__ import annotations

from pydantic import Field

from icelake.citations import Citations, CitationSource
from icelake.models.admin import MeterPurpose
from icelake.models.common import FrozenModel
from icelake.ports.llm import ChatLLM, LlmMessage, MessageRole
from icelake.retrieval.injection import snippet
from icelake.structured import complete_structured

#: Hard caps keep the call cheap and bounded regardless of reply length.
MAX_CLAIMS = 12
MAX_SOURCES_PER_CLAIM = 3
ATTRIBUTION_MAX_TOKENS = 1024

_INSTRUCTION = (
    "You attach sources to claims. You are given a REPLY and a numbered SOURCE "
    "LIST. Find every claim in the reply that a source directly supports.\n"
    "Rules:\n"
    "- `claim` must be a VERBATIM substring of the reply, copied exactly "
    "(a sentence or clause; exclude trailing punctuation).\n"
    "- List a source only when it directly supports the claim. Never cite a "
    "source that was merely available but not used.\n"
    "- Claims with no supporting source are omitted, never force-cited.\n"
    "- If nothing in the reply is supported by any source, return an empty list."
)


class AttributedClaim(FrozenModel):
    """One reply span verified against the closed set.

    ``start``/``end`` are character offsets into the reply; ``claim`` is always
    the verbatim substring ``reply[start:end]``. ``sources`` are the supporting
    members of the closed set.
    """

    claim: str
    start: int
    end: int
    sources: tuple[CitationSource, ...]


class ReplyAttribution(FrozenModel):
    """Structured result of attributing one reply: span-level, used-only."""

    claims: tuple[AttributedClaim, ...] = ()

    @property
    def used(self) -> tuple[CitationSource, ...]:
        """Sources that support at least one claim, first-use order, deduped."""
        seen: set[str] = set()
        out: list[CitationSource] = []
        for claim in self.claims:
            for source in claim.sources:
                if source.ref not in seen:
                    seen.add(source.ref)
                    out.append(source)
        return tuple(out)


class _ClaimOut(FrozenModel):
    """Wire contract for one attributed claim (strict json_schema)."""

    claim: str
    sources: list[int] = Field(default_factory=list, max_length=MAX_SOURCES_PER_CLAIM)


class _AttributionOut(FrozenModel):
    """Wire contract for the attribution response (strict json_schema)."""

    attributions: list[_ClaimOut] = Field(default_factory=list, max_length=MAX_CLAIMS)


async def attribute(
    text: str,
    citations: Citations,
    llm: ChatLLM,
    *,
    guild_id: str | None = None,
) -> ReplyAttribution:
    """Map claims in ``text`` to the closed set via one structured LLM call.

    Returns an empty :class:`Attribution` — never raises, never guesses — when
    there is nothing to cite, the call fails validation, or the attributor
    finds no supported claim. Callers weave with :meth:`Citations.apply`.
    """
    citable = list(citations.citable)
    if not text.strip() or not citable:
        return ReplyAttribution()
    listing = "\n".join(
        f"[{number}] {snippet(source.excerpt or source.title or source.url)}"
        for number, source in enumerate(citable, 1)
    )
    output = await complete_structured(
        llm,
        model=_AttributionOut,
        messages=(
            LlmMessage(role=MessageRole.SYSTEM, content=_INSTRUCTION),
            LlmMessage(role=MessageRole.USER, content=f"REPLY:\n{text}\n\nSOURCE LIST:\n{listing}"),
        ),
        max_tokens=ATTRIBUTION_MAX_TOKENS,
        purpose=MeterPurpose.ATTRIBUTION,
        guild_id=guild_id,
    )
    if output is None:
        return ReplyAttribution()
    return ReplyAttribution(claims=_validate(text, citable, output))


def _validate(
    text: str, citable: list[CitationSource], output: _AttributionOut
) -> tuple[AttributedClaim, ...]:
    """Keep only verbatim, in-range, non-overlapping claims — drop the rest."""
    claims: list[AttributedClaim] = []
    cursor = 0
    for item in output.attributions:
        claim = item.claim.strip()
        if not claim:
            continue
        start = text.find(claim, cursor)
        if start < 0:
            continue  # not verbatim (or out of order) — never trust it
        resolved = tuple(
            dict.fromkeys(citable[n - 1] for n in item.sources if 1 <= n <= len(citable))
        )
        if not resolved:
            continue
        end = start + len(claim)
        claims.append(AttributedClaim(claim=claim, start=start, end=end, sources=resolved))
        cursor = end
    return tuple(claims)


__all__ = ["AttributedClaim", "ReplyAttribution", "attribute"]
