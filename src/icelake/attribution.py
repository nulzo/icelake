"""Closed-set attribution with conservative defaults and explicit verification.

Exact restatements can link to their registered evidence. Embeddings only select
candidate evidence for an optional verifier; similarity is never proof of support.
A citation indicates provenance, not independent validation of the source's truth.
"""

from __future__ import annotations

import logging

from icelake.citations import Citations, CitationSource
from icelake.models.common import FrozenModel
from icelake.ports.citations import CitationCandidate, CitationVerifier
from icelake.ports.llm import Embedder
from icelake.ports.vectors import cosine

logger = logging.getLogger(__name__)

#: Bounds keep one pathological reply from exploding the embed batch.
MAX_CLAIMS = 12
MAX_SOURCES_PER_CLAIM = 3
#: Segments with fewer word characters than this are voice noise ("lol", "💀").
MIN_SEGMENT_WORD_CHARS = 3
DEFAULT_MIN_SCORE = 0.55

_SENTENCE_END = frozenset(".!?…")
#: Trailing characters excluded from the claim span so links land before the
#: sentence's punctuation: ``he likes mayo [[1]].``, not ``…mayo. [[1]]``.
_TRAILING = frozenset(" .!?,;:\"')]}…\t\n" + "\u201d\u2019")


class AttributedClaim(FrozenModel):
    """One reply span matched to sources from the closed set.

    ``start``/``end`` are character offsets into the reply; ``claim`` is always
    the verbatim substring ``reply[start:end]``. ``sources`` are the supporting
    members of the closed set, best match first.
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


class AttributedReply(FrozenModel):
    """Consumer-facing result of ``DiscordMemory.cite``.

    ``text`` is the reply with the canonical Discord weave applied (or the raw
    reply when ``weave=False``). ``claims``/``sources`` are the structured
    spans and used-only sources for consumers that render their own way —
    footers, dashboards, non-Discord surfaces.
    """

    text: str
    claims: tuple[AttributedClaim, ...] = ()
    sources: tuple[CitationSource, ...] = ()


def _normalized(text: str) -> str:
    return " ".join(text.strip().removesuffix(".").split())


async def attribute(
    text: str,
    citations: Citations,
    embedder: Embedder | None = None,
    *,
    threshold: float = DEFAULT_MIN_SCORE,
    verifier: CitationVerifier | None = None,
) -> ReplyAttribution:
    """Attribute exact restatements, or verify bounded semantic candidates.

    Without a verifier, no provider is called. With one, embeddings shortlist
    at most three sources per claim; only explicitly verified pairs are cited.
    Provider failures abstain. Cancellation propagates to the caller.
    """
    segments = _segments(text)[:MAX_CLAIMS]
    sources = [source for source in citations.citable if source.excerpt.strip()][:100]
    matches = {
        index: [source for source in sources if _normalized(claim) == _normalized(source.excerpt)][
            :MAX_SOURCES_PER_CLAIM
        ]
        for index, (_, _, claim) in enumerate(segments)
    }
    if verifier is not None and embedder is not None and segments and sources:
        try:
            verified = await _verify_candidates(segments, sources, embedder, verifier, threshold)
            for index, supported in verified.items():
                matches[index] = list(dict.fromkeys([*matches[index], *supported]))[
                    :MAX_SOURCES_PER_CLAIM
                ]
        except Exception:
            logger.warning(
                "citation verification failed; uncertain claims stay uncited", exc_info=True
            )
    return ReplyAttribution(
        claims=tuple(
            AttributedClaim(claim=claim, start=start, end=end, sources=tuple(matches[index]))
            for index, (start, end, claim) in enumerate(segments)
            if matches[index]
        )
    )


async def _verify_candidates(
    segments: list[tuple[int, int, str]],
    sources: list[CitationSource],
    embedder: Embedder,
    verifier: CitationVerifier,
    threshold: float,
) -> dict[int, list[CitationSource]]:
    vectors = await embedder.embed(
        [claim for _, _, claim in segments] + [source.excerpt[:4000] for source in sources]
    )
    if len(vectors) != len(segments) + len(sources):
        return {}
    candidates: list[CitationCandidate] = []
    bindings: dict[int, tuple[int, CitationSource]] = {}
    for index, (_, _, claim) in enumerate(segments):
        if len(claim) > 4000:
            continue
        ranked = sorted(
            zip(sources, vectors[len(segments) :], strict=True),
            key=lambda pair: cosine(vectors[index], pair[1]),
            reverse=True,
        )
        for source, vector in ranked[:MAX_SOURCES_PER_CLAIM]:
            if cosine(vectors[index], vector) < threshold:
                continue
            candidate_id = len(candidates)
            candidates.append(
                CitationCandidate(id=candidate_id, claim=claim, evidence=source.evidence[:4000])
            )
            bindings[candidate_id] = (index, source)
    if not candidates:
        return {}
    result = await verifier.verify(tuple(candidates))
    supported: dict[int, list[CitationSource]] = {}
    for candidate_id in dict.fromkeys(result.supported_ids):
        if candidate_id in bindings:
            index, source = bindings[candidate_id]
            supported.setdefault(index, []).append(source)
    return supported


def _segments(text: str) -> list[tuple[int, int, str]]:
    """Sentence-ish spans with offsets into ``text``.

    Boundaries are newlines and terminal punctuation followed by whitespace —
    text segmentation only; nothing here parses citation syntax.
    """
    out: list[tuple[int, int, str]] = []
    start = 0
    for index, char in enumerate(text):
        at_end = index + 1 == len(text)
        if char == "\n" or (char in _SENTENCE_END and (at_end or text[index + 1] in " \n\t")):
            _emit(text, start, index + (0 if char == "\n" else 1), out)
            start = index + 1
    _emit(text, start, len(text), out)
    return out


def _emit(text: str, start: int, end: int, out: list[tuple[int, int, str]]) -> None:
    while start < end and text[start] in " \t\n":
        start += 1
    while end > start and text[end - 1] in _TRAILING:
        end -= 1
    claim = text[start:end]
    if sum(char.isalnum() for char in claim) >= MIN_SEGMENT_WORD_CHARS:
        out.append((start, end, claim))


__all__ = ["AttributedClaim", "AttributedReply", "ReplyAttribution", "attribute"]
