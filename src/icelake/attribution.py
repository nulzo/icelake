"""Claim-level attribution: deterministic matching of reply spans to sources.

This is the ALCE ``POSTCITE`` pattern (Gao et al., EMNLP 2023) — the same
machinery retrieval already uses, pointed at the answer instead of the
question. Sentences of the reply and the closed set's source texts are
embedded in one batched call; a source is cited at a span when cosine
similarity clears the threshold. Dense embeddings absorb paraphrase and
persona voice the same way they do at retrieval time, so a mutated reply
("bro is a mayo goblin") still matches the canonical fact text ("klim likes
mayonnaise").

Why not an LLM: production memory systems (Graphiti, Mem0, Letta) treat
provenance as data written at ingest and never run a model to attach
citations; the answer model never sees citation syntax at all. The failure
mode here is asymmetric by design — a sentence that matches nothing is simply
uncited; a wrong citation is never produced. No chat completion, no parsing
of model output, deterministic and unit-testable with a fake embedder.
"""

from __future__ import annotations

import logging

from icelake.citations import Citations, CitationSource
from icelake.models.common import FrozenModel
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


async def attribute(
    text: str,
    citations: Citations,
    embedder: Embedder,
    *,
    threshold: float = DEFAULT_MIN_SCORE,
) -> ReplyAttribution:
    """Match reply segments to the closed set by embedding similarity.

    One batched embed call per reply, only when there is something to cite.
    Never raises and never guesses: embedder failure or an empty match set
    returns an empty attribution, and the caller sends the reply uncited.
    """
    segments = _segments(text)[:MAX_CLAIMS]
    sources = [
        (source.excerpt or source.title, source)
        for source in citations.citable
        if (source.excerpt or source.title).strip()
    ]
    if not segments or not sources:
        return ReplyAttribution()
    try:
        vectors = await embedder.embed(
            [claim for _, _, claim in segments] + [source_text for source_text, _ in sources]
        )
    except Exception:
        logger.warning("attribution embed failed; reply stays uncited", exc_info=True)
        return ReplyAttribution()
    if len(vectors) != len(segments) + len(sources):
        logger.warning("attribution embed count mismatch; reply stays uncited")
        return ReplyAttribution()
    segment_vectors = vectors[: len(segments)]
    source_vectors = vectors[len(segments) :]
    claims: list[AttributedClaim] = []
    for (start, end, claim), segment_vector in zip(segments, segment_vectors, strict=True):
        ranked = sorted(
            (
                (cosine(segment_vector, source_vector), source)
                for source_vector, (_, source) in zip(source_vectors, sources, strict=True)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        matched = tuple(
            dict.fromkeys(source for score, source in ranked if score >= threshold)
        )[:MAX_SOURCES_PER_CLAIM]
        if matched:
            claims.append(AttributedClaim(claim=claim, start=start, end=end, sources=matched))
    return ReplyAttribution(claims=tuple(claims))


def _segments(text: str) -> list[tuple[int, int, str]]:
    """Sentence-ish spans with offsets into ``text``.

    Boundaries are newlines and terminal punctuation followed by whitespace —
    text segmentation only; nothing here parses citation syntax.
    """
    out: list[tuple[int, int, str]] = []
    start = 0
    for index, char in enumerate(text):
        at_end = index + 1 == len(text)
        if char == "\n" or (
            char in _SENTENCE_END and (at_end or text[index + 1] in " \n\t")
        ):
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
