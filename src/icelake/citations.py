"""Reply-side citations: a closed source set with structured attribution.

This mirrors how production grounded systems (ChatGPT, Claude, Perplexity,
Graphiti/Zep) keep citations reliable, expressed as a model-agnostic SDK:

1. **Closed set.** Code — never the model — registers every source a reply may
   cite: memory bindings from :class:`PromptContext`, tool results, web
   annotations. The model may only *reference* the set, never extend it.
2. **Citations are data, not text.** The answer model never sees tags or URLs
   and never writes citation syntax. Attribution is a separate structured
   stage — :func:`icelake.attribution.attribute` maps verbatim reply spans to
   set members; provider web annotations arrive as character offsets.
3. **Deterministic rendering.** :meth:`Citations.apply` weaves links by
   splicing at validated offsets. There is no parsing of model output, so
   invented refs, mangled links, and source-list dumps have no way to exist.

How a reply *looks* beyond the inline weave — a footer, nothing — is the
consumer's rendering concern, fed by :attr:`ReplyAttribution.used`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from icelake.models.common import FrozenModel
from icelake.models.facts import FactRecord
from icelake.retrieval.injection import message_url, snippet

if TYPE_CHECKING:
    from icelake.attribution import ReplyAttribution
    from icelake.models.retrieval import Citation, PromptContext, ScoredFact

MAX_EXCERPT_CHARS = 200


class CitationSource(FrozenModel):
    """One registered, citable source — the unit of the closed set."""

    #: Registry key (``mem:N`` seeded from memory bindings, ``src:N`` minted).
    ref: str
    url: str
    title: str = ""
    excerpt: str = ""
    #: Origin namespace (``memory``, ``web``, ``message``) for consumers that
    #: want to group or label sources by kind.
    kind: str = "web"


class Citations:
    """Closed citation set for one reply: register sources, weave output.

    Build from :meth:`from_prompt_context` (memory turns) or empty, register
    link sources with :meth:`add_source` / :meth:`add_message`, attribute the
    reply with :func:`icelake.attribution.attribute`, then :meth:`apply` once.
    Per-turn, thread-confined, allocation-cheap — one set per reply, discarded
    after.
    """

    def __init__(
        self,
        citations: Iterable[Citation] = (),
        facts: Iterable[ScoredFact] = (),
    ) -> None:
        self._sources: list[CitationSource] = []
        self._by_ref: dict[str, CitationSource] = {}
        self._by_url: dict[str, CitationSource] = {}
        #: ``(offset, ref)`` provider-anchored splice points — one source may
        #: anchor several spans (providers repeat a URL across a reply).
        self._anchors: list[tuple[int, str]] = []
        fact_text = {scored.fact.id: scored.fact.text for scored in facts}
        for citation in citations:
            self._register_memory(citation, fact_text.get(citation.fact_id, ""))

    @classmethod
    def from_prompt_context(cls, ctx: PromptContext) -> Citations:
        """Seed the closed set from a memory turn's citation bindings.

        Fact text joins each binding so the attribution stage sees the same
        statement the answer model saw, not a raw message snippet.
        """
        return cls(ctx.citations, facts=ctx.facts)

    def add_source(
        self,
        url: str,
        *,
        title: str = "",
        excerpt: str = "",
        kind: str = "web",
        end_index: int | None = None,
    ) -> CitationSource:
        """Register a link source and return it.

        URLs dedupe: re-registering a known URL returns the existing source
        (one number per source, however many spans the provider annotated).
        ``end_index`` is a provider-anchored splice point (character offset in
        the reply, e.g. a web-search ``url_citation`` span) — anchored sources
        are woven deterministically, no attribution needed.
        """
        source = self._by_url.get(url)
        if source is None:
            source = CitationSource(
                ref=f"src:{len(self._sources) + 1}",
                url=url,
                title=title,
                excerpt=excerpt,
                kind=kind,
            )
            self._sources.append(source)
            self._by_ref[source.ref] = source
            self._by_url[url] = source
        if end_index is not None:
            self._anchors.append((end_index, source.ref))
        return source

    def add_message(
        self,
        guild_id: str,
        channel_id: str,
        message_id: str,
        *,
        title: str = "",
        excerpt: str = "",
    ) -> CitationSource:
        """Register a Discord message as a source (jump link built by the library)."""
        return self.add_source(
            message_url(guild_id, channel_id, message_id),
            title=title,
            excerpt=excerpt,
            kind="message",
        )

    @property
    def sources(self) -> tuple[CitationSource, ...]:
        """Registered sources, in registry order."""
        return tuple(self._sources)

    @property
    def citable(self) -> tuple[CitationSource, ...]:
        """Sources that can actually be cited (have a URL), in registry order.

        Display numbers — inline labels, footers, the attributor's source
        list — are 1-based positions in this list, so they always agree.
        """
        return tuple(source for source in self._sources if source.url)

    @property
    def anchored(self) -> bool:
        """True when any source carries a provider-anchored splice point."""
        return bool(self._anchors)

    def apply(self, text: str, attribution: ReplyAttribution | None = None) -> str:
        """Discord convenience: weave citations into masked jump links.

        Both inputs are structured and validated before they arrive: provider
        anchors (character offsets from web annotations) and attributed claims
        (verbatim spans from :func:`icelake.attribution.attribute`). Each
        becomes ``[[N]](<url>)`` where ``N`` is the source's :attr:`citable`
        position. Insertions splice back-to-front so offsets stay valid; no
        model text is parsed and URLs only ever come from the registry.
        """
        if not text:
            return text
        positions = {source.ref: number for number, source in enumerate(self.citable, 1)}
        insertions: list[tuple[int, str]] = []
        used_offsets: set[int] = set()
        for end_index, ref in self._anchors:
            source = self._by_ref.get(ref)
            offset = min(end_index, len(text))
            if source is None or not source.url or offset in used_offsets:
                continue  # providers repeat spans; one link per anchor point
            used_offsets.add(offset)
            insertions.append((offset, f" [[{positions[source.ref]}]](<{source.url}>)"))
        for claim in attribution.claims if attribution is not None else ():
            links = "".join(
                f" [[{positions[source.ref]}]](<{source.url}>)"
                for source in claim.sources
                if source.url
            )
            if links:
                insertions.append((claim.end, links))
        for offset, chunk in sorted(insertions, key=lambda item: item[0], reverse=True):
            text = f"{text[:offset]}{chunk}{text[offset:]}"
        return text

    # -- internals ---------------------------------------------------------

    def _register_memory(self, citation: Citation, fact_text: str = "") -> None:
        """Seed a memory binding, keeping its ``mem:N`` ref."""
        ref = citation.ref if citation.ref.startswith("mem:") else f"mem:{citation.ref}"
        source = CitationSource(
            ref=ref,
            url=citation.url,
            title=citation.subject_name,
            excerpt=snippet(fact_text) if fact_text else citation.snippet,
            kind="memory",
        )
        self._sources.append(source)
        self._by_ref[ref] = source
        if citation.url:
            self._by_url[citation.url] = source


def fact_source_urls(record: FactRecord) -> tuple[str, ...]:
    """Distinct source URLs for a fact — stored snapshot or rebuilt from snowflakes.

    This is the single place the URL-fallback rule lives; consumers building
    citation payloads use it instead of re-deriving URLs from ``SourceRef``.
    """
    return tuple(
        dict.fromkeys(
            source.message_url or message_url(source.guild_id, source.channel_id, source.message_id)
            for source in record.citations
            if source.message_id or source.message_url
        )
    )


def render_fact_list(records: Iterable[FactRecord]) -> str:
    """Render fact records as a Discord-safe list with numbered jump links.

    Every distinct source URL on the fact gets a link; one numbered pool is
    shared across the whole list so a source cited by two facts keeps its
    number. Facts with no resolvable source render bare. URLs come from the
    stored snapshot or are rebuilt from snowflakes — never from model output.
    """
    lines: list[str] = []
    order: dict[str, int] = {}
    for record in records:
        urls = fact_source_urls(record)
        if not urls:
            lines.append(f"- {record.text}")
            continue
        links: list[str] = []
        for url in urls:
            if url not in order:
                order[url] = len(order) + 1
            links.append(f"[[{order[url]}]](<{url}>)")
        lines.append(f"- {record.text} {' '.join(links)}")
    return "\n".join(lines)


__all__ = [
    "CitationSource",
    "Citations",
    "fact_source_urls",
    "message_url",
    "render_fact_list",
]
