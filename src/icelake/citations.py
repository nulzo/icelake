"""Reply-side citations: a closed source set with deny-by-default parsing.

This mirrors how production assistants (ChatGPT, Claude, Perplexity, Cohere)
keep citations reliable, expressed as a model-agnostic SDK:

1. **Closed set.** Code — never the model — registers every source a reply may
   cite: memory bindings from :class:`PromptContext`, web results, referenced
   messages. The model may only *reference* the set, never extend it.
2. **Model echoes IDs, not URLs.** Sources are labeled ``[1]``, ``[2]`` … in
   the prompt; the contract the model sees is :attr:`Citations.instructions`.
3. **One parsing boundary.** :meth:`Citations.parse` resolves echoed IDs against
   the set, splices provider-anchored sources at their offsets, and removes
   every citation-shaped token outside the set — invented ``[5]``, self-written
   ``[1](url)`` links, ``【…】`` artifacts. URLs only ever leave the library
   from registered sources.

The library returns **structured data** (:class:`ParsedReply` with cleaned
``text`` and resolved ``citations``). How a reply *looks* — inline links,
a footer, nothing — is the consumer's rendering concern, not the library's.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from enum import StrEnum
from typing import TYPE_CHECKING

from icelake.models.common import FrozenModel
from icelake.models.facts import FactRecord
from icelake.retrieval.injection import message_url, snippet

if TYPE_CHECKING:
    from icelake.models.retrieval import Citation, PromptContext

MAX_EXCERPT_CHARS = 200

#: Exact rules a consumer adds to the prompt when link sources are listed.
#: The memory half of the contract ships inside ``PromptContext.injection_block``.
INSTRUCTIONS = (
    "Cite sources by echoing their [src:N] tag inline where the claim is used. "
    "Only echo tags from the source list: never invent or renumber tags, never "
    "paste URLs, and never write markdown links."
)


class MarkerMode(StrEnum):
    """What ``parse`` does with echoed markers in the returned text."""

    #: Replace markers with nothing — clean prose, citations still resolved.
    STRIP = "strip"
    #: Leave ``[src:N]`` / ``[mem:N]`` in place for a consumer renderer.
    KEEP = "keep"


class CitationSource(FrozenModel):
    """One registered, citable source — the unit of the closed set."""

    #: Echo tag the model uses to cite this source (``N``); registry key.
    ref: str
    url: str
    title: str = ""
    excerpt: str = ""
    #: Origin namespace (``memory``, ``web``, ``message``) for consumers that
    #: want to group or label sources by kind.
    kind: str = "web"


class UsedSource(FrozenModel):
    """A source the generator actually used, resolved from the closed set.

    ``claim`` is the matched text span (echoed tag or structured claim) that
    motivated the citation; empty when the consumer only passed an ID.
    """

    ref: str
    url: str
    title: str = ""
    excerpt: str = ""
    kind: str = "web"
    claim: str = ""
    #: Memory provenance when this source came from a fact (else empty).
    fact_id: str = ""
    message_id: str = ""
    channel_id: str = ""


class ParsedReply(FrozenModel):
    """Structured parse of one model reply against the closed set.

    ``text`` is the reply with citation handling applied per the requested
    :class:`MarkerMode`. ``citations`` are the sources the model actually
    used, in first-appearance order, deduped. Presentation is the consumer's.
    """

    text: str
    citations: tuple[UsedSource, ...] = ()


# Echoed refs: ``[mem:3]`` / ``[src:1]``. The namespace prefix is load-bearing:
# it is what lets a citation tag coexist with prose like ``array[0]`` — bare
# ``[N]`` is ambiguous, so the model always echoes a namespaced tag. Lookbehind
# so a second pass cannot eat the inner tag of an already-woven ``[[mem:1]](…)``.
# An optional leading space is captured (group 1) so a stripped tag leaves no
# double space, while a woven link re-emits it.
_TAG_RE = re.compile(r"(?<!\[)( ?)\[((?:mem|src):(\d+))\]")

# Leftover citation-shaped tokens (unknown refs, non-numeric labels). Captures
# the leading space (group 1) and inner token (group 2) so the registry-aware
# stripper can re-emit registered refs intact; a stripped tag leaves no double
# space because the space is consumed with it.
_RESIDUE_RE = re.compile(r"(?<!\[)( ?)\[((?:mem|src):[^\]]*)\]")

# Model-written markdown links with a citation label: ``[mem:1](https://…)``.
_MANGLED_LINK_RE = re.compile(r"(?<!\[)( ?)\[((?:mem|src):)?(\d{1,3})\]\((https?://[^)\s]+)\)")

# ChatGPT-style CJK bracket artifacts copied by other models: ``【4†source】``.
_CJK_RE = re.compile(r" ?【[^】]{0,80}】")


class Citations:
    """Closed citation set for one reply: register sources, parse output.

    Build from :meth:`from_prompt_context` (memory turns) or empty, register
    link sources with :meth:`add_source` / :meth:`add_message`, hand the model
    :attr:`instructions` and :meth:`source_list`, then run the reply through
    :meth:`parse` exactly once. Per-turn, thread-confined, allocation-cheap —
    one set per reply, discarded after.

    Sources are namespaced by origin — memory binds ``mem:N``, link sources
    mint ``src:N`` — so a citation tag is never ambiguous with prose like
    ``array[0]``. Both namespaces live in one ordered registry, so the model
    sees a single contract and consumers get a single ordered list.
    """

    def __init__(self, citations: Iterable[Citation] = ()) -> None:
        self._sources: list[CitationSource] = []
        self._by_ref: dict[str, CitationSource] = {}
        self._by_url: dict[str, CitationSource] = {}
        #: ``(offset, ref)`` provider-anchored splice points — one source may
        #: anchor several spans (providers repeat a URL across a reply).
        self._anchors: list[tuple[int, str]] = []
        #: Memory provenance keyed by ref, for ``UsedSource`` enrichment.
        self._memory: dict[str, Citation] = {}
        for citation in citations:
            self._register_memory(citation)

    @classmethod
    def from_prompt_context(cls, ctx: PromptContext) -> Citations:
        """Seed the closed set from a memory turn's citation bindings."""
        return cls(ctx.citations)

    def add_source(
        self,
        url: str,
        *,
        title: str = "",
        excerpt: str = "",
        kind: str = "web",
        end_index: int | None = None,
    ) -> CitationSource:
        """Register a link source and return it; the model cites it as ``[src:N]``.

        URLs dedupe: re-registering a known URL returns the existing source
        (one number per source, however many spans the provider annotated).
        ``end_index`` is a provider-anchored splice point (character offset in
        the reply, e.g. a web-search ``url_citation`` span) — anchored sources
        are woven deterministically, no model echo needed.
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
        """Registered sources, in ``[N]`` order."""
        return tuple(self._sources)

    @property
    def anchored(self) -> bool:
        """True when any source carries a provider-anchored splice point."""
        return bool(self._anchors)

    @property
    def instructions(self) -> str:
        """Citation rules to show the model; empty when no link sources exist.

        Only the ``[src:N]`` (link-source) half of the contract is returned —
        the memory half already ships inside ``PromptContext.injection_block``,
        so a memory-only turn needs nothing added here.
        """
        has_links = any(source.kind != "memory" for source in self._sources)
        return INSTRUCTIONS if has_links else ""

    def source_list(self) -> str:
        """Prompt-facing source list. Refs are echoable; URLs are withheld."""
        lines: list[str] = []
        for source in self._sources:
            label = source.title or snippet(source.excerpt, MAX_EXCERPT_CHARS) or source.url
            if source.title and source.excerpt:
                label = f"{source.title} — {snippet(source.excerpt, MAX_EXCERPT_CHARS)}"
            lines.append(f"- [{source.ref}] {label}")
        return "\n".join(lines)

    def parse(self, text: str, *, markers: MarkerMode = MarkerMode.STRIP) -> ParsedReply:
        """Validate and parse one model reply against the closed set.

        Order matters: anchored splices land first (offsets index the raw
        text), mangled links are canonicalized before ref resolution so a
        ``[1](url)`` cannot leave a dangling ``(url)``, then echoed refs are
        resolved, and finally all remaining citation-shaped residue is removed.

        ``markers=STRIP`` (default) returns clean prose with markers removed;
        ``markers=KEEP`` leaves ``[N]`` in place for a consumer renderer.
        Either way, ``citations`` holds the resolved sources in first-use order.
        """
        if not text:
            return ParsedReply(text=text)
        used: list[UsedSource] = []
        seen: set[str] = set()

        def note(ref: str, claim: str = "") -> CitationSource | None:
            source = self._by_ref.get(ref)
            if source is None or not source.url:
                return None
            if ref not in seen:
                seen.add(ref)
                used.append(self._to_used(source, claim))
            return source

        text = self._splice_anchored(text, note)
        text = self._canonicalize_links(text, note)
        text = self._resolve_refs(text, note, markers)
        text = self._strip_residue(text)
        text = _CJK_RE.sub("", text)
        return ParsedReply(text=text, citations=tuple(used))

    def apply(self, text: str) -> str:
        """Discord convenience: weave used citations into masked jump links.

        Echoed ``[N]`` tags become ``[[N]](<url>)``; anchored sources splice
        at their offsets; residue is stripped. Kept for the common Discord
        bot path — richer rendering (footers, opt-out) belongs in the consumer
        via :meth:`parse`.
        """
        if not text:
            return text
        seen: set[str] = set()

        def note(ref: str) -> CitationSource | None:
            source = self._by_ref.get(ref)
            if source is None or not source.url:
                return None
            seen.add(ref)
            return source

        text = self._splice_anchored(text, note)
        text = self._canonicalize_links(text, note)

        def weave(match: re.Match[str]) -> str:
            space, ref = match.group(1), match.group(2)
            source = self._by_ref.get(ref)
            if source is None or not source.url:
                return ""
            return f"{space}[[{ref}]](<{source.url}>)"

        text = _TAG_RE.sub(weave, text)
        text = self._strip_residue(text)
        text = _CJK_RE.sub("", text)
        return text

    def _strip_residue(self, text: str) -> str:
        """Remove citation-shaped tokens that are not in the closed set.

        Registry-aware: a tag matching a registered ref (already woven or kept)
        is left alone; anything else citation-shaped is deleted. This is the
        deny-by-default backstop.
        """

        def sub(match: re.Match[str]) -> str:
            space, token = match.group(1), match.group(2)
            # Registered refs survive intact (re-emit space + bracketed tag);
            # anything else citation-shaped is deleted along with its space.
            return f"{space}[{token}]" if token in self._by_ref else ""

        return _RESIDUE_RE.sub(sub, text)

    # -- internals ---------------------------------------------------------

    def _register_memory(self, citation: Citation) -> None:
        """Seed a memory binding, keeping its ``mem:N`` ref."""
        ref = citation.ref if citation.ref.startswith("mem:") else f"mem:{citation.ref}"
        source = CitationSource(
            ref=ref,
            url=citation.url,
            title=citation.subject_name,
            excerpt=citation.snippet,
            kind="memory",
        )
        self._sources.append(source)
        self._by_ref[ref] = source
        if citation.url:
            self._by_url[citation.url] = source
        self._memory[ref] = citation

    def _to_used(self, source: CitationSource, claim: str) -> UsedSource:
        memory = self._memory.get(source.ref)
        return UsedSource(
            ref=source.ref,
            url=source.url,
            title=source.title,
            excerpt=source.excerpt,
            kind=source.kind,
            claim=claim,
            fact_id=memory.fact_id if memory else "",
            message_id=(memory.message_id or "") if memory else "",
            channel_id=(memory.channel_id or "") if memory else "",
        )

    def _splice_anchored(self, text: str, note: Callable[[str], CitationSource | None]) -> str:
        if not self._anchors:
            return text
        out = text
        used_offsets: set[int] = set()
        # Descending so earlier offsets stay valid as later insertions shift text.
        for end_index, ref in sorted(self._anchors, reverse=True):
            offset = min(end_index, len(out))
            if offset in used_offsets:
                continue  # providers repeat spans; one link per anchor
            used_offsets.add(offset)
            source = note(ref)
            if source is not None:
                out = f"{out[:offset]} [{ref}]{out[offset:]}"
        return out

    def _canonicalize_links(self, text: str, note: Callable[[str], CitationSource | None]) -> str:
        """Rewrite model-written ``[src:N](url)`` to the registered ref.

        The ref claim is checked against the set; the written URL is untrusted
        and always replaced with the registered one. Constructs matching
        nothing in the set are deleted — URLs never originate from the model.
        """

        def sub(match: re.Match[str]) -> str:
            space, prefix, number, url = match.groups()
            source = self._by_ref.get(f"{prefix}{number}" if prefix else number)
            if source is None:
                source = self._by_url.get(url)
            if source is None or not source.url:
                return ""
            note(source.ref)
            return f"{space}[{source.ref}]"

        return _MANGLED_LINK_RE.sub(sub, text)

    def _resolve_refs(
        self, text: str, note: Callable[[str], CitationSource | None], markers: MarkerMode
    ) -> str:
        keep = markers is MarkerMode.KEEP

        def sub(match: re.Match[str]) -> str:
            space, ref = match.group(1), match.group(2)
            source = note(ref)
            if source is None:
                return ""
            return f"{space}[{ref}]" if keep else ""

        return _TAG_RE.sub(sub, text)


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
        urls = list(
            dict.fromkeys(
                source.message_url
                or message_url(source.guild_id, source.channel_id, source.message_id)
                for source in record.citations
                if source.message_id or source.message_url
            )
        )
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
    "INSTRUCTIONS",
    "CitationSource",
    "Citations",
    "MarkerMode",
    "ParsedReply",
    "UsedSource",
    "message_url",
    "render_fact_list",
]
