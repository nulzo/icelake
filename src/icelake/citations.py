"""Reply-side citation system: a closed source set with deny-by-default rendering.

This is the discipline production assistants (ChatGPT, Claude, Perplexity) use:

1. **Closed set.** Code — never the model — registers every source a reply may
   cite: memory citations from :class:`PromptContext`, web results, referenced
   messages. The model may only *reference* the set, never extend it.
2. **Model echoes refs, not URLs.** Sources are labeled ``[mem:N]`` (memory
   facts, minted by the injection block) or ``[src:N]`` (link sources, minted
   here). The contract the model sees is :attr:`CitationRegistry.prompt_contract`.
3. **One rendering boundary.** :meth:`CitationRegistry.apply` splices
   provider-anchored sources at their offsets, weaves echoed refs into
   Discord-safe links, canonicalizes mangled-but-known references, and deletes
   every citation-shaped token outside the set — invented ``[5]``, unknown
   ``[mem:99]``, self-written ``[1](url)`` links, ``【…】`` artifacts.
   Citation URLs only ever leave the library from registered sources.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

from icelake.models.common import FrozenModel
from icelake.models.facts import FactRecord
from icelake.retrieval.injection import message_url, snippet

if TYPE_CHECKING:
    from icelake.models.retrieval import Citation, PromptContext

MAX_EXCERPT_CHARS = 200

#: Exact rules a consumer adds to the prompt when link sources are listed.
#: The memory half of the contract ships inside ``PromptContext.injection_block``.
PROMPT_CONTRACT = (
    "Cite sources by echoing their [src:N] tag inline where the claim is used. "
    "Only echo tags from the source list: never invent or renumber tags, never "
    "paste URLs, and never write markdown links."
)

# Echoed refs: ``[mem:3]`` / ``[src:1]``. Lookbehind so a second pass cannot
# eat the inner tag of an already-woven ``[[mem:3]](<url>)``.
_TAG_RE = re.compile(r"(?<!\[)\[((?:mem|src):)(\d+)\]")

# Leftover citation-shaped tokens (unknown refs, non-numeric labels).
_RESIDUE_RE = re.compile(r"(?<!\[) ?\[(?:mem|src):[^\]]*\]")

# Model-written markdown links with a numeric label: ``[1](https://…)``.
_MANGLED_LINK_RE = re.compile(r"(?<!\[) ?\[((?:mem|src):)?(\d{1,3})\]\((https?://[^)\s]+)\)")

# Bare ``[N]`` the model invented (or echoed for a link source). The delimiter
# (whitespace/punctuation/start) is captured and re-emitted, so word-attached
# (``array[0]``) and index-chained (``matrix[i][2]``) prose is never touched.
_BARE_NUM_RE = re.compile(r"(^|[\s,;:!?)])\[(\d{1,3})\](?![\w(])", re.MULTILINE)

# ChatGPT-style CJK bracket artifacts copied by other models: ``【4†source】``.
_CJK_RE = re.compile(r" ?【[^】]{0,80}】")


class CitationSource(FrozenModel):
    """One registered, citable source — the unit of the closed set."""

    #: Echo tag the model uses to cite this source (``src:N``); registry key.
    ref: str
    url: str
    title: str = ""
    excerpt: str = ""


class CitationRegistry:
    """Closed citation set for one reply: register sources, render output.

    Build from :meth:`from_prompt_context` (memory turns) or empty, register
    link sources with :meth:`add_source` / :meth:`add_message`, hand the model
    :attr:`prompt_contract` and :meth:`source_list`, then run the reply through
    :meth:`apply` exactly once before posting. Per-turn, thread-confined,
    allocation-cheap — one registry per reply, discarded after.
    """

    def __init__(self, citations: Iterable[Citation] = ()) -> None:
        self._memory: dict[str, Citation] = {c.ref: c for c in citations}
        self._sources: list[CitationSource] = []
        self._by_url: dict[str, CitationSource] = {}
        #: ``(offset, ref)`` provider-anchored splice points — one source may
        #: anchor several spans (providers repeat a URL across a reply).
        self._anchors: list[tuple[int, str]] = []

    @classmethod
    def from_prompt_context(cls, ctx: PromptContext) -> CitationRegistry:
        """Seed the closed set from a memory turn's citation bindings."""
        return cls(ctx.citations)

    def add_source(
        self,
        url: str,
        *,
        title: str = "",
        excerpt: str = "",
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
                ref=f"src:{len(self._sources) + 1}", url=url, title=title, excerpt=excerpt
            )
            self._sources.append(source)
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
            message_url(guild_id, channel_id, message_id), title=title, excerpt=excerpt
        )

    @property
    def sources(self) -> tuple[CitationSource, ...]:
        """Registered link sources, in ``src:N`` order."""
        return tuple(self._sources)

    @property
    def anchored(self) -> bool:
        """True when any source carries a provider-anchored splice point."""
        return bool(self._anchors)

    @property
    def prompt_contract(self) -> str:
        """Citation rules to show the model; empty when no link sources exist.

        The memory half of the contract already ships inside
        ``PromptContext.injection_block`` — do not duplicate it.
        """
        return PROMPT_CONTRACT if self._sources else ""

    def source_list(self) -> str:
        """Prompt-facing source list. Refs are echoable; URLs are withheld."""
        lines: list[str] = []
        for source in self._sources:
            label = source.title or snippet(source.excerpt, MAX_EXCERPT_CHARS) or source.url
            if source.title and source.excerpt:
                label = f"{source.title} — {snippet(source.excerpt, MAX_EXCERPT_CHARS)}"
            lines.append(f"- [{source.ref}] {label}")
        return "\n".join(lines)

    def woven_list(self) -> str:
        """Per-line source list with links already woven: ``- label [[N]](<url>)``.

        Self-contained (no echo refs), so it is safe inside tool results that a
        downstream reply may quote — woven links pass :meth:`apply` unchanged.
        """
        return "\n".join(
            f"- {s.title or snippet(s.excerpt, MAX_EXCERPT_CHARS) or s.url} "
            f"{self._link(s.ref, s.url)}"
            for s in self._sources
            if s.url
        )

    def sources_footer(self) -> str:
        """Deterministic ``**Sources:** [[1]]…`` footer for user-facing output."""
        links = " ".join(self._link(s.ref, s.url) for s in self._sources if s.url)
        return f"**Sources:** {links}" if links else ""

    def apply(self, text: str) -> str:
        """Validate and render one model reply against the closed set.

        Order matters: anchored splices land first (offsets index the raw
        text), mangled links are canonicalized before ref weaving so a
        ``[mem:1](url)`` cannot leave a dangling ``(url)``, then bare numbers
        resolve, and finally all remaining citation-shaped residue is removed.
        """
        if not text:
            return text
        text = self._splice_anchored(text)
        text = self._canonicalize_links(text)
        text = self._weave_refs(text)
        text = self._resolve_bare_numbers(text)
        text = _RESIDUE_RE.sub("", text)
        return _CJK_RE.sub("", text)

    # -- internals ---------------------------------------------------------

    def _link(self, ref: str, url: str) -> str:
        label = ref.removeprefix("src:")
        return f"[[{label}]](<{url}>)"

    def _by_ref(self, ref: str) -> Citation | CitationSource | None:
        if ref.startswith("mem:"):
            return self._memory.get(ref)
        return next((s for s in self._sources if s.ref == ref), None)

    def _by_number(self, number: int) -> Citation | CitationSource | None:
        """Bare ``[N]`` resolves to link source N first, then memory ref N."""
        return self._by_ref(f"src:{number}") or self._by_ref(f"mem:{number}")

    def _splice_anchored(self, text: str) -> str:
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
            source = self._by_ref(ref)
            if source is not None and source.url:
                out = f"{out[:offset]} {self._link(ref, source.url)}{out[offset:]}"
        return out

    def _canonicalize_links(self, text: str) -> str:
        """Rewrite model-written ``[N](url)`` to the registered link for that ref.

        The ref claim is checked against the set; the written URL is untrusted
        and always replaced with the registered one. Constructs matching
        nothing in the set are deleted — URLs never originate from the model.
        """

        def sub(match: re.Match[str]) -> str:
            prefix, number, url = match.group(1), int(match.group(2)), match.group(3)
            source = self._by_ref(f"{prefix}{number}") if prefix else self._by_number(number)
            if source is None:
                source = next(
                    (s for s in self._sources if s.url == url),
                    next((c for c in self._memory.values() if c.url == url), None),
                )
            if source is None or not source.url:
                return ""
            return f" {self._link(source.ref, source.url)}"

        return _MANGLED_LINK_RE.sub(sub, text)

    def _weave_refs(self, text: str) -> str:
        def sub(match: re.Match[str]) -> str:
            ref = f"{match.group(1)}{match.group(2)}"
            source = self._by_ref(ref)
            if source is None or not source.url:
                return ""
            return self._link(ref, source.url)

        return _TAG_RE.sub(sub, text)

    def _resolve_bare_numbers(self, text: str) -> str:
        def sub(match: re.Match[str]) -> str:
            delimiter, number = match.group(1), int(match.group(2))
            source = self._by_number(number)
            if source is None or not source.url:
                # Invented marker: delete token and its separating space.
                return "" if delimiter.isspace() else delimiter
            return f"{delimiter}{self._link(source.ref, source.url)}"

        return _BARE_NUM_RE.sub(sub, text)


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
    "PROMPT_CONTRACT",
    "CitationRegistry",
    "CitationSource",
    "message_url",
    "render_fact_list",
]
