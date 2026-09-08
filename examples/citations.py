"""Presenting citations: closed set, weave, custom footers, used-only.

Runnable WITHOUT Discord or an LLM. ``python examples/citations.py``

The golden reply path is three steps:

1. Build a closed set. ``prompt_context`` seeds it from retrieved facts.
   Tools and web results join through ``add_source`` / ``add_message``.
2. Generate plain prose. The answer model never sees tags or URLs.
3. ``memory.cite(reply, set)`` matches reply spans to the set with one
   batched embed and weaves Discord-safe ``[[N]](<url>)`` at the matches.

Presentation beyond that weave is yours. Inline only, a numbered footer,
both, or nothing. Numbers always come from ``Citations.citable`` so the
inline label and the footer agree. A retrieved-but-unused source never
renders. Banter with no match is left untouched.

``facts.remember`` does not stamp Discord jump links (there is no source
message). Observed/extracted facts do. This file registers sources itself
so the presentation layer can run without an extractor.
"""

from __future__ import annotations

import asyncio

from icelake import (
    Citations,
    CitationSource,
    FactRecord,
    Memory,
    MemoryConfig,
    SourceRef,
    fact_source_urls,
    message_url,
    render_fact_list,
)

GUILD = "555"
CHANNEL = "777"
ALICE = "100000000000000001"

RUST_MSG = "9001"
MINECRAFT_MSG = "9002"
RUST_URL = message_url(GUILD, CHANNEL, RUST_MSG)
MINECRAFT_URL = message_url(GUILD, CHANNEL, MINECRAFT_MSG)
BOOK_URL = "https://doc.rust-lang.org/book/"

RUST_FACT = "alice has been learning rust for about a year"
MINECRAFT_FACT = "bob hosts sunday minecraft builds"


def _footer(text: str, citations: Citations, used: tuple[CitationSource, ...]) -> str:
    """Numbered source list whose labels match the inline weave.

    ``cited.sources`` is first-use order. Re-numbering that tuple as 1, 2, 3
    desyncs the footer from ``[[N]]`` in the reply. Walk ``citable`` instead
    and keep the original positions, dropping unused rows.
    """
    used_refs = {source.ref for source in used}
    lines = [text, "", "Sources"]
    for number, source in enumerate(citations.citable, 1):
        if source.ref not in used_refs:
            continue
        label = source.title or source.kind
        lines.append(f"[{number}] {label}  {source.url}")
    return "\n".join(lines)


def demo_closed_set() -> None:
    print("=== 1. Closed set: code registers every source, URLs dedupe ===")
    citations = Citations()
    first = citations.add_source(BOOK_URL, title="The Rust Book", excerpt="Rust language")
    again = citations.add_source(BOOK_URL, title="ignored, URL already known")
    discord = citations.add_message(GUILD, CHANNEL, RUST_MSG, title="alice", excerpt=RUST_FACT)
    print(f"  same URL returns the same source: {first is again}")
    print(f"  refs: {[source.ref for source in citations.sources]}")
    print(f"  kinds: {[source.kind for source in citations.sources]}")
    print(f"  discord jump: {discord.url}")
    print()


def demo_provider_anchors() -> None:
    print("=== 2. Provider-anchored web citations (no embed, no LLM) ===")
    text = "See the official book for ownership."
    citations = Citations()
    citations.add_source(BOOK_URL, title="The Rust Book", end_index=len(text))
    woven = citations.apply(text)
    print(f"  {woven}")
    print("  end_index is a character offset from the provider (OpenAI url_citation).")
    print("  anchored sources splice directly. attribution is not involved.")
    print()


async def demo_cite(memory: Memory) -> None:
    print("=== 3. memory.cite: used-only, unmatched stays bare ===")
    citations = Citations()
    citations.add_source(RUST_URL, title="alice", excerpt=RUST_FACT, kind="memory")
    citations.add_source(
        BOOK_URL,
        title="The Rust Book",
        excerpt="Rust is a language empowering everyone to build reliable software",
        kind="web",
    )
    citations.add_source(MINECRAFT_URL, title="bob", excerpt=MINECRAFT_FACT, kind="memory")

    reply = (
        "You've been learning rust for about a year. bob hosts sunday minecraft builds. lol yeah."
    )
    cited = await memory.cite(reply, citations)
    print(f"  woven:\n    {cited.text}")
    print(f"  claims: {[claim.claim for claim in cited.claims]}")
    print(f"  used:   {[source.title for source in cited.sources]}")
    print("  the rust book never matched a span, so it is not cited.")
    print("  'lol yeah' matched nothing, so it stays bare.")
    print()

    print("=== 4. Custom rendering (weave=False + footer) ===")
    raw = await memory.cite(reply, citations, weave=False)
    print(_footer(raw.text, citations, raw.sources))
    print()
    print("  inline + footer together, numbers kept in sync:")
    print(_footer(cited.text, citations, cited.sources))
    print()


async def demo_paraphrase_limit(memory: Memory) -> None:
    print("=== 5. Hashing embeddings are lexical. Persona paraphrase can miss ===")
    citations = Citations()
    citations.add_source(
        RUST_URL,
        title="alice",
        excerpt=RUST_FACT,
        kind="memory",
    )
    near = await memory.cite("You've been learning rust for about a year.", citations)
    voice = await memory.cite("bro has been grinding rust for like a year now", citations)
    print(f"  near-verbatim cited: {bool(near.sources)}  ->  {near.text}")
    print(f"  persona voice cited: {bool(voice.sources)}  ->  {voice.text}")
    print("  default hashing is fine for tests. a live bot should set")
    print('  embeddings="openai://..." or embeddings="local" so paraphrase still matches.')
    print()


def demo_fact_list() -> None:
    print("=== 6. /memory show lists: render_fact_list shares one number pool ===")
    rust = FactRecord(
        id="fct_1",
        guild_id=GUILD,
        subject_id=ALICE,
        text=RUST_FACT,
        citations=(
            SourceRef(
                message_id=RUST_MSG,
                channel_id=CHANNEL,
                guild_id=GUILD,
                author_id=ALICE,
                message_url=RUST_URL,
            ),
        ),
    )
    minecraft = FactRecord(
        id="fct_2",
        guild_id=GUILD,
        subject_id=ALICE,
        text=MINECRAFT_FACT,
        citations=(
            SourceRef(
                message_id=MINECRAFT_MSG,
                channel_id=CHANNEL,
                guild_id=GUILD,
                author_id=ALICE,
                message_url=MINECRAFT_URL,
            ),
        ),
    )
    restated = FactRecord(
        id="fct_3",
        guild_id=GUILD,
        subject_id=ALICE,
        text="alice is still on that rust learning streak",
        citations=rust.citations,
    )
    print(render_fact_list((rust, minecraft, restated)))
    print(f"  fact_source_urls(rust) = {fact_source_urls(rust)}")
    print("  two facts pointing at the same message keep number [1].")
    print()


def demo_scale() -> None:
    print("=== 7. Reliable at scale ===")
    print("  one Citations object per reply, discarded after send")
    print("  register this turn's sources only (retrieved facts + tool results)")
    print("  attribution is one embed batch, no LLM, no metered tokens")
    print("  caps: 12 claim spans, 3 sources per span")
    print("  same URL keeps one number no matter how many spans point at it")
    print("  used-only: retrieved-but-unused memories never render")
    print("  embedder failure or no match: send the reply uncited, never invent a link")
    print("  hashing embeddings miss paraphrase. use a real embedder in production")
    print("  footer numbers must walk Citations.citable, not re-index cited.sources")


async def main() -> None:
    demo_closed_set()
    demo_provider_anchors()
    memory = Memory(MemoryConfig(storage="sqlite://:memory:", llm=None))
    async with memory:
        await demo_cite(memory)
        await demo_paraphrase_limit(memory)
    demo_fact_list()
    demo_scale()


if __name__ == "__main__":
    asyncio.run(main())
