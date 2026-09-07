"""Citations: closed-set parsing, deny-by-default validation, structured output.

These tests pin the reliability contract: the model may only *reference* the
registered set, every citation-shaped token outside it is deleted, and
``parse`` returns structured data (clean text + resolved sources) with
presentation left to the consumer.
"""

from icelake import (
    Citation,
    Citations,
    FactRecord,
    MarkerMode,
    PromptContext,
    SourceRef,
)
from icelake.citations import message_url, render_fact_list

GUILD = "111"
URL_A = "https://example.com/a"
URL_B = "https://example.com/b"
URL_MEM = "https://discord.com/channels/111/222/333"


def _memory() -> Citations:
    return Citations(
        citations=(
            Citation(ref="mem:1", fact_id="fct_a", url=URL_MEM),
            Citation(ref="mem:2", fact_id="fct_b", url=""),  # unresolvable
        )
    )


def _fact(url: str = "", text: str = "likes tea") -> FactRecord:
    citations = (
        (
            SourceRef(
                message_id="333",
                channel_id="222",
                guild_id=GUILD,
                author_id="1",
                message_url=url,
            ),
        )
        if url
        else ()
    )
    return FactRecord(
        id="fct_x",
        guild_id=GUILD,
        subject_id="42",
        text=text,
        citations=citations,
    )


# --- registration ---------------------------------------------------------


def test_add_source_mints_sequential_refs() -> None:
    reg = Citations()
    first = reg.add_source(URL_A, title="A")
    second = reg.add_source(URL_B)
    assert first.ref == "src:1"
    assert second.ref == "src:2"
    assert [s.ref for s in reg.sources] == ["src:1", "src:2"]


def test_memory_and_link_sources_share_one_registry() -> None:
    reg = _memory()
    web = reg.add_source(URL_A)
    assert [s.ref for s in reg.sources] == ["mem:1", "mem:2", "src:3"]
    assert web.ref == "src:3"
    assert reg.sources[0].kind == "memory"
    assert web.kind == "web"


def test_add_source_dedupes_by_url_and_merges_anchors() -> None:
    reg = Citations()
    first = reg.add_source(URL_A, end_index=5)
    again = reg.add_source(URL_A, end_index=20)
    assert again is first
    assert len(reg.sources) == 1
    out = reg.apply("0123456789012345678901234")
    assert out.count(f"[[src:1]](<{URL_A}>)") == 2  # both spans anchored to one number


def test_add_message_builds_jump_link() -> None:
    reg = Citations()
    source = reg.add_message(GUILD, "222", "333", excerpt="hello")
    assert source.url == URL_MEM
    assert source.kind == "message"


def test_message_url_sentinel_channel() -> None:
    assert message_url("g", "", "m") == "https://discord.com/channels/g/0/m"


# --- prompt side ----------------------------------------------------------


def test_instructions_empty_without_sources() -> None:
    assert Citations().instructions == ""
    assert _memory().instructions == ""  # mem contract ships in the block


def test_instructions_and_source_list_for_link_sources() -> None:
    reg = Citations()
    reg.add_source(URL_A, title="Python docs", excerpt="all about python")
    contract = reg.instructions
    assert "[src:N]" in contract
    assert "never" in contract
    listing = reg.source_list()
    assert "[src:1]" in listing
    assert "Python docs — all about python" in listing
    assert URL_A not in listing  # URLs withheld from the prompt


# --- apply: Discord convenience weaving ------------------------------------


def test_weaves_echoed_ref() -> None:
    reg = Citations()
    reg.add_source(URL_A)
    out = reg.apply("see the docs [src:1] for details")
    assert out == f"see the docs [[src:1]](<{URL_A}>) for details"


def test_weaves_echoed_mem_ref_and_strips_url_less() -> None:
    out = _memory().apply("they game [mem:1] a lot [mem:2] and [mem:9] gone")
    assert f"[[mem:1]](<{URL_MEM}>)" in out
    assert "mem:2" not in out
    assert "mem:9" not in out


def test_mixed_memory_and_link_sources_coexist() -> None:
    reg = _memory()
    reg.add_source(URL_A)
    out = reg.apply("memory [mem:1] plus web [src:3]")
    assert f"[[mem:1]](<{URL_MEM}>)" in out
    assert f"[[src:3]](<{URL_A}>)" in out


# --- deny-by-default ------------------------------------------------------


def test_invented_bare_number_is_stripped() -> None:
    # bare [N] is not a citation tag — it is prose and survives untouched
    assert _memory().apply("prefers algebra [5]") == "prefers algebra [5]"


def test_unknown_namespaced_ref_is_stripped() -> None:
    reg = Citations()
    reg.add_source(URL_A)
    assert reg.apply("as reported [src:9] today") == "as reported today"


def test_prose_brackets_survive() -> None:
    reg = _memory()
    reg.add_source(URL_A)
    text = "array[0] and matrix[i][2] and [not a citation] and nums[10]"
    assert reg.apply(text) == text


def test_model_written_link_with_registered_url_is_canonicalized() -> None:
    reg = Citations()
    reg.add_source(URL_A)
    out = reg.apply(f"read it [src:1]({URL_A}) now")
    assert out == f"read it [[src:1]](<{URL_A}>) now"


def test_model_written_link_with_wrong_url_uses_registered_url() -> None:
    reg = Citations()
    reg.add_source(URL_A)
    out = reg.apply("read it [src:1](https://evil.example) now")
    assert out == f"read it [[src:1]](<{URL_A}>) now"


def test_model_written_link_matching_nothing_is_deleted() -> None:
    reg = Citations()
    reg.add_source(URL_A)
    out = reg.apply("read it [src:7](https://invented.example/x) now")
    assert out == "read it now"


def test_free_standing_markdown_links_survive() -> None:
    reg = Citations()
    reg.add_source(URL_A)
    text = "the [rust book](https://doc.rust-lang.org/book) covers it"
    assert reg.apply(text) == text


def test_cjk_citation_artifacts_stripped() -> None:
    reg = _memory()
    assert reg.apply("claims things【4†source】here") == "claims thingshere"
    assert reg.apply("claims things 【4:0†source】 here") == "claims things here"


def test_non_numeric_mem_residue_stripped() -> None:
    out = _memory().apply("YOUR man [mem: facts]. also [mem:1]")
    assert out == f"YOUR man. also [[mem:1]](<{URL_MEM}>)"


def test_already_woven_links_are_idempotent() -> None:
    reg = _memory()
    reg.add_source(URL_A)
    text = f"klim [[mem:1]](<{URL_MEM}>) and [[src:3]](<{URL_A}>) leftover"
    assert reg.apply(text) == text
    assert reg.apply(reg.apply("they game [mem:1]")) == reg.apply("they game [mem:1]")


# --- anchored (provider offset) splicing -----------------------------------


def test_anchored_source_splices_at_offset() -> None:
    reg = Citations()
    text = "OpenAI released GPT-5 today and stocks moved."
    reg.add_source(URL_A, end_index=text.index(" today"))
    out = reg.apply(text)
    assert out == f"OpenAI released GPT-5 [[src:1]](<{URL_A}>) today and stocks moved."


def test_anchored_sources_dedupe_shared_offsets() -> None:
    reg = Citations()
    reg.add_source(URL_A, end_index=10)
    reg.add_source(URL_B, end_index=10)
    out = reg.apply("0123456789 rest")
    assert out.count("[[") == 1


def test_anchored_offsets_beyond_text_clamp() -> None:
    reg = Citations()
    reg.add_source(URL_A, end_index=10_000)
    out = reg.apply("short text")
    assert out.endswith(f" [[src:1]](<{URL_A}>)")


def test_anchored_and_echoed_render_together() -> None:
    reg = _memory()
    reg.add_source(URL_A, end_index=5)
    out = reg.apply("hello world [mem:1]")
    assert out.startswith(f"hello [[src:3]](<{URL_A}>) world")
    assert f"[[mem:1]](<{URL_MEM}>)" in out


# --- parse: structured output ----------------------------------------------


def test_parse_strip_removes_markers_and_resolves_used() -> None:
    reg = Citations()
    reg.add_source(URL_A, title="A")
    reg.add_source(URL_B, title="B")
    parsed = reg.parse("alpha [src:1] beta [src:2] gamma", markers=MarkerMode.STRIP)
    assert parsed.text == "alpha beta gamma"
    assert [c.ref for c in parsed.citations] == ["src:1", "src:2"]
    assert parsed.citations[0].url == URL_A
    assert parsed.citations[1].title == "B"


def test_parse_keep_leaves_markers_for_consumer_rendering() -> None:
    reg = Citations()
    reg.add_source(URL_A)
    parsed = reg.parse("see [src:1] here", markers=MarkerMode.KEEP)
    assert parsed.text == "see [src:1] here"
    assert [c.url for c in parsed.citations] == [URL_A]


def test_parse_drops_unknown_and_dedupes() -> None:
    reg = Citations()
    reg.add_source(URL_A)
    parsed = reg.parse("a [src:1] b [src:9] c [src:1] d")
    assert [c.ref for c in parsed.citations] == ["src:1"]
    assert "src:9" not in parsed.text


def test_parse_memory_source_carries_provenance() -> None:
    parsed = _memory().parse("they game [mem:1] a lot")
    assert len(parsed.citations) == 1
    used = parsed.citations[0]
    assert used.kind == "memory"
    assert used.fact_id == "fct_a"
    assert used.url == URL_MEM


def test_parse_passthrough_for_empty_and_plain_text() -> None:
    reg = _memory()
    assert reg.parse("").text == ""
    plain = reg.parse("plain reply, no citations")
    assert plain.text == "plain reply, no citations"
    assert plain.citations == ()


# --- fact list rendering ---------------------------------------------------


def test_render_fact_list_weaves_jump_links() -> None:
    out = render_fact_list([_fact(url=URL_MEM), _fact(text="orphan fact")])
    assert out == f"- likes tea [[1]](<{URL_MEM}>)\n- orphan fact"


def test_render_fact_list_shares_numbers_across_facts() -> None:
    out = render_fact_list([_fact(url=URL_MEM, text="a"), _fact(url=URL_MEM, text="b")])
    assert out == f"- a [[1]](<{URL_MEM}>)\n- b [[1]](<{URL_MEM}>)"


def test_from_prompt_context_seeds_memory_set() -> None:
    ctx = PromptContext(
        injection_block="",
        citations=(Citation(ref="mem:1", fact_id="f", url=URL_MEM),),
    )
    reg = Citations.from_prompt_context(ctx)
    assert reg.apply("x [mem:1]") == f"x [[mem:1]](<{URL_MEM}>)"
