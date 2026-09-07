"""CitationRegistry: closed-set rendering, deny-by-default validation.

These tests pin the reliability contract: the model may only *reference* the
registered set, and every citation-shaped token outside it is deleted.
"""

from icelake import Citation, CitationRegistry, FactRecord, SourceRef
from icelake.citations import message_url, render_fact_list

GUILD = "111"
URL_A = "https://example.com/a"
URL_B = "https://example.com/b"
URL_MEM = "https://discord.com/channels/111/222/333"


def _memory_registry() -> CitationRegistry:
    return CitationRegistry(
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
    registry = CitationRegistry()
    first = registry.add_source(URL_A, title="A")
    second = registry.add_source(URL_B)
    assert first.ref == "src:1"
    assert second.ref == "src:2"
    assert [s.ref for s in registry.sources] == ["src:1", "src:2"]


def test_add_source_dedupes_by_url_and_merges_anchors() -> None:
    registry = CitationRegistry()
    first = registry.add_source(URL_A, end_index=5)
    again = registry.add_source(URL_A, end_index=20)
    assert again is first
    assert len(registry.sources) == 1
    out = registry.apply("0123456789012345678901234")
    assert out.count(f"[[1]](<{URL_A}>)") == 2  # both spans anchored to one number


def test_add_message_builds_jump_link() -> None:
    registry = CitationRegistry()
    source = registry.add_message(GUILD, "222", "333", excerpt="hello")
    assert source.url == URL_MEM


def test_message_url_sentinel_channel() -> None:
    assert message_url("g", "", "m") == "https://discord.com/channels/g/0/m"


# --- prompt side ----------------------------------------------------------


def test_prompt_contract_empty_without_sources() -> None:
    assert CitationRegistry().prompt_contract == ""
    assert _memory_registry().prompt_contract == ""  # mem contract ships in the block


def test_prompt_contract_and_source_list_for_link_sources() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A, title="Python docs", excerpt="all about python")
    contract = registry.prompt_contract
    assert "[src:N]" in contract
    assert "never" in contract
    listing = registry.source_list()
    assert "[src:1]" in listing
    assert "Python docs — all about python" in listing
    assert URL_A not in listing  # URLs withheld from the prompt


# --- echo weaving ---------------------------------------------------------


def test_weaves_echoed_src_ref() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A)
    out = registry.apply("see the docs [src:1] for details")
    assert out == f"see the docs [[1]](<{URL_A}>) for details"


def test_weaves_echoed_mem_ref_and_strips_url_less() -> None:
    out = _memory_registry().apply("they game [mem:1] a lot [mem:2] and [mem:9] gone")
    assert f"[[mem:1]](<{URL_MEM}>)" in out
    assert "mem:2" not in out
    assert "mem:9" not in out


def test_mixed_memory_and_link_sources_coexist() -> None:
    registry = _memory_registry()
    registry.add_source(URL_A)
    out = registry.apply("memory [mem:1] plus web [src:1]")
    assert f"[[mem:1]](<{URL_MEM}>)" in out
    assert f"[[1]](<{URL_A}>)" in out


# --- deny-by-default ------------------------------------------------------


def test_invented_bare_number_is_stripped() -> None:
    registry = _memory_registry()
    assert registry.apply("prefers algebra [5]") == "prefers algebra"


def test_bare_number_upgrades_to_known_link_source() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A)
    registry.add_source(URL_B)
    out = registry.apply("as reported [2] today")
    assert out == f"as reported [[2]](<{URL_B}>) today"


def test_bare_number_upgrades_to_known_memory_ref() -> None:
    out = _memory_registry().apply("likes tea [1] apparently")
    assert out == f"likes tea [[mem:1]](<{URL_MEM}>) apparently"


def test_prose_brackets_survive() -> None:
    registry = _memory_registry()
    registry.add_source(URL_A)
    text = "array[0] and matrix[i][2] and [not a citation] and nums[10]"
    assert registry.apply(text) == text


def test_model_written_link_with_registered_url_is_canonicalized() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A)
    out = registry.apply(f"read it [1]({URL_A}) now")
    assert out == f"read it [[1]](<{URL_A}>) now"


def test_model_written_link_with_wrong_url_uses_registered_url() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A)
    out = registry.apply("read it [src:1](https://evil.example) now")
    assert out == f"read it [[1]](<{URL_A}>) now"


def test_model_written_link_matching_nothing_is_deleted() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A)
    out = registry.apply("read it [7](https://invented.example/x) now")
    assert out == "read it now"


def test_free_standing_markdown_links_survive() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A)
    text = "the [rust book](https://doc.rust-lang.org/book) covers it"
    assert registry.apply(text) == text


def test_cjk_citation_artifacts_stripped() -> None:
    registry = _memory_registry()
    assert registry.apply("claims things【4†source】here") == "claims thingshere"
    assert registry.apply("claims things 【4:0†source】 here") == "claims things here"


def test_non_numeric_mem_residue_stripped() -> None:
    out = _memory_registry().apply("YOUR man [mem: facts]. also [mem:1]")
    assert out == f"YOUR man. also [[mem:1]](<{URL_MEM}>)"


def test_already_woven_links_are_idempotent() -> None:
    registry = _memory_registry()
    registry.add_source(URL_A)
    text = f"klim [[mem:1]](<{URL_MEM}>) and [[1]](<{URL_A}>) leftover"
    assert registry.apply(text) == text
    assert registry.apply(registry.apply("they game [mem:1]")) == registry.apply(
        "they game [mem:1]"
    )


# --- anchored (provider offset) splicing -----------------------------------


def test_anchored_source_splices_at_offset() -> None:
    registry = CitationRegistry()
    text = "OpenAI released GPT-5 today and stocks moved."
    registry.add_source(URL_A, end_index=text.index(" today"))
    out = registry.apply(text)
    assert out == f"OpenAI released GPT-5 [[1]](<{URL_A}>) today and stocks moved."


def test_anchored_sources_dedupe_shared_offsets() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A, end_index=10)
    registry.add_source(URL_B, end_index=10)
    out = registry.apply("0123456789 rest")
    assert out.count("[[") == 1


def test_anchored_offsets_beyond_text_clamp() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A, end_index=10_000)
    out = registry.apply("short text")
    assert out.endswith(f" [[1]](<{URL_A}>)")


def test_anchored_and_echoed_render_together() -> None:
    registry = _memory_registry()
    registry.add_source(URL_A, end_index=5)
    out = registry.apply("hello world [mem:1]")
    assert out.startswith(f"hello [[1]](<{URL_A}>) world")
    assert f"[[mem:1]](<{URL_MEM}>)" in out


# --- footer ----------------------------------------------------------------


def test_sources_footer_lists_link_sources() -> None:
    registry = CitationRegistry()
    registry.add_source(URL_A)
    registry.add_source(URL_B)
    assert registry.sources_footer() == (f"**Sources:** [[1]](<{URL_A}>) [[2]](<{URL_B}>)")


def test_sources_footer_empty_without_sources() -> None:
    assert CitationRegistry().sources_footer() == ""
    assert _memory_registry().sources_footer() == ""


# --- fact list rendering ---------------------------------------------------


def test_render_fact_list_weaves_jump_links() -> None:
    out = render_fact_list([_fact(url=URL_MEM), _fact(text="orphan fact")])
    assert out == f"- likes tea [[1]](<{URL_MEM}>)\n- orphan fact"


def test_render_fact_list_shares_numbers_across_facts() -> None:
    out = render_fact_list([_fact(url=URL_MEM, text="a"), _fact(url=URL_MEM, text="b")])
    assert out == f"- a [[1]](<{URL_MEM}>)\n- b [[1]](<{URL_MEM}>)"


def test_from_prompt_context_seeds_memory_set() -> None:
    from icelake import PromptContext

    ctx = PromptContext(
        injection_block="",
        citations=(Citation(ref="mem:1", fact_id="f", url=URL_MEM),),
    )
    registry = CitationRegistry.from_prompt_context(ctx)
    assert registry.apply("x [mem:1]") == f"x [[mem:1]](<{URL_MEM}>)"


def test_apply_passthrough_for_empty_and_plain_text() -> None:
    registry = _memory_registry()
    assert registry.apply("") == ""
    assert registry.apply("plain reply, no citations") == "plain reply, no citations"
