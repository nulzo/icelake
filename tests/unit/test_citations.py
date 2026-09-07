"""Closed-set registration, deterministic weaving, and structured attribution."""

from __future__ import annotations

import json

import pytest

from icelake import (
    AttributedClaim,
    Citations,
    ReplyAttribution,
    message_url,
    render_fact_list,
)
from icelake.attribution import attribute
from icelake.models import Citation, FactRecord, PromptContext, ScoredFact, SourceRef
from tests.conftest import ScriptedLLM

URL_MEM = "https://discord.com/channels/g1/c1/m1"
URL_A = "https://example.com/a"
URL_B = "https://example.com/b"


def _citation(ref: str, url: str = URL_MEM, fact_id: str = "fct_1") -> Citation:
    return Citation(
        ref=ref, fact_id=fact_id, url=url, snippet="source message", subject_name="klim"
    )


def _fact(fact_id: str, text: str) -> ScoredFact:
    return ScoredFact(fact=FactRecord(id=fact_id, guild_id="g1", text=text), score=0.9)


def _memory() -> Citations:
    return Citations([_citation("mem:1")], facts=[_fact("fct_1", "klim likes mayo")])


def _attribution(*claims: AttributedClaim) -> ReplyAttribution:
    return ReplyAttribution(claims=tuple(claims))


# --- closed set -------------------------------------------------------------


def test_from_prompt_context_seeds_memory_set_with_fact_text() -> None:
    ctx = PromptContext(
        injection_block="...",
        facts=(_fact("fct_1", "klim likes mayo"),),
        citations=(_citation("mem:1"),),
    )
    reg = Citations.from_prompt_context(ctx)
    assert [source.ref for source in reg.sources] == ["mem:1"]
    # The attributor must see the fact statement the answer model saw.
    assert reg.sources[0].excerpt == "klim likes mayo"
    assert reg.sources[0].kind == "memory"


def test_add_source_dedupes_by_url() -> None:
    reg = Citations()
    first = reg.add_source(URL_A, title="A")
    second = reg.add_source(URL_A, title="A again")
    assert first is second
    assert len(reg.sources) == 1


def test_add_message_builds_jump_url() -> None:
    reg = Citations()
    source = reg.add_message("g1", "c1", "m1", excerpt="hello")
    assert source.url == message_url("g1", "c1", "m1")
    assert source.kind == "message"


def test_citable_excludes_url_less_sources() -> None:
    reg = Citations([_citation("mem:1", url="")])
    reg.add_source(URL_A)
    assert len(reg.sources) == 2
    assert [source.url for source in reg.citable] == [URL_A]


# --- apply: deterministic weaving -------------------------------------------


def test_apply_weaves_attributed_claim_at_span_end() -> None:
    reg = _memory()
    text = "he likes mayo."
    claim = AttributedClaim(claim="he likes mayo", start=0, end=13, sources=(reg.sources[0],))
    assert reg.apply(text, _attribution(claim)) == f"he likes mayo [[1]](<{URL_MEM}>)."


def test_apply_splices_anchored_source_at_offset() -> None:
    reg = Citations()
    text = "the sky is blue"
    reg.add_source(URL_A, end_index=len(text))
    assert reg.apply(text) == f"the sky is blue [[1]](<{URL_A}>)"


def test_apply_combines_claims_and_anchors_back_to_front() -> None:
    reg = _memory()
    text = "mayo is great and the sky is blue"
    reg.add_source(URL_A, end_index=len(text))
    claim = AttributedClaim(claim="mayo is great", start=0, end=13, sources=(reg.sources[0],))
    out = reg.apply(text, _attribution(claim))
    assert out == f"mayo is great [[1]](<{URL_MEM}>) and the sky is blue [[2]](<{URL_A}>)"


def test_apply_labels_use_citable_positions() -> None:
    # A URL-less memory binding consumes no display number.
    reg = Citations([_citation("mem:1", url="")])
    reg.add_source(URL_A)
    reg.add_source(URL_B)
    text = "claim one and claim two"
    claims = _attribution(
        AttributedClaim(claim="claim one", start=0, end=9, sources=(reg.sources[1],)),
        AttributedClaim(claim="claim two", start=14, end=23, sources=(reg.sources[2],)),
    )
    assert reg.apply(text, claims) == (f"claim one [[1]](<{URL_A}>) and claim two [[2]](<{URL_B}>)")


def test_apply_without_attribution_or_anchors_returns_text() -> None:
    assert _memory().apply("plain reply") == "plain reply"


def test_apply_empty_text() -> None:
    assert _memory().apply("") == ""


def test_apply_skips_url_less_claim_sources() -> None:
    reg = Citations([_citation("mem:1", url="")])
    text = "a claim"
    claim = AttributedClaim(claim="a claim", start=0, end=7, sources=(reg.sources[0],))
    assert reg.apply(text, _attribution(claim)) == "a claim"


# --- attribute: structured claim mapping -------------------------------------


def _attribution_llm(payload: dict) -> ScriptedLLM:
    return ScriptedLLM({"attribution": json.dumps(payload)})


@pytest.mark.asyncio
async def test_attribute_maps_verbatim_claims_to_sources() -> None:
    reg = _memory()
    llm = _attribution_llm({"attributions": [{"claim": "he likes mayo", "sources": [1]}]})
    result = await attribute("he likes mayo for sure", reg, llm)
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert (claim.start, claim.end) == (0, 13)
    assert [source.url for source in claim.sources] == [URL_MEM]


@pytest.mark.asyncio
async def test_attribute_drops_non_verbatim_claims() -> None:
    reg = _memory()
    llm = _attribution_llm({"attributions": [{"claim": "he loves mayo", "sources": [1]}]})
    result = await attribute("he likes mayo", reg, llm)
    assert result.claims == ()


@pytest.mark.asyncio
async def test_attribute_drops_out_of_range_source_numbers() -> None:
    reg = _memory()
    llm = _attribution_llm({"attributions": [{"claim": "he likes mayo", "sources": [9]}]})
    result = await attribute("he likes mayo", reg, llm)
    assert result.claims == ()


@pytest.mark.asyncio
async def test_attribute_skips_llm_when_nothing_citable() -> None:
    llm = ScriptedLLM()
    result = await attribute("hello there", Citations(), llm)
    assert result.claims == ()
    assert llm.calls == []  # banter never pays for an attribution call


@pytest.mark.asyncio
async def test_attribute_empty_when_text_blank() -> None:
    llm = ScriptedLLM()
    result = await attribute("   ", _memory(), llm)
    assert result.claims == ()
    assert llm.calls == []


@pytest.mark.asyncio
async def test_attribute_invalid_json_returns_empty() -> None:
    llm = ScriptedLLM({"attribution": "not json at all"})
    result = await attribute("he likes mayo", _memory(), llm)
    assert result.claims == ()


@pytest.mark.asyncio
async def test_attribute_numbers_only_citable_sources() -> None:
    reg = Citations([_citation("mem:1", url="")])
    reg.add_source(URL_A, title="Doc A")
    llm = _attribution_llm({"attributions": [{"claim": "the claim", "sources": [1]}]})
    result = await attribute("the claim", reg, llm)
    # [1] in the attributor's list is the first CITABLE source, not mem:1.
    assert [source.url for source in result.claims[0].sources] == [URL_A]


def test_attribution_used_dedupes_first_use_order() -> None:
    reg = _memory()
    reg.add_source(URL_A)
    mem, web = reg.sources
    result = _attribution(
        AttributedClaim(claim="a", start=0, end=1, sources=(mem, web)),
        AttributedClaim(claim="b", start=2, end=3, sources=(mem,)),
    )
    assert [source.ref for source in result.used] == ["mem:1", "src:2"]


# --- render_fact_list ---------------------------------------------------------


def _record(
    text: str,
    *,
    message_id: str = "",
    channel_id: str = "",
    stored_url: str = "",
) -> FactRecord:
    sources = ()
    if message_id or stored_url:
        sources = (
            SourceRef(
                guild_id="g1",
                channel_id=channel_id,
                message_id=message_id,
                author_id="u1",
                message_url=stored_url,
            ),
        )
    return FactRecord(id="fct_x", guild_id="g1", text=text, citations=sources)


def test_render_fact_list_shares_number_pool_across_facts() -> None:
    out = render_fact_list(
        [
            _record("first", stored_url=URL_A),
            _record("second", stored_url=URL_B),
            _record("third", stored_url=URL_A),
        ]
    )
    assert out == (f"- first [[1]](<{URL_A}>)\n- second [[2]](<{URL_B}>)\n- third [[1]](<{URL_A}>)")


def test_render_fact_list_rebuilds_url_from_snowflakes() -> None:
    out = render_fact_list([_record("fact", message_id="m9", channel_id="c9")])
    assert out == f"- fact [[1]](<{message_url('g1', 'c9', 'm9')}>)"


def test_render_fact_list_bare_when_no_source() -> None:
    assert render_fact_list([_record("bare fact")]) == "- bare fact"
