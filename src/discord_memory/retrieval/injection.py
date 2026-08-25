"""Injection block builder: budgeted, labeled, cite-tagged prompt section (§5.3)."""

from __future__ import annotations

from discord_memory.models.retrieval import Citation, ScoredFact, render_citation_tag

CHARS_PER_TOKEN = 4
MAX_FACT_CHARS = 280


def snippet(text: str, max_chars: int = MAX_FACT_CHARS) -> str:
    trimmed = text.strip().replace("\n", " ")
    if len(trimmed) > max_chars:
        return trimmed[: max_chars - 1] + "…"
    return trimmed


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars/token)."""
    return max(1, len(text) // CHARS_PER_TOKEN)


def message_url(guild_id: str, channel_id: str, message_id: str) -> str:
    """Discord jump link; channel sentinel 0 when unknown (still resolvable by client)."""
    return f"https://discord.com/channels/{guild_id}/{channel_id or '0'}/{message_id}"


class InjectionBuilder:
    """Packs scored facts into labeled sections with citation refs within a budget."""

    def build(
        self,
        *,
        asker_id: str,
        facts_by_section: dict[str, tuple[ScoredFact, ...]],
        summaries: dict[str, str | None],
        token_budget: int,
        guild_id: str,
    ) -> tuple[str, tuple[Citation, ...], bool]:
        """Returns ``(block, citations, trimmed)``.

        Sections render in dict order with headers:
        - ``CURRENT ASKER`` for the asking user's own profile,
        - ``REFERENCED USER`` per other subject — attribution-critical labels,
        - ``SERVER`` for community facts.
        """
        lines: list[str] = ["[MEMORY CONTEXT]", ""]
        citations: list[Citation] = []
        used_tokens = len("\n".join(lines)) // CHARS_PER_TOKEN
        trimmed = False

        header_map = {
            "asker": "WHAT I KNOW ABOUT THE CURRENT ASKER",
            "server": "SERVER COMMUNITY FACTS",
        }

        for section_key, facts in facts_by_section.items():
            if not facts and not (summaries.get(section_key)):
                continue
            subject_id = facts[0].fact.subject_id if facts else None
            if section_key == "asker":
                header = header_map["asker"]
                label = "Facts about the asker ONLY:"
            elif section_key == "server":
                header = header_map["server"]
                label = "Community-wide traits:"
            else:
                display = facts[0].fact.attribution.speaker_name or subject_id or section_key
                header = f"REFERENCED USER: {display}"
                label = f"Facts about {display} ONLY. Do NOT attribute these to the asker."
            summary_text = summaries.get(section_key)
            section_lines = [header]
            if summary_text:
                section_lines.append(f"Summary: {snippet(summary_text, 480)}")
            section_lines.append(label)
            section_tokens = estimate_tokens("\n".join(section_lines))
            if used_tokens + section_tokens > token_budget:
                trimmed = True
                continue

            emitted = 0
            for scored in facts:
                ref_number = len(citations) + 1
                fact_line = f"{render_citation_tag(ref_number)} {snippet(scored.fact.text)}"
                cost = estimate_tokens(fact_line)
                if used_tokens + section_tokens + cost > token_budget:
                    trimmed = True
                    break
                primary = _primary_citation(scored, guild_id, ref_number)
                if primary is not None:
                    citations.append(primary)
                    section_lines.append(fact_line)
                    section_tokens += cost
                    emitted += 1
                else:
                    section_lines.append(f"- {snippet(scored.fact.text)}")
                    section_tokens += cost
                    emitted += 1
            if emitted == 0 and not summary_text:
                continue
            lines.extend(section_lines)
            lines.append("")
            used_tokens += section_tokens

        block = "\n".join(lines).rstrip() + "\n"
        return block, tuple(citations), trimmed


def _primary_citation(scored: ScoredFact, guild_id: str, ref_number: int) -> Citation | None:
    record = scored.fact
    source_ref = next(
        (c for c in record.citations if c.role.value == "primary"),
        record.citations[0] if record.citations else None,
    )
    url = ""
    snippet_text = ""
    if source_ref is not None:
        url = source_ref.message_url or message_url(
            guild_id,
            source_ref.channel_id,
            source_ref.message_id,
        )
        snippet_text = source_ref.content_snippet
    elif record.subject_id:
        return None
    if not url:
        return None
    return Citation(
        ref=f"mem:{ref_number}",
        fact_id=record.id,
        url=url,
        snippet=snippet_text,
        subject_id=record.subject_id,
        subject_name=record.attribution.speaker_name or "",
    )


__all__ = ["InjectionBuilder", "estimate_tokens", "message_url", "snippet"]
