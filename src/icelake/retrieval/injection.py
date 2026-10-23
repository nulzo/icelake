"""Injection block builder: budgeted, labeled prompt section (§5.3).

Facts render as plain prose — the answer model never sees citation tags or
URLs. Provenance is bound as data (``Citation`` objects on the returned
context) and reattached post-generation by the attribution stage.
"""

from __future__ import annotations

from icelake.models.facts import AttributionType
from icelake.models.retrieval import Citation, ScoredFact

CHARS_PER_TOKEN = 4

MEMORY_VS_CONVERSATION = (
    "CRITICAL: this block is LONG-TERM MEMORY accumulated over time — background "
    "knowledge about people and this server. It is NOT the live chat; the message "
    "history in the conversation turns is what is actually happening now. Do not "
    "treat stored facts as things that just happened."
)

USAGE_GUIDELINES = (
    "Use these memories naturally: personalize replies, remember preferences, "
    "resolve who people are. Never recite random trivia unless it fits the "
    "exchange. Facts about one person must never be attributed to another."
)
MAX_FACT_CHARS = 280


def snippet(text: str, max_chars: int = MAX_FACT_CHARS) -> str:
    trimmed = text.strip().replace("\n", " ")
    if len(trimmed) > max_chars:
        return trimmed[: max_chars - 1] + "…"
    return trimmed


def estimate_tokens(text: str) -> int:
    """Cheap deterministic token estimate (~4 chars/token)."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


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
        alias_notes: dict[str, str] | None = None,
        display_names: dict[str, str] | None = None,
    ) -> tuple[str, tuple[Citation, ...], bool]:
        """``alias_notes`` maps section-key -> coreference line ("Also known as…").
        ``display_names`` maps section-key -> the subject's resolved name for
        referenced-user headers. Referenced-user labels fall back to the
        canonical subject id — never to the speaker name stored on a fact.
        Supply ``display_names`` for every referenced subject to render names
        instead of ids.

        Returns ``(block, citations, trimmed)``.

        Sections render in dict order with headers:
        - ``CURRENT ASKER`` for the asking user's own profile,
        - ``REFERENCED USER`` per other subject — attribution-critical labels,
        - ``SERVER`` for community facts.
        """
        lines: list[str] = [
            "[MEMORY CONTEXT]",
            "",
            MEMORY_VS_CONVERSATION,
            "",
        ]
        citations: list[Citation] = []

        def fits(candidate_lines: list[str]) -> bool:
            block = "\n".join([*candidate_lines, "", USAGE_GUIDELINES]).rstrip() + "\n"
            return estimate_tokens(block) <= token_budget

        if not fits(lines):
            return "", (), True
        trimmed = False

        header_map = {
            "asker": "WHAT I KNOW ABOUT THE CURRENT ASKER",
            "server": "SERVER COMMUNITY FACTS",
        }

        for section_key, facts in facts_by_section.items():
            if not facts:
                continue
            subject_id = facts[0].fact.subject_id if facts else None
            if section_key == "asker":
                header = header_map["asker"]
                label = "Facts about the asker ONLY:"
            elif section_key == "server":
                header = header_map["server"]
                label = "Community-wide traits:"
            else:
                display = (display_names or {}).get(section_key) or subject_id or section_key
                header = f"REFERENCED USER: {display}"
                label = f"Facts about {display} ONLY. Do NOT attribute these to the asker."
            # Summaries are navigation aids, never independent evidence. Only
            # the selected facts are injected, each with its own provenance.
            section_lines = [header]
            alias_note = (alias_notes or {}).get(section_key)
            if alias_note:
                section_lines.append(alias_note)
            section_lines.append(label)
            if not fits([*lines, *section_lines, ""]):
                trimmed = True
                continue

            display = (display_names or {}).get(section_key, "")
            emitted = 0
            for scored in facts:
                ref_number = len(citations) + 1
                speaker = scored.fact.attribution.speaker_name or scored.fact.attribution.speaker_id
                qualifier = (
                    f" [reported by {speaker}]"
                    if scored.fact.attribution.type is AttributionType.THIRD_PARTY
                    else ""
                )
                fact_line = f"- {snippet(scored.fact.text)}{qualifier}"
                if not fits([*lines, *section_lines, fact_line, ""]):
                    trimmed = True
                    break
                citations.extend(fact_citations(scored, guild_id, ref_number, display_name=display))
                section_lines.append(fact_line)
                emitted += 1
            if emitted == 0:
                continue
            lines.extend(section_lines)
            lines.append("")

        lines.append("")
        lines.append(USAGE_GUIDELINES)
        block = "\n".join(lines).rstrip() + "\n"
        return block, tuple(citations), trimmed


def fact_citations(
    scored: ScoredFact,
    guild_id: str,
    ref_number: int = 1,
    *,
    display_name: str = "",
) -> tuple[Citation, ...]:
    """Bind every supporting message, preserving the original speaker and quote."""
    record = scored.fact
    return tuple(
        Citation(
            ref=f"mem:{ref_number + index}",
            fact_id=record.id,
            url=source.message_url or message_url(guild_id, source.channel_id, source.message_id),
            snippet=source.content_snippet,
            subject_id=record.subject_id,
            subject_name=display_name or record.subject_id or "",
            message_id=source.message_id,
            channel_id=source.channel_id,
            author_id=source.author_id,
            author_name=source.author_name,
            created_at=source.created_at,
            attribution=record.attribution,
            score=scored.score,
            rerank_score=scored.rerank_score,
        )
        for index, source in enumerate(record.citations)
        if source.message_id and source.guild_id == guild_id
    )


__all__ = ["InjectionBuilder", "estimate_tokens", "fact_citations", "message_url", "snippet"]
