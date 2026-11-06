"""Conversational names derived from active, audience-visible self reports.

Aliases identify accounts; they are not consent to address someone by a name.
This projection uses original statements and inherits the owning facts' lifecycle.
It deliberately abstains on indirect, quoted or ambiguous introductions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from icelake.models.facts import AttributionType, FactRecord


class NameKind(StrEnum):
    PREFERRED = "preferred"
    STATED = "stated"
    AVOID = "avoid"
    DISCORD = "discord"


@dataclass(frozen=True)
class NameStatement:
    kind: NameKind
    name: str = ""


@dataclass(frozen=True)
class NameEvidence:
    statement: NameStatement
    fact: FactRecord
    at: datetime
    message_id: str


_PREFIX = r"(?:(?:hey|hi|btw|actually)[,!]?\s+)?(?:please\s+)?"
_PATTERNS = (
    (NameKind.PREFERRED, r"(?:call me|i go by|i prefer to be called)"),
    (NameKind.STATED, r"my (?:(?:real|first|full) )?name(?: is|'s|\u2019s)"),
    (NameKind.AVOID, r"(?:do not call me|don't call me|don\u2019t call me|stop calling me)"),
)
_NAME_FACT = re.compile(r"\b(?:name|named|called|call|goes by|go by)\b", re.IGNORECASE)
_FALLBACK = re.compile(
    _PREFIX + r"(?:use|call me by) my (?:discord |display |user)name[.!]?", re.IGNORECASE
)


def name_statements(text: str) -> tuple[NameStatement, ...]:
    """Recognize direct English self-introductions, never arbitrary name mentions."""
    text = re.sub(r"^(?:<@!?\d+>\s*)+", "", text.strip())
    text = re.sub(r"^(?:hey|hi|btw|actually)[,!]?\s+", "", text, flags=re.IGNORECASE)
    if any(char in text for char in ("\n", "\r", '"', "“", "”", "`", ">", "?")):
        return ()
    statements = []
    for clause in re.split(r"[,;]\s*(?:but\s+)?", text, flags=re.IGNORECASE):
        if _FALLBACK.fullmatch(clause):
            statements.append(NameStatement(NameKind.DISCORD))
            continue
        for kind, marker in _PATTERNS:
            match = re.fullmatch(
                _PREFIX + marker + r"\s+(.+?)(?:\s+(?:please|btw|from now on|now))?[.!]?",
                clause,
                re.IGNORECASE,
            )
            if match is None:
                continue
            name = match[1].strip()
            if (
                1 <= len(name) <= 64
                and not name.casefold().startswith(("not ", "no longer "))
                and len(name.split()) <= 4
                and any(char.isalpha() for char in name)
                and all(char.isalnum() or char in " '-\u2019._" for char in name)
            ):
                statements.append(NameStatement(kind, name))
                break
        else:
            # Every clause must be a declaration: no quoted stories or hypotheticals.
            return ()
    return tuple(statements)


def name_evidence(fact: FactRecord) -> tuple[NameEvidence, ...]:
    """Require a name fact and a matching original statement by its own subject."""
    if (
        not fact.subject_id
        or fact.attribution.type not in {AttributionType.SELF, AttributionType.MANUAL}
        or fact.attribution.speaker_id not in {None, fact.subject_id}
        or not _NAME_FACT.search(fact.text)
    ):
        return ()
    evidence = []
    for source in fact.citations:
        if source.author_id != fact.subject_id or source.guild_id != fact.guild_id:
            continue
        for statement in name_statements(source.content_snippet):
            if statement.name and statement.name.casefold() not in fact.text.casefold():
                continue
            at = source.created_at or fact.observed_at or fact.created_at
            if at is not None:
                evidence.append(
                    NameEvidence(
                        statement, fact, at.replace(tzinfo=at.tzinfo or UTC), source.message_id
                    )
                )
    return tuple(evidence)


def choose_name(records: tuple[FactRecord, ...]) -> NameEvidence | None:
    """Prefer explicit addressing preferences, respecting newer withdrawals.

    The chronological scan stays together so withdrawals, resets and older
    preferences share one state transition order rather than separate filters.
    """
    evidence = sorted(
        (item for fact in records for item in name_evidence(fact)),
        key=lambda item: (item.at, item.message_id),
        reverse=True,
    )
    denied: set[str] = set()
    stated: list[NameEvidence] = []
    preferred: NameEvidence | None = None
    preference_seen = False
    fallback: NameEvidence | None = None
    for item in evidence:
        statement = item.statement
        if statement.kind is NameKind.DISCORD:
            fallback = item
            break
        if statement.kind is NameKind.AVOID:
            fallback = fallback or item
            denied.add(statement.name.casefold())
        elif statement.kind is NameKind.PREFERRED:
            if not preference_seen:
                preference_seen = True
                if statement.name.casefold() in denied:
                    continue
                # Same source/time with conflicting preferences is ambiguous.
                peers = {
                    peer.statement.name.casefold()
                    for peer in evidence
                    if peer.at == item.at
                    and peer.message_id == item.message_id
                    and peer.statement.kind is NameKind.PREFERRED
                }
                if len(peers) != 1:
                    return None
                preferred = item
        elif statement.name.casefold() not in denied:
            stated.append(item)
    return preferred or (stated[0] if stated else fallback)
