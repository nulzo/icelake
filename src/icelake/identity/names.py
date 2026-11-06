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

from icelake.identity.aliases import extract_self_name_aliases
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

    @property
    def aspect(self) -> NameKind:
        """Separate self-introductions from addressing preferences and withdrawals."""
        return NameKind.STATED if self.kind is NameKind.STATED else NameKind.PREFERRED


@dataclass(frozen=True)
class NameEvidence:
    statement: NameStatement
    fact: FactRecord
    at: datetime
    message_id: str


_PREFIX = r"(?:(?:hey|hi|btw|actually)[,!]?\s+)?(?:please\s+)?"
_PATTERNS = (
    (
        NameKind.AVOID,
        r"(?:do not call me|don't call me|don\u2019t call me|stop calling me|"
        r"i am not|i'm not|i\u2019m not|my name (?:is not|isn't|isn\u2019t))",
    ),
    (NameKind.PREFERRED, r"(?:call me|i go by|i prefer to be called)"),
    (NameKind.STATED, r"my (?:(?:real|first|full) )?name(?: is|'s|\u2019s)"),
)
_TEMPORARY = re.compile(
    r"\b(?:today|tonight|tomorrow|this week|this weekend|this month|for now|until)\b",
    re.IGNORECASE,
)
_FALLBACK = re.compile(
    _PREFIX + r"(?:use|call me by) my (?:discord |display |user)name[.!]?", re.IGNORECASE
)


def name_statements(text: str) -> tuple[NameStatement, ...]:
    """Recognize direct English self-introductions, never arbitrary name mentions."""
    text = re.sub(r"^(?:<@!?\d+>\s*)+", "", text.strip())
    text = re.sub(r"^(?:hey|hi|btw|actually)[,!]?\s+", "", text, flags=re.IGNORECASE)
    if _TEMPORARY.search(text) or any(
        char in text for char in ("\n", "\r", '"', "“", "”", "`", ">", "?")
    ):
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
    """Require an active fact's matching original statement by its own subject.

    Extracted wording and category may vary. The direct self-statement supplies
    the naming intent; matching its name in the fact binds that evidence to it.
    """
    if (
        not fact.subject_id
        or fact.attribution.type not in {AttributionType.SELF, AttributionType.MANUAL}
        or fact.attribution.speaker_id not in {None, fact.subject_id}
    ):
        return ()
    evidence: list[NameEvidence] = []
    # Manual self-corrections are already authenticated by the host's command
    # boundary. They need no invented Discord message or citation to be useful.
    if (
        fact.attribution.type is AttributionType.MANUAL
        and fact.attribution.actor_id == fact.subject_id
        and not fact.citations
        and (at := fact.observed_at or fact.created_at)
    ):
        evidence.extend(
            NameEvidence(statement, fact, at.replace(tzinfo=at.tzinfo or UTC), fact.id)
            for statement in name_statements(fact.text)
        )
    for source in fact.citations:
        if source.author_id != fact.subject_id or source.guild_id != fact.guild_id:
            continue
        statements = tuple(
            statement
            for statement in name_statements(source.content_snippet)
            if not statement.name or statement.name.casefold() in fact.text.casefold()
        ) or _normalized_evidence(fact.text, source.content_snippet)
        for statement in statements:
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


def _normalized_evidence(text: str, source: str) -> tuple[NameStatement, ...]:
    """Use the extractor's normalized name while checking the original self-report.

    Ordinary prose around an introduction is allowed; the model cannot promote a
    roster label, third-party name, question or quoted story to a self-introduction.
    This reuses the existing fact extractor, without another model call or store.
    """
    if _TEMPORARY.search(source) or re.search(
        r'[\n\r"“”`>?]|\b(?:if|pretend|imagine|joking|joke|he said|she said|they said)\b',
        source,
        re.IGNORECASE,
    ):
        return ()
    # Reuse the existing self-name span parser when the extractor combines the
    # introduction with another fact. Selection must not depend on prose layout.
    for name, _ in extract_self_name_aliases(source):
        if name.casefold() not in text.casefold():
            continue
        for kind, marker in _PATTERNS:
            if kind is NameKind.AVOID:
                continue
            if re.search(
                r"(?:^|[.!;,]\s*|\b(?:actually|btw|yeah|hey)\s+)"
                + marker
                + r"\s+"
                + re.escape(name)
                + r"(?=$|[\s.!;,])",
                source.strip(),
                re.IGNORECASE,
            ):
                return (NameStatement(kind, name),)
    forms = (
        (r".+? (?:prefers to be called|goes by)", "call me"),
        (r".+?(?:'s|\u2019s) (?:(?:real|first|full) )?name is", "my name is"),
        (r".+? (?:is not|does not want to be called|must not be called)", "do not call me"),
    )
    for pattern, replacement in forms:
        normalized = re.sub(r"^" + pattern + r"\s+", replacement + " ", text, flags=re.I)
        for statement in name_statements(normalized):
            if statement.kind is NameKind.DISCORD:
                continue
            marker = next(marker for kind, marker in _PATTERNS if kind is statement.kind)
            if re.search(
                r"(?:^|[.!;,]\s*|\b(?:actually|btw|yeah|hey)\s+)"
                + marker
                + r"\s+"
                + re.escape(statement.name)
                + r"(?=$|[\s.!;,])",
                source.strip(),
                re.IGNORECASE,
            ):
                return (statement,)
    return ()


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
