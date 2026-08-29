"""Batch roster: minted participant tokens — the anti-hallucination protocol (§3.1)."""

from __future__ import annotations

import re
from collections.abc import Mapping

from discord_memory.models.common import FrozenModel

SERVER_TOKEN = "server"
_ROSTER_TOKEN = re.compile(r"\bp\d+\b")


class RosterParticipant(FrozenModel):
    token: str
    user_id: str
    display_name: str


def bind_roster_names(text: str, names_by_token: Mapping[str, str]) -> str:
    """Replace minted ``pN`` tokens in prose with display names.

    Tokens stay on ``subject_token`` / ``speaker_token``; stored fact text is
    human-readable. Unknown tokens are left unchanged. Possessives (``p0's``)
    bind correctly because ``'`` is a word boundary.
    """
    if not names_by_token:
        return text
    return _ROSTER_TOKEN.sub(
        lambda match: names_by_token.get(match.group(0), match.group(0)),
        text,
    )


class Roster:
    """Ordered participants with opaque reference tokens.

    Tokens are the only identity surface the extraction LLM sees. Verification maps
    tokens back to hardened IDs deterministically; unknown tokens are rejected.
    """

    def __init__(self) -> None:
        self._participants: list[RosterParticipant] = []
        self._by_token: dict[str, RosterParticipant] = {}
        self._by_user: dict[str, RosterParticipant] = {}

    def add(self, user_id: str, display_name: str) -> str:
        """Register a participant; idempotent per user. Returns their token."""
        if user_id in self._by_user:
            return self._by_user[user_id].token
        token = f"p{len(self._participants)}"
        participant = RosterParticipant(
            token=token,
            user_id=user_id,
            display_name=display_name or user_id,
        )
        self._participants.append(participant)
        self._by_token[token] = participant
        self._by_user[user_id] = participant
        return token

    @property
    def participants(self) -> tuple[RosterParticipant, ...]:
        return tuple(self._participants)

    def user_id_for(self, token: str) -> str | None:
        participant = self._by_token.get(token.strip())
        return participant.user_id if participant else None

    def token_for(self, user_id: str) -> str | None:
        participant = self._by_user.get(user_id)
        return participant.token if participant else None

    def name_for(self, token: str) -> str | None:
        participant = self._by_token.get(token.strip())
        return participant.display_name if participant else None

    def display_name(self, user_id: str) -> str | None:
        participant = self._by_user.get(user_id)
        return participant.display_name if participant else None

    def bind_names(self, text: str) -> str:
        """Rewrite ``pN`` tokens in ``text`` to this roster's display names."""
        return bind_roster_names(
            text,
            {participant.token: participant.display_name for participant in self._participants},
        )

    def knows(self, token: str) -> bool:
        return token.strip() in {SERVER_TOKEN, *self._by_token}

    def render(self) -> str:
        lines = [
            f"{participant.token} = {participant.display_name} (verified participant)"
            for participant in self._participants
        ]
        lines.append(f"{SERVER_TOKEN} = the server community as a whole")
        return "\n".join(lines)
