"""Profile/guild digest summarization — derived representations (PLAN.md Part 7).

Background-only: runs under the pipeline lease after batches, never on read paths.
LLM output is validated against source facts via embedding-similarity sanity check;
failures keep the previous summary (never drift).
"""

from __future__ import annotations

import logging

from discord_memory.config import MemoryConfig
from discord_memory.models.facts import FactRecord, ProfileSummary
from discord_memory.ports.llm import ChatLLM, ChatRequest, Embedder, LlmMessage
from discord_memory.ports.store import MemoryStore

logger = logging.getLogger(__name__)

MAX_SUMMARY_FACTS = 40


def profile_summary_due(
    *,
    adds: int,
    threshold: int,
    fact_count: int,
    last_source_fact_count: int | None,
) -> bool:
    """Whether the profile digest should regenerate after this batch.

    Threshold is lifetime, not per-batch: first fire once ``fact_count`` reaches
    it, then again each time that many new facts land since the last digest.
    """
    if adds <= 0 or threshold <= 0 or fact_count < threshold:
        return False
    if last_source_fact_count is None:
        return True
    return fact_count - last_source_fact_count >= threshold


SUMMARY_PROMPT = """\
Synthesize the following durable facts about {subject} into one concise paragraph \
(<=120 words). Preserve only information present in the facts. Do not invent details. \
Facts:
{facts}
"""


class ConsolidationService:
    """Regenerates per-subject and guild digests."""

    def __init__(
        self,
        *,
        store: MemoryStore,
        llm: ChatLLM | None,
        embedder: Embedder | None,
        config: MemoryConfig,
    ) -> None:
        self._store = store
        self._llm = llm
        self._embedder = embedder
        self._config = config

    async def maybe_refresh_profile(
        self,
        *,
        guild_id: str,
        subject_id: str,
        adds: int,
    ) -> ProfileSummary | None:
        """Regenerate when lifetime fact count crosses the configured cadence."""
        threshold = self._config.extraction.auto_consolidate_after_adds
        if adds <= 0 or threshold <= 0:
            return None
        records = await self._store.top_strength_facts(
            guild_id,
            subject_ids=(subject_id,),
            limit=self._config.lifecycle.max_facts_per_user,
        )
        existing = await self._store.get_summary(guild_id, subject_id)
        last = existing.source_fact_count if existing is not None else None
        if not profile_summary_due(
            adds=adds,
            threshold=threshold,
            fact_count=len(records),
            last_source_fact_count=last,
        ):
            return existing
        return await self.regenerate_profile(guild_id=guild_id, subject_id=subject_id)

    async def regenerate_profile(
        self,
        *,
        guild_id: str,
        subject_id: str | None,
        subject_name: str = "",
    ) -> ProfileSummary | None:
        """Rebuild the digest from top-strength active facts."""
        records = await self._store.top_strength_facts(
            guild_id,
            subject_ids=(subject_id,) if subject_id else None,
            server_only=subject_id is None,
            limit=self._config.lifecycle.max_facts_per_user,
        )
        if len(records) < 2 or self._llm is None:
            return await self._store.get_summary(guild_id, subject_id)

        label = subject_name or (subject_id or "this server community")
        fact_block = "\n".join(f"- {record.text}" for record in records[:MAX_SUMMARY_FACTS])
        response = await self._llm.complete(
            ChatRequest(
                messages=(
                    LlmMessage(role="system", content="You are a precise memory summarizer."),
                    LlmMessage(
                        role="user",
                        content=SUMMARY_PROMPT.format(
                            subject=label,
                            facts=fact_block,
                        ),
                    ),
                ),
                max_tokens=1000,
                purpose="summarize",
                guild_id=guild_id,
            )
        )
        text = response.text.strip()
        if not text or not await self._sane(text, records[:MAX_SUMMARY_FACTS]):
            logger.warning("Summary sanity check failed; keeping previous summary")
            return await self._store.get_summary(guild_id, subject_id)

        summary = ProfileSummary(
            guild_id=guild_id,
            subject_id=subject_id,
            text=text,
            generated_at=None,
            source_fact_count=len(records),
        )
        await self._store.put_summary(summary)
        return summary

    async def _sane(self, text: str, records: tuple[FactRecord, ...]) -> bool:
        if self._embedder is None:
            return True
        threshold = self._config.extraction.summary_sanity_threshold
        (text_embedding,) = await self._embedder.embed((text,))
        centroid = [0.0] * len(text_embedding)
        for record in records[:20]:
            (embedding,) = await self._embedder.embed((record.text,))
            for index, value in enumerate(embedding):
                centroid[index] += value / min(len(records), 20)
        norm = sum(v * v for v in centroid) ** 0.5 or 1.0
        unit_centroid = [v / norm for v in centroid]
        dot = sum(a * b for a, b in zip(text_embedding, unit_centroid, strict=False))
        return dot >= threshold


__all__ = ["ConsolidationService", "profile_summary_due"]
