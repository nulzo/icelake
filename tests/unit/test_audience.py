"""Permission failures never disclose stored evidence to the reply or reranker."""

from __future__ import annotations

import pytest

from icelake import Audience, DiscordMemory, MemoryConfig, RecallQuery, Scope
from icelake.models.facts import FactRecord, SourceRef
from icelake.permissions import AudienceGate
from tests.conftest import make_config


def fact(identifier, *channels):
    return FactRecord(
        id=identifier,
        guild_id="g",
        subject_id="u",
        text="likes coffee",
        citations=tuple(
            SourceRef(message_id=f"m-{c}", guild_id="g", channel_id=c, author_id="u")
            for c in channels
        ),
    )


async def test_same_channel_requires_all_sources_and_known_provenance():
    records = (
        fact("visible", "public"),
        fact("hidden", "private"),
        fact("mixed", "public", "private"),
        fact("unknown"),
    )
    allowed = await AudienceGate().filter(
        "g", Audience(channel_id="public", requester_id="u"), records
    )
    assert [f.id for f in allowed] == ["visible"]


@pytest.mark.parametrize("mode", ["error", "empty", "wrong_channel"])
async def test_live_authorizer_denies_unknowns(mode):
    class Policy:
        async def allowed_channels(self, guild_id, audience, source_channels):
            if mode == "error":
                raise OSError("Discord unavailable")
            return frozenset({"unknown"}) if mode == "wrong_channel" else frozenset()

    assert not await AudienceGate(Policy()).filter(
        "g", Audience(channel_id="public", requester_id="u"), (fact("f", "private"),)
    )


async def test_secure_default_requires_audience():
    async with DiscordMemory(
        MemoryConfig(storage="sqlite://:memory:", workers={"enabled": False})
    ) as m:
        await m.facts.remember(guild_id="g", subject_id="u", text="likes coffee")
        result = await m.recall(RecallQuery(guild_id="g", scope=Scope.GUILD))
        assert not result.facts and result.warnings[0].value == "audience_required"
        assert not (await m.prompt_context(guild_id="g", asker_id="u")).injection_block


async def test_private_content_never_sent_to_reranker_and_revocation_rechecked():
    class Policy:
        calls = 0

        async def allowed_channels(self, guild_id, audience, source_channels):
            self.calls += 1
            return frozenset({"public"}) if self.calls == 1 else frozenset()

    class Ranker:
        def __init__(self):
            self.documents = []

        async def score(self, query, documents):
            self.documents.extend(documents)
            return [1.0] * len(documents)

    ranker = Ranker()
    async with DiscordMemory(make_config(), reranker=ranker, channel_authorizer=Policy()) as m:
        await m._store.insert_fact(fact("f", "public"))
        private = fact("secret", "private").model_copy(update={"text": "classified private secret"})
        await m._store.insert_fact(private)
        result = await m.recall(
            RecallQuery(
                guild_id="g",
                scope=Scope.GUILD,
                text="coffee",
                audience=Audience(channel_id="public", requester_id="u"),
            )
        )
        assert all("secret" not in doc for doc in ranker.documents)
        assert not result.facts


async def test_citation_rechecks_purge_and_permission():
    async with DiscordMemory(make_config()) as m:
        record = await m.facts.remember(
            guild_id="g",
            subject_id="123456789012345678",
            text="likes coffee",
            source_refs=(
                SourceRef(message_id="111", guild_id="g", channel_id="public", author_id="u"),
            ),
        )
        context = await m.prompt_context(
            guild_id="g",
            asker_id=record.subject_id,
            audience=Audience(channel_id="public", requester_id="u"),
        )
        await m.admin.purge_user("g", "u", dry_run=False)
        reply = await m.cite("likes coffee", context)
        assert not reply.sources


@pytest.mark.parametrize("mode", ["denied", "error", "timeout"])
async def test_access_failure_has_diagnostics_and_never_looks_like_empty_storage(mode):
    import asyncio

    from icelake.models.retrieval import RecallWarning
    from icelake.permissions import AccessStatus, MemoryAccessError

    class Policy:
        async def allowed_channels(self, guild_id, audience, source_channels):
            if mode == "error":
                raise OSError("offline")
            if mode == "timeout":
                await asyncio.sleep(1)
            return frozenset()

    config = make_config(access={"require_audience": True, "timeout_seconds": 0.01})
    audience = Audience(channel_id="public", requester_id="u")
    async with DiscordMemory(config, channel_authorizer=Policy()) as memory:
        record = fact("f", "public").model_copy(update={"subject_id": "123456789012345678"})
        await memory.identity.register_alias("g", "123456789012345678", "Alice")
        await memory._store.insert_fact(record)
        expected = AccessStatus.RESTRICTED if mode == "denied" else AccessStatus.UNAVAILABLE
        with pytest.raises(MemoryAccessError) as error:
            await memory.visible_facts("g", ("f",), audience=audience, strict=True)
        assert error.value.status is expected
        assert not await memory.visible_facts("g", ("f",), audience=audience)
        with pytest.raises(MemoryAccessError):
            await memory.visible_sources("g", record.citations, audience=audience, strict=True)
        result = await memory.recall(
            RecallQuery(guild_id="g", scope=Scope.GUILD, audience=audience)
        )
        warning = (
            RecallWarning.ACCESS_RESTRICTED
            if mode == "denied"
            else RecallWarning.ACCESS_UNAVAILABLE
        )
        assert result.warnings == (warning,)
        context = await memory.prompt_context(
            guild_id="g", asker_id="123456789012345678", audience=audience
        )
        assert warning in context.warnings
        assert not context.facts
        assert not result.facts
        assert not await memory.visible_facts("g", ("absent",), audience=audience, strict=True)


async def test_strict_missing_audience_and_unscoped_evidence_are_restricted():
    from icelake.permissions import AccessStatus, MemoryAccessError

    async with DiscordMemory(make_config(access={"require_audience": True})) as memory:
        await memory._store.insert_fact(fact("f"))
        for audience in (None, Audience(channel_id="public", requester_id="u")):
            with pytest.raises(MemoryAccessError) as error:
                await memory.visible_facts("g", ("f",), audience=audience, strict=True)
            assert error.value.status is AccessStatus.RESTRICTED
