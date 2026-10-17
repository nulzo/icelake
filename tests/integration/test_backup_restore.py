"""A backup counts only after a restore drill through real store APIs."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from icelake import DiscordMemory
from icelake.adapters.sqlite.backup import restore_backup
from icelake.adapters.sqlite.store import SqliteStore
from icelake.models.recovery import ProjectionJob
from icelake.ports.queue import BatchKey, MessageStatus, StoredMessage
from tests.conftest import make_config


async def test_sqlite_operational_restore_preserves_full_state_and_releases_claims(tmp_path):
    source = SqliteStore(f"sqlite:///{tmp_path / 'source.sqlite'}")
    memory = DiscordMemory(make_config(), store=source)
    async with memory:
        fact = await memory.facts.remember(guild_id="g", subject_id="u", text="likes coffee")
        await memory.identity.register_alias("g", "u", "Alice")
        await memory.admin.set_opt_out("g", "blocked", True)
        await source.bump_governance_generation("g", "erased")
        await source.set_cursor("g", "ingest:user:m1", "committed")
        await source.charge_guild_tokens(
            "g", day_key="2026-10-17", month_key="2026-10", prompt_tokens=42
        )
        await source.enqueue_projection(
            ProjectionJob(guild_id="g", fact_id=fact.id, version=fact.version)
        )
        now = datetime.now(UTC)
        await source.queue.put_message(
            StoredMessage(
                message_id="q1",
                guild_id="g",
                channel_id="c",
                author_id="u",
                subject_key="u",
                content="queued",
                created_at=now,
            )
        )
        await source.queue.claim_batch(
            BatchKey(guild_id="g", subject_key="u"),
            now=now,
            owner="crashed",
            lease_seconds=600,
            limit=10,
        )
        before = await source.nodes_for_fact("g", fact.id)
        archive = tmp_path / "backup"
        manifest = await memory.admin.backup(archive)
        assert manifest.size_bytes > 0
    destination = tmp_path / "restored.sqlite"
    await restore_backup(archive, destination)
    target = SqliteStore(f"sqlite:///{destination}")
    await target.setup()
    try:
        assert (await target.get_fact("g", fact.id)).text == fact.text
        assert await target.nodes_for_fact("g", fact.id) == before
        assert await target.get_opt_out("g", "blocked")
        assert (await target.consent_state("g", "erased")).generation == 1
        assert await target.get_cursor("g", "ingest:user:m1") == "committed"
        assert await target.aliases_for_user("g", "u")
        assert await target.vectors.count("g") == 1
        assert len(await target.pending_projections(10)) == 1
        queued = await target.queue.recent_messages("g", 10)
        assert queued[0].status is MessageStatus.PENDING
        claim = await target.queue.claim_batch(
            BatchKey(guild_id="g", subject_key="u"),
            now=now,
            owner="new",
            lease_seconds=60,
            limit=10,
        )
        assert len(claim.messages) == 1
    finally:
        await target.close()


@pytest.mark.parametrize(
    "damage", ["checksum", "missing_manifest", "version", "oversized", "occupied"]
)
async def test_restore_rejects_corruption_and_never_overwrites(tmp_path, damage):
    store = SqliteStore("sqlite://:memory:")
    await store.setup()
    archive = tmp_path / "backup"
    await store.backup(archive)
    await store.close()
    destination = tmp_path / "target.sqlite"
    kwargs = {}
    if damage == "checksum":
        snapshot = archive / "memory.sqlite"
        data = bytearray(snapshot.read_bytes())
        data[-1] ^= 0xFF
        snapshot.write_bytes(data)
    elif damage == "missing_manifest":
        (archive / "manifest.json").unlink()
    elif damage == "version":
        data = json.loads((archive / "manifest.json").read_text())
        data["schema_version"] = 999
        (archive / "manifest.json").write_text(json.dumps(data))
    elif damage == "oversized":
        kwargs["max_bytes"] = 1
    else:
        destination.write_bytes(b"existing database")
    with pytest.raises((ValueError, FileNotFoundError, FileExistsError)):
        await restore_backup(archive, destination, **kwargs)
    if damage == "occupied":
        assert destination.read_bytes() == b"existing database"
    else:
        assert not destination.exists()
