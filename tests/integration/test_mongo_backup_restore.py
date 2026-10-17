"""Optional native-tool restore drill; target must be a disposable empty deployment."""

from __future__ import annotations

import os
import shutil

import pytest

from icelake.adapters.mongo import MongoStore
from icelake.adapters.mongo.backup import MongoReplicaBackup
from icelake.models.facts import FactRecord
from icelake.models.recovery import ProjectionJob


async def test_native_mongo_restore_drill(mongo_store, tmp_path):
    target_uri = os.environ.get("ICELAKE_RESTORE_MONGODB_URI")
    if not target_uri or not shutil.which("mongodump") or not shutil.which("mongorestore"):
        pytest.skip(
            "requires native tools and ICELAKE_RESTORE_MONGODB_URI pointing to an empty deployment"
        )
    source = mongo_store
    target = MongoStore(target_uri, database=source.db.name)
    try:
        async with source.transaction():
            await source.insert_fact(
                FactRecord(id="f", guild_id="g", text="durable restore sentinel")
            )
            await source.enqueue_projection(ProjectionJob(guild_id="g", fact_id="f", version=1))
            await source.set_cursor("g", "ingest:receipt", "committed")
        await source.set_opt_out("g", "blocked", True)
        await source.bump_governance_generation("g", "erased")
        adapter = MongoReplicaBackup(timeout_seconds=60, max_bytes=50_000_000)
        snapshot = tmp_path / "snapshot"
        manifest = await adapter.create(os.environ["MONGODB_URI"], snapshot)
        await adapter.restore(snapshot, target_uri)
        await target.setup()
        assert (await target.get_fact("g", "f")).text == "durable restore sentinel"
        assert await target.get_opt_out("g", "blocked")
        assert (await target.consent_state("g", "erased")).generation == 1
        assert await target.get_cursor("g", "ingest:receipt") == "committed"
        assert len(await target.pending_projections(10)) == 1
        assert manifest.size_bytes > 0
        with pytest.raises(ValueError, match="empty"):
            await adapter.restore(snapshot, target_uri)
        (snapshot / "deployment.archive.gz").write_bytes(b"damaged")
        with pytest.raises(ValueError, match="integrity"):
            await adapter.restore(snapshot, target_uri)
    finally:
        # Only our generated fixture database is removed. A production target
        # would be rejected by the adapter's empty-deployment check above.
        if source.db.name.startswith("icelake_test_"):
            await target.db.client.drop_database(source.db.name)
        await target.close()
