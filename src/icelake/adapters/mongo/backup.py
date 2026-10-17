"""Native full-replica-set backups using MongoDB Database Tools.

This is intentionally deployment-level, not a guild export. --oplog requires a
full dump. Restore only to a fresh, offline deployment; never merge into a live
server. Managed Atlas backups remain the preferred service at larger scale.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pymongo import AsyncMongoClient

from icelake.models.backup import BackupBackend, BackupManifest


class MongoReplicaBackup:
    """Bounded native-tool adapter; executable paths and deadlines are configurable."""

    def __init__(
        self,
        *,
        dump_executable: str = "mongodump",
        restore_executable: str = "mongorestore",
        timeout_seconds: float = 300,
        max_bytes: int = 10 * 1024**3,
    ) -> None:
        if timeout_seconds <= 0 or max_bytes <= 0:
            raise ValueError("backup limits must be positive")
        self._dump = dump_executable
        self._restore = restore_executable
        self._timeout_seconds = timeout_seconds
        self._max_bytes = max_bytes

    async def create(self, uri: str, directory: Path) -> BackupManifest:
        """Snapshot ALL databases in a replica set, including concurrent writes.

        Supply a deployment URI without a database path. Requires backup privileges.
        Do not run schema migrations during the dump. A failed dump has no manifest.
        """
        async with AsyncMongoClient[dict[str, Any]](uri, serverSelectionTimeoutMS=5000) as client:
            hello = await client.admin.command("hello")
            if not hello.get("setName"):
                raise ValueError("consistent online Mongo backup requires a replica set")
        await asyncio.to_thread(directory.mkdir, mode=0o700, exist_ok=False)
        archive = directory / "deployment.archive.gz"
        await self._run(self._dump, uri, archive, "--oplog", "--gzip", f"--archive={archive}")
        manifest = await asyncio.to_thread(self._manifest, archive)
        await asyncio.to_thread(_publish_manifest, directory, manifest)
        return manifest

    async def restore(self, directory: Path, target_uri: str) -> BackupManifest:
        """Verify then restore into an EMPTY, offline deployment; stop on any error.

        Partial restores are intentionally not exposed as complete. On failure,
        discard the destination and retry into another fresh deployment. Reconcile
        erasures newer than the backup before connecting any application clients.
        """
        manifest = BackupManifest.model_validate_json(
            await asyncio.to_thread((directory / "manifest.json").read_bytes)
        )
        archive = directory / "deployment.archive.gz"
        if manifest.backend is not BackupBackend.MONGO or manifest.schema_version != 1:
            raise ValueError("unsupported Mongo backup format")
        actual = await asyncio.to_thread(self._manifest, archive)
        if actual.sha256 != manifest.sha256 or actual.size_bytes != manifest.size_bytes:
            raise ValueError("backup integrity mismatch")
        async with AsyncMongoClient[dict[str, Any]](
            target_uri, serverSelectionTimeoutMS=5000
        ) as client:
            databases = await client.list_database_names()
            if set(databases) - {"local", "config", "admin"}:
                raise ValueError("restore requires an empty, offline Mongo deployment")
            if set(await client.admin.list_collection_names()) - {"system.version", "system.keys"}:
                raise ValueError("restore refuses an existing administrative database")
        await self._run(
            self._restore,
            target_uri,
            archive,
            "--oplogReplay",
            "--gzip",
            "--stopOnError",
            f"--archive={archive}",
        )
        async with AsyncMongoClient[dict[str, Any]](
            target_uri, serverSelectionTimeoutMS=5000
        ) as client:
            for database in await client.list_database_names():
                if database in {"admin", "config", "local"}:
                    continue
                db = client[database]
                if "dm_messages" in await db.list_collection_names():
                    await db["dm_messages"].update_many(
                        {"status": "claimed"},
                        {
                            "$set": {"status": "pending"},
                            "$unset": {"lease_owner": "", "lease_until": ""},
                        },
                    )
                    await db["dm_batch_leases"].delete_many({})
        return manifest

    def _manifest(self, path: Path) -> BackupManifest:
        size = path.stat().st_size
        if size > self._max_bytes:
            raise ValueError("backup exceeds configured size limit")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return BackupManifest(
            backend=BackupBackend.MONGO,
            created_at=datetime.now(UTC),
            schema_version=1,
            sha256=digest,
            size_bytes=size,
        )

    async def _run(self, executable: str, uri: str, archive: Path, *arguments: str) -> None:
        # JSON is a YAML subset accepted by Database Tools. Credentials stay out
        # of argv, logs and exceptions; the temporary config is mode 0600.
        with tempfile.TemporaryDirectory(prefix="icelake-mongo-") as temporary:
            config = Path(temporary) / "credentials.yml"
            await asyncio.to_thread(_credentials, config, uri)
            process = await asyncio.create_subprocess_exec(
                executable,
                f"--config={config}",
                "--quiet",
                *arguments,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            finished = asyncio.create_task(process.wait())
            try:
                async with asyncio.timeout(self._timeout_seconds):
                    while process.returncode is None:
                        try:
                            await asyncio.wait_for(asyncio.shield(finished), timeout=0.25)
                        except TimeoutError:
                            size = await asyncio.to_thread(_archive_size, archive)
                            if size > self._max_bytes:
                                raise ValueError("backup exceeds configured size limit") from None
                    if process.returncode != 0:
                        raise RuntimeError(f"Mongo backup tool failed (exit {process.returncode})")
            finally:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                await finished


def _credentials(path: Path, uri: str) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump({"uri": uri}, stream)


def _publish_manifest(directory: Path, manifest: BackupManifest) -> None:
    archive = directory / "deployment.archive.gz"
    archive.chmod(0o600)
    with archive.open("rb") as stream:
        os.fsync(stream.fileno())
    with (directory / "manifest.json").open("x") as stream:
        stream.write(manifest.model_dump_json(indent=2))
        stream.flush()
        os.fsync(stream.fileno())


def _archive_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0
