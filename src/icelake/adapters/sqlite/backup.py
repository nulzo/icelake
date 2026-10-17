"""SQLite online backups and verified restore into a new, offline database.

The native backup API includes all tables: governance, raw queue, receipts,
indexes, aliases, graph incidence, summaries, budgets and history. Independent
queue/vector overrides require their own backup adapter and are not included.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from icelake.adapters.sqlite.connection import SCHEMA_VERSION, SqliteConnection
from icelake.models.backup import BackupBackend, BackupManifest


async def create_backup(connection: SqliteConnection, directory: Path) -> BackupManifest:
    """Create a new directory; the manifest appears only after a complete snapshot.

    Never overwrite an existing backup. A crash can leave an incomplete directory;
    restoration rejects it. Store these sensitive files with restricted access.
    """
    await asyncio.to_thread(directory.mkdir, mode=0o700, parents=False, exist_ok=False)
    snapshot = directory / "memory.sqlite"
    await connection.backup_to(snapshot)
    manifest = await asyncio.to_thread(_manifest, snapshot)
    await asyncio.to_thread(_write_manifest, directory, manifest)
    return manifest


def _manifest(snapshot: Path) -> BackupManifest:
    with snapshot.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return BackupManifest(
        backend=BackupBackend.SQLITE,
        created_at=datetime.now(UTC),
        schema_version=SCHEMA_VERSION,
        sha256=digest,
        size_bytes=snapshot.stat().st_size,
    )


def _write_manifest(directory: Path, manifest: BackupManifest) -> None:
    with (directory / "manifest.json").open("x") as stream:
        stream.write(manifest.model_dump_json(indent=2))
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(directory)


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


async def restore_backup(
    directory: Path,
    destination: Path,
    *,
    max_bytes: int = 10 * 1024**3,
) -> BackupManifest:
    """Verify and atomically publish a new database; existing targets are refused.

    Run before opening the destination. Reconcile erasures since the snapshot
    before exposing restored data. Checksums detect corruption, not authenticity.
    """
    return await asyncio.to_thread(_restore, directory, destination, max_bytes)


def _restore(directory: Path, destination: Path, max_bytes: int) -> BackupManifest:
    manifest = BackupManifest.model_validate_json((directory / "manifest.json").read_bytes())
    if manifest.backend is not BackupBackend.SQLITE or manifest.schema_version != SCHEMA_VERSION:
        raise ValueError("unsupported backup backend or schema version")
    source = directory / "memory.sqlite"
    if manifest.size_bytes > max_bytes or source.stat().st_size != manifest.size_bytes:
        raise ValueError("backup size mismatch or configured limit exceeded")
    if destination.exists() or any(
        Path(str(destination) + suffix).exists() for suffix in ("-wal", "-shm")
    ):
        raise FileExistsError("restore requires a new, offline destination")
    descriptor, temporary = tempfile.mkstemp(prefix=".memory-restore-", dir=destination.parent)
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        shutil.copyfile(source, temporary_path)
        if _manifest(temporary_path).sha256 != manifest.sha256:
            raise ValueError("backup checksum mismatch")
        _validate_and_reset(temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        # Hard-link publication is atomic AND refuses a target created concurrently.
        os.link(temporary_path, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
    return manifest


def _validate_and_reset(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("backup failed SQLite integrity check")
        version = connection.execute(
            "SELECT value FROM dm_meta WHERE key='schema_version'"
        ).fetchone()
        if version != (str(SCHEMA_VERSION),):
            raise ValueError("snapshot schema does not match manifest")
        with connection:
            connection.execute(
                "UPDATE dm_messages SET status='pending', lease_owner=NULL, lease_until=NULL "
                "WHERE status='claimed'"
            )
            connection.execute("DELETE FROM dm_batch_leases")
