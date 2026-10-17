"""Optional operational snapshot capability; portable content exports are separate."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from icelake.models.backup import BackupManifest


@runtime_checkable
class BackupProvider(Protocol):
    async def backup(self, directory: Path) -> BackupManifest:
        """Write a complete snapshot of the provider's documented resource scope."""
        ...
