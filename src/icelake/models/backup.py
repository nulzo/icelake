"""Versioned operational-backup manifest (separate from compliance exports)."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from icelake.models.common import FrozenModel


class BackupBackend(StrEnum):
    SQLITE = "sqlite"
    MONGO = "mongo"


class BackupManifest(FrozenModel):
    format_version: Literal[1] = 1
    backend: BackupBackend
    created_at: datetime
    schema_version: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
