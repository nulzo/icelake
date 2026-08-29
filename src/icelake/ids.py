"""Opaque, sortable, prefix-namespaced identifiers.

Facts use ``fct_<ulid>``, jobs ``job_<ulid>``, batches ``bat_<ulid>``. ULID-style ids
are time-ordered so store keys cluster chronologically (hot inserts append to indexes)
and log lines sort naturally.
"""

from __future__ import annotations

import os
import time

_ENCODE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid() -> str:
    """Generate a 26-character uppercase Crockford base32 ULID."""
    ts_ms = time.time_ns() // 1_000_000
    entropy = int.from_bytes(os.urandom(10), "big")
    value = ((ts_ms & 0xFFFFFFFFFFFF) << 80) | (entropy & 0xFFFFFFFFFF)
    out = [_ENCODE32[(value >> (5 * i)) & 0x1F] for i in range(26)]
    return "".join(reversed(out))


def prefixed(prefix: str) -> str:
    """Return ``<prefix>_<ulid>`` (e.g. ``fct_01J...``)."""
    return f"{prefix}_{ulid().lower()}"
