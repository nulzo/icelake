"""Lifecycle package: tiers, strength decay, forgetting, cap prune selection."""

from discord_memory.lifecycle.prune import select_prune_victims, select_prune_victims_by_anchor
from discord_memory.lifecycle.strength import retention, should_forget
from discord_memory.lifecycle.tiers import assign_tier

__all__ = [
    "assign_tier",
    "retention",
    "select_prune_victims",
    "select_prune_victims_by_anchor",
    "should_forget",
]
