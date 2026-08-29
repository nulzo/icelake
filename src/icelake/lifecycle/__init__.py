"""Lifecycle package: tiers, strength decay, forgetting, cap prune selection."""

from icelake.lifecycle.prune import select_prune_victims, select_prune_victims_by_anchor
from icelake.lifecycle.strength import retention, should_forget
from icelake.lifecycle.tiers import assign_tier

__all__ = [
    "assign_tier",
    "retention",
    "select_prune_victims",
    "select_prune_victims_by_anchor",
    "should_forget",
]
