"""Lifecycle package: tiers, strength decay, forgetting."""

from discord_memory.lifecycle.strength import retention, should_forget
from discord_memory.lifecycle.tiers import assign_tier

__all__ = ["assign_tier", "retention", "should_forget"]
