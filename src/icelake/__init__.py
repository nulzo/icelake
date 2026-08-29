"""icelake: accurate, scalable agentic memory for Discord bots.

Quickstart::

    from icelake import Memory, MemoryConfig, MessageEvent

    memory = Memory(MemoryConfig(
        storage="sqlite:///memory.db",
        llm="openai://$KEY@openrouter.ai/api/v1?model=google/gemini-2.5-flash",
    ))
    async with memory:
        await memory.observe(event)
        ctx = await memory.prompt_context(guild_id=g, asker_id=u, text=msg)
        system_prompt += ctx.injection_block

"""

from importlib.metadata import version

from icelake.api.classify import CommandAction, UserMemoryCommand
from icelake.api.client import DiscordMemory
from icelake.config import (
    BatchingConfig,
    BudgetsConfig,
    EmbeddingsConfig,
    ExtractionConfig,
    LifecycleConfig,
    LlmConfig,
    MemoryConfig,
    ObserveConfig,
    PrivacyConfig,
    RetrievalConfig,
    StorageConfig,
    WorkersConfig,
)
from icelake.errors import (
    BudgetExceededError,
    ConfigError,
    DiscordMemoryError,
    FactNotFoundError,
    LlmCapabilityError,
    SchemaValidationError,
    StorageUnavailableError,
    StructuredOutputError,
    SubjectNotAllowedError,
    WorkerNotRunningError,
)
from icelake.models import (
    CHANNELS_ALL,
    CHANNELS_DEFAULT,
    CHANNELS_DISCOVERY,
    Attribution,
    AttributionType,
    ChannelName,
    Citation,
    EdgeKind,
    EntityRecord,
    FactCategory,
    FactHistoryEntry,
    FactRecord,
    GuildStats,
    HealthReport,
    IgnoreReason,
    MemoryExport,
    MemoryTier,
    MessageEvent,
    NeighborInfo,
    NodeType,
    ObserveReceipt,
    ObserveStatus,
    Polarity,
    PromptContext,
    PurgeReport,
    RecallQuery,
    RecallResult,
    RecallWarning,
    RejectReason,
    RelationEdge,
    Resolution,
    ResolvedCandidate,
    Scope,
    ScoreComponents,
    ScoredFact,
    SourceRef,
    StanceSummary,
    channels,
)
from icelake.models.events import (
    BatchCompleted,
    ComponentDegraded,
    ExtractionFailed,
    FactCommitted,
    FactSupersededEvent,
)

__version__ = version("icelake")

# Public, transport-free name. ``DiscordMemory`` stays as the back-compat
# alias; the class itself is store/LLM-agnostic and Discord is an extra.
Memory = DiscordMemory

__all__ = [
    "CHANNELS_ALL",
    "CHANNELS_DEFAULT",
    "CHANNELS_DISCOVERY",
    "Attribution",
    "AttributionType",
    "BatchCompleted",
    "BatchingConfig",
    "BudgetExceededError",
    "BudgetsConfig",
    "ChannelName",
    "Citation",
    "CommandAction",
    "ComponentDegraded",
    "ConfigError",
    "DiscordMemory",
    "DiscordMemoryError",
    "EdgeKind",
    "EmbeddingsConfig",
    "EntityRecord",
    "ExtractionConfig",
    "ExtractionFailed",
    "FactCategory",
    "FactCommitted",
    "FactHistoryEntry",
    "FactNotFoundError",
    "FactRecord",
    "FactSupersededEvent",
    "GuildStats",
    "HealthReport",
    "IgnoreReason",
    "LifecycleConfig",
    "LlmCapabilityError",
    "LlmConfig",
    "Memory",
    "MemoryConfig",
    "MemoryExport",
    "MemoryTier",
    "MessageEvent",
    "NeighborInfo",
    "NodeType",
    "ObserveConfig",
    "ObserveReceipt",
    "ObserveStatus",
    "Polarity",
    "PrivacyConfig",
    "PromptContext",
    "PurgeReport",
    "RecallQuery",
    "RecallResult",
    "RecallWarning",
    "RejectReason",
    "RelationEdge",
    "Resolution",
    "ResolvedCandidate",
    "RetrievalConfig",
    "SchemaValidationError",
    "Scope",
    "ScoreComponents",
    "ScoredFact",
    "SourceRef",
    "StanceSummary",
    "StorageConfig",
    "StorageUnavailableError",
    "StructuredOutputError",
    "SubjectNotAllowedError",
    "UserMemoryCommand",
    "WorkerNotRunningError",
    "WorkersConfig",
    "channels",
]
