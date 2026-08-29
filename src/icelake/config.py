"""MemoryConfig: validated, frozen, nested settings (API.md Part 4).

Providers are configured by URL strings for frictionless setup::

    storage   = "sqlite:///data/memory.db"
    llm       = "openai://$OPENROUTER_API_KEY@openrouter.ai/api/v1?model=google/gemini-2.5-flash"
    embeddings= "hashing"            # free deterministic default
                # "openai://$KEY@api.openai.com/v1?model=text-embedding-3-small"
                # "local"             # sentence-transformers if installed

Every nested group can also be passed as a typed object for programmatic composition.
Unknown keys raise ConfigError immediately (typo protection).
"""

from __future__ import annotations

import os
from enum import StrEnum
from typing import Literal
from urllib.parse import parse_qs, unquote, urlsplit

from pydantic import Field, field_validator, model_validator

from icelake.errors import ConfigError
from icelake.models.common import FrozenModel


def _expand(value: str) -> str:
    if value.startswith("$"):
        env_value = os.environ.get(value[1:], "")
        if not env_value:
            raise ConfigError(f"environment variable {value} is not set")
        return env_value
    return value


class StorageConfig(FrozenModel):
    """Persistence backend selection."""

    url: str = "sqlite:///icelake.db"

    @model_validator(mode="after")
    def _validate_scheme(self) -> StorageConfig:
        _ = self.backend  # raises ConfigError for unsupported schemes
        return self

    @property
    def backend(self) -> str:
        scheme = urlsplit(self.url).scheme
        if scheme in {"sqlite", "memory", ""}:
            return "sqlite"
        if scheme in {"mongodb", "mongodb+srv"}:
            return "mongo"
        if scheme in {"postgres", "postgresql"}:
            return "postgres"
        raise ConfigError(f"unsupported storage url scheme {scheme!r}")


class LlmConfig(FrozenModel):
    """OpenAI-chat-completions-compatible provider settings.

    ``small_model`` routes cheap structured tasks (reconcile, classify, profile
    summaries) to a smaller tier; ``None`` uses ``model`` for everything.
    ``reasoning_effort`` forwards OpenRouter's ``reasoning.effort`` — extraction,
    reconcile, and classify are structured tasks that gain little from long
    reasoning, so ``low`` cuts latency and completion-token cost sharply on
    reasoning models. ``None`` sends nothing (provider default).

    Capability declarations — set these to match what the model's endpoints
    actually accept; mismatches fail loudly (``LlmCapabilityError``) instead of
    silently degrading:

    - ``temperature``: default ``0.0`` (deterministic extraction). Set to
      ``None`` to omit the parameter entirely — reasoning-model endpoints
      (e.g. GPT-5.x on OpenAI/Azure) reject it, and OpenRouter's
      ``require_parameters`` then excludes every endpoint.
    - ``structured_outputs``: ``"strict"`` sends ``json_schema`` (constrained
      decoding on enforcing endpoints; OpenRouter also pins routing with
      ``require_parameters``). ``"json_object"`` declares the endpoints only
      guarantee valid JSON — shape is then recovered by the schema-feedback
      repair turn.
    - ``params``: expert passthrough merged into every request body (``seed``,
      ``top_p``, OpenRouter ``provider`` preferences, ...). Library-managed
      keys (``model``, ``messages``, ``response_format``, ``max_tokens`` /
      ``max_completion_tokens``) take precedence over colliding entries.
    - ``max_tokens_key``: ``"max_tokens"`` (default) or ``"max_completion_tokens"``
      for Azure OpenAI-style endpoints that reject ``max_tokens``.
    """

    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    small_model: str | None = None
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None
    temperature: float | None = Field(default=0.0, ge=0.0, le=2.0)
    structured_outputs: Literal["strict", "json_object"] = "strict"
    params: dict[str, object] = Field(default_factory=dict)
    max_tokens: int = Field(default=1800, ge=16)
    max_tokens_key: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    timeout_seconds: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    cache_responses: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    @classmethod
    def from_url(cls, url: str) -> LlmConfig:
        parsed = urlsplit(url)
        if parsed.scheme != "openai":
            raise ConfigError(f"unsupported llm scheme {parsed.scheme!r}; expected 'openai://'")
        api_key = _expand(unquote(parsed.username or "")) or None
        base_url = f"https://{parsed.hostname}" + (f":{parsed.port}" if parsed.port else "")
        path = parsed.path.strip("/")
        if path:
            base_url += f"/{path}"
        query = parse_qs(parsed.query)

        def q(name: str) -> str | None:
            values = query.get(name)
            return values[0] if values else None

        model = q("model")
        temperature = q("temperature")
        max_tokens_raw = q("max_tokens")
        config = cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            small_model=q("small_model"),
            reasoning_effort=q("reasoning"),  # type: ignore[arg-type]  # pydantic validates the Literal
            structured_outputs=q("structured_outputs") or "strict",  # type: ignore[arg-type]
            max_tokens_key=q("max_tokens_key") or "max_tokens",  # type: ignore[arg-type]
        )
        if max_tokens_raw is not None:
            object.__setattr__(config, "max_tokens", int(max_tokens_raw))
        if temperature is not None:
            # "none" omits the parameter entirely (endpoints that reject it).
            value = None if temperature.strip().lower() == "none" else float(temperature)
            object.__setattr__(config, "temperature", value)
        return config


class EmbeddingsProvider(StrEnum):
    HASHING = "hashing"
    OPENAI = "openai"
    LOCAL = "local"


class EmbeddingsConfig(FrozenModel):
    provider: EmbeddingsProvider = EmbeddingsProvider.HASHING
    dimensions: int = Field(default=256, ge=32)
    cache_enabled: bool = True
    cache_max_entries: int = Field(default=50_000, ge=100)
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None

    @model_validator(mode="after")
    def _validate_openai_requirements(self) -> EmbeddingsConfig:
        if self.provider is EmbeddingsProvider.OPENAI and not (self.base_url and self.model):
            raise ConfigError(
                "openai embeddings require base_url and model "
                "(use a full 'openai://key@host/v1?model=...' spec)",
            )
        return self

    @classmethod
    def from_spec(cls, spec: str) -> EmbeddingsConfig:
        text = spec.strip().lower()
        if text == "hashing":
            return cls(provider=EmbeddingsProvider.HASHING)
        if text == "local":
            return cls(
                provider=EmbeddingsProvider.LOCAL,
                model="sentence-transformers/all-MiniLM-L6-v2",
                dimensions=384,
            )
        parsed = urlsplit(spec)
        if parsed.scheme != "openai":
            raise ConfigError(
                f"unknown embeddings spec {spec!r}; use 'hashing', 'local' or 'openai://'"
            )
        model_values = parse_qs(parsed.query).get("model")
        dims_values = parse_qs(parsed.query).get("dimensions")
        return cls(
            provider=EmbeddingsProvider.OPENAI,
            base_url=f"https://{parsed.hostname}"
            + (f":{parsed.port}" if parsed.port else "")
            + parsed.path,
            api_key=_expand(unquote(parsed.username or "")) or None,
            model=(model_values[0] if model_values else "text-embedding-3-small"),
            dimensions=int(dims_values[0]) if dims_values else 1536,
        )


class BatchingConfig(FrozenModel):
    batch_size_messages: int = Field(default=10, ge=1)
    max_age_seconds: float = Field(default=300, ge=10)
    lease_seconds: float = Field(default=120, ge=30)
    server_scope_window: int = Field(default=100, ge=10)


class ExtractionConfig(FrozenModel):
    min_confidence: float = Field(default=0.55, ge=0.0, le=1.0)
    max_candidates_per_batch: int = Field(default=12, ge=1)
    reconcile_collision_threshold: float = Field(default=0.85, ge=0.5, le=1.0)
    # 0.96, not 0.92: near-dup reinforce bypasses the LLM, so it must only fire
    # for true paraphrases. At 0.92, refinements like "promoted to charge nurse"
    # were swallowed into "works as a nurse" without the update ever applying.
    near_duplicate_threshold: float = Field(default=0.96, ge=0.5, le=1.0)
    noise_gate: bool = True
    auto_consolidate_after_adds: int = Field(
        default=5,
        ge=0,
        description=(
            "Regenerate the profile digest once this many active facts exist, "
            "and again each time that many new facts land since the last digest. "
            "0 disables. Counted across the subject's lifetime, not per batch."
        ),
    )
    summary_sanity_threshold: float = Field(default=0.55, ge=0.0, le=1.0)


class LifecycleConfig(FrozenModel):
    short_term_days: int = Field(default=7, ge=1)
    mid_term_days: int = Field(default=45, ge=1)
    long_term_days: int = Field(default=180, ge=1)
    forget_retention_floor: float = Field(default=0.05, ge=0.0, le=1.0)
    max_facts_per_user: int = Field(default=300, ge=10)
    max_server_facts: int = Field(default=500, ge=10)


class RetrievalConfig(FrozenModel):
    rrf_k: int = Field(default=60, ge=1)
    recall_limit: int = Field(default=50, ge=5)
    rerank_pool_size: int = Field(default=100, ge=10)
    candidate_cap: int = Field(default=500, ge=50)
    default_token_budget: int = Field(default=600, ge=64)
    max_per_subject: int = Field(default=4, ge=1)
    hop_depth: int = Field(default=2, ge=1, le=2)
    fan_out_per_node: int = Field(default=24, ge=1)
    reinforce_on_recall: bool = False
    weight_semantic: float = Field(default=0.55, ge=0, le=1)
    weight_lexical: float = Field(default=0.25, ge=0, le=1)
    weight_entity: float = Field(default=0.10, ge=0, le=1)
    weight_strength: float = Field(default=0.10, ge=0, le=1)


class BudgetsConfig(FrozenModel):
    guild_daily_prompt_tokens: int | None = None
    guild_monthly_prompt_tokens: int | None = None


class PrivacyConfig(FrozenModel):
    """Privacy posture knobs.

    ``store_raw_messages=False`` keeps only hashes — trading replay/backfill
    ability for footprint (API.md Part 5). ``processed_retention_days`` prunes
    processed queue rows so the message table stays proportional to live work
    (P0-10: it previously grew forever and was scanned every poll tick).
    """

    store_raw_messages: bool = True
    processed_retention_days: int = Field(default=30, ge=1)


class WorkersConfig(FrozenModel):
    enabled: bool = True
    count: int = Field(default=2, ge=1, le=32)
    poll_interval_seconds: float = Field(default=1.0, gt=0)
    heartbeat_seconds: float = Field(default=20, ge=5)


class ObserveConfig(FrozenModel):
    """Write-path gates applied at observation time (before any LLM cost)."""

    min_message_chars: int = Field(default=3, ge=1)
    ignore_patterns: tuple[str, ...] = ()
    max_queue_depth_per_guild: int | None = Field(default=10_000, ge=1)


class MeterConfig(FrozenModel):
    """Meter backend selection."""

    backend: Literal["memory"] = "memory"


class MemoryConfig(FrozenModel):
    """Root configuration object. See module docstring for URL forms."""

    storage: StorageConfig = Field(default_factory=StorageConfig)
    observe: ObserveConfig = Field(default_factory=ObserveConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    lifecycle: LifecycleConfig = Field(default_factory=LifecycleConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    budgets: BudgetsConfig = Field(default_factory=BudgetsConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    meter_config: MeterConfig = Field(default_factory=MeterConfig)

    @field_validator("storage", mode="before")
    @classmethod
    def _coerce_storage(cls, value: object) -> object:
        return StorageConfig(url=value) if isinstance(value, str) else value

    @field_validator("llm", mode="before")
    @classmethod
    def _coerce_llm(cls, value: object) -> object:
        if value is None:
            return LlmConfig()
        return LlmConfig.from_url(value) if isinstance(value, str) else value

    @field_validator("embeddings", mode="before")
    @classmethod
    def _coerce_embeddings(cls, value: object) -> object:
        return EmbeddingsConfig.from_spec(value) if isinstance(value, str) else value
