"""Config contract tests: URL parsing, coercion, typo protection."""

from __future__ import annotations

import pytest

from discord_memory.config import (
    EmbeddingsConfig,
    EmbeddingsProvider,
    LlmConfig,
    MemoryConfig,
)
from discord_memory.errors import ConfigError


def test_default_config_is_valid() -> None:
    config = MemoryConfig()
    assert config.storage.backend == "sqlite"
    assert not config.llm.enabled
    assert config.embeddings.provider is EmbeddingsProvider.HASHING


def test_sqlite_url_forms() -> None:
    assert MemoryConfig(storage="sqlite:///data/m.db").storage.url.endswith("m.db")
    assert MemoryConfig(storage="sqlite://:memory:").storage.backend == "sqlite"


def test_unknown_storage_scheme_raises() -> None:
    with pytest.raises(ConfigError, match="unsupported storage"):
        MemoryConfig(storage="oracle://x")


def test_llm_url_parses_key_host_model() -> None:
    config = MemoryConfig(
        llm="openai://sk-key@openrouter.ai/api/v1?model=test-model&temperature=0.5",
    )
    assert config.llm.enabled
    assert config.llm.base_url == "https://openrouter.ai/api/v1"
    assert config.llm.api_key == "sk-key"
    assert config.llm.model == "test-model"
    assert config.llm.temperature == 0.5


def test_llm_url_parses_small_model() -> None:
    config = MemoryConfig(llm="openai://k@host/v1?model=big&small_model=cheap")
    assert config.llm.model == "big"
    assert config.llm.small_model == "cheap"


def test_small_model_routes_reconcile_classify_consolidation() -> None:
    from discord_memory import DiscordMemory

    config = MemoryConfig(
        storage="sqlite://:memory:",
        llm="openai://k@host/v1?model=big&small_model=cheap",
    )
    client = DiscordMemory(config)
    assert client._llm is not None and client._llm.model_name == "big"
    assert client._small_llm is not None and client._small_llm.model_name == "cheap"
    assert client._pipeline._reconciler._llm is client._small_llm
    assert client._classifier._llm is client._small_llm


def test_small_model_defaults_to_main_llm() -> None:
    from discord_memory import DiscordMemory

    config = MemoryConfig(
        storage="sqlite://:memory:",
        llm="openai://k@host/v1?model=big",
    )
    client = DiscordMemory(config)
    assert client._small_llm is client._llm


@pytest.mark.parametrize("env_name", ["TEST_LLM_KEY"])
def test_llm_url_expands_env(monkeypatch: pytest.MonkeyPatch, env_name: str) -> None:
    monkeypatch.setenv(env_name, "secret-value")
    config = MemoryConfig(llm=f"openai://${env_name}@host/v1?model=m")
    assert config.llm.api_key == "secret-value"


def test_llm_missing_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISSING_KEY_XYZ", raising=False)
    with pytest.raises(ConfigError, match="not set"):
        MemoryConfig(llm="openai://$MISSING_KEY_XYZ@host/v1?model=m")


def test_bad_llm_scheme_raises() -> None:
    with pytest.raises(ConfigError, match="unsupported llm scheme"):
        MemoryConfig(llm="anthropic://k@h")


def test_embeddings_spec_hashing_and_local() -> None:
    hashing = MemoryConfig(embeddings="hashing")
    assert hashing.embeddings.provider is EmbeddingsProvider.HASHING
    local = MemoryConfig(embeddings="local")
    assert local.embeddings.dimensions == 384


def test_openai_embeddings_requires_full_spec() -> None:
    with pytest.raises(ConfigError, match="openai embeddings require"):
        EmbeddingsConfig(provider=EmbeddingsProvider.OPENAI)


def test_unknown_top_level_key_rejected() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(**{"storag": "sqlite://:memory:"})  # type: ignore[arg-type]


def test_openai_embedder_requires_credentials() -> None:
    from discord_memory.adapters.embedders import OpenAICompatEmbedder

    with pytest.raises(ConfigError):
        OpenAICompatEmbedder(LlmConfig(base_url=None, model="m"))  # type: ignore[arg-type]
