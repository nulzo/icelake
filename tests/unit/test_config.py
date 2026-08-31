"""Config contract tests: URL parsing, coercion, typo protection."""

from __future__ import annotations

import pytest

from icelake.config import (
    EmbeddingsConfig,
    EmbeddingsProvider,
    LlmConfig,
    MemoryConfig,
)
from icelake.errors import ConfigError


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


def test_llm_url_parses_reasoning_effort() -> None:
    config = MemoryConfig(llm="openai://k@host/v1?model=big&reasoning=low")
    assert config.llm.reasoning_effort == "low"
    assert MemoryConfig(llm="openai://k@host/v1?model=big").llm.reasoning_effort is None
    off = MemoryConfig(llm="openai://k@host/v1?model=big&reasoning=none")
    assert off.llm.reasoning_effort == "none"


def test_llm_url_parses_capability_knobs() -> None:
    config = MemoryConfig(
        llm="openai://k@host/v1?model=big&temperature=none&structured_outputs=json_object"
    )
    assert config.llm.temperature is None
    assert config.llm.structured_outputs == "json_object"

    defaulted = MemoryConfig(llm="openai://k@host/v1?model=big&temperature=0.2")
    assert defaulted.llm.temperature == 0.2
    assert defaulted.llm.structured_outputs == "strict"


def test_llm_url_parses_max_tokens_and_key() -> None:
    config = MemoryConfig(
        llm="openai://k@host/v1?model=big&max_tokens=512&max_tokens_key=max_completion_tokens"
    )
    assert config.llm.max_tokens == 512
    assert config.llm.max_tokens_key == "max_completion_tokens"


def test_postgres_url_is_recognized_but_unimplemented() -> None:
    from icelake import DiscordMemory

    config = MemoryConfig(storage="postgresql://localhost/memory")
    assert config.storage.backend == "postgres"
    with pytest.raises(ConfigError, match="not implemented"):
        DiscordMemory(config)


def test_small_model_routes_reconcile_classify_consolidation() -> None:
    from icelake import DiscordMemory

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
    from icelake import DiscordMemory

    config = MemoryConfig(
        storage="sqlite://:memory:",
        llm="openai://k@host/v1?model=big",
    )
    client = DiscordMemory(config)
    assert client._small_llm is client._llm


def test_openrouter_host_selects_openrouter_adapter() -> None:
    from icelake import DiscordMemory
    from icelake.adapters.llm_openai_compat import OpenAICompatLLM
    from icelake.adapters.llm_openrouter import OpenRouterLLM

    openrouter = DiscordMemory(
        MemoryConfig(
            storage="sqlite://:memory:",
            llm="openai://k@openrouter.ai/api/v1?model=big",
        )
    )
    assert isinstance(openrouter._llm._inner, OpenRouterLLM)  # MeteredLLM wraps it

    generic = DiscordMemory(
        MemoryConfig(storage="sqlite://:memory:", llm="openai://k@host/v1?model=big")
    )
    inner = generic._llm._inner  # type: ignore[union-attr]
    assert type(inner) is OpenAICompatLLM


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


def test_public_capability_types_are_exported() -> None:
    from icelake import LlmCapabilityError, ObserveConfig, StructuredOutputError

    assert issubclass(LlmCapabilityError, Exception)
    assert issubclass(StructuredOutputError, Exception)
    assert ObserveConfig is not None


def test_retrieval_caps_default_and_override() -> None:
    retrieval = MemoryConfig().retrieval
    assert retrieval.top_k == 8
    assert retrieval.max_per_subject == 4
    widened = MemoryConfig(retrieval={"top_k": 30, "max_per_subject": 12}).retrieval
    assert widened.top_k == 30
    assert widened.max_per_subject == 12


def test_unknown_top_level_key_rejected() -> None:
    with pytest.raises(ValueError):
        MemoryConfig(**{"storag": "sqlite://:memory:"})  # type: ignore[arg-type]


def test_openai_embedder_requires_credentials() -> None:
    from icelake.adapters.embedders import OpenAICompatEmbedder

    with pytest.raises(ConfigError):
        OpenAICompatEmbedder(LlmConfig(base_url=None, model="m"))  # type: ignore[arg-type]
