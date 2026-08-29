"""OpenRouter-specific LLM adapter: routing constraints, real cost, reasoning dial.

Extends the generic OpenAI-compatible client with three OpenRouter features
(docs: provider-routing, structured-outputs, usage-accounting):

- ``provider.require_parameters`` pins structured requests to endpoints that
  actually enforce json_schema (strict mode only). If the declared parameter
  set excludes every endpoint, OpenRouter 404s at attempt 0 and the base
  client raises ``LlmCapabilityError`` — fix the declared capabilities
  (e.g. ``temperature=None`` for reasoning-model endpoints) rather than
  silently losing enforcement.
- ``usage.include`` returns the real charged cost per call (``usage.cost``).
- ``reasoning.effort`` forwards the configured reasoning dial.
"""

from __future__ import annotations

import logging

from icelake.adapters.llm_openai_compat import OpenAICompatLLM
from icelake.ports.llm import ChatRequest

logger = logging.getLogger(__name__)


class OpenRouterLLM(OpenAICompatLLM):
    """OpenAI-compat client with OpenRouter routing constraints and cost reporting."""

    def _apply_provider_extras(self, body: dict[str, object], request: ChatRequest) -> None:
        body["usage"] = {"include": True}
        if self._config.reasoning_effort:
            body["reasoning"] = {"effort": self._config.reasoning_effort}
        if request.response_schema is not None and self._config.structured_outputs == "strict":
            # Merge over any user-declared provider prefs from LlmConfig.params.
            existing = body.get("provider")
            provider: dict[str, object] = dict(existing) if isinstance(existing, dict) else {}
            provider["require_parameters"] = True
            body["provider"] = provider


__all__ = ["OpenRouterLLM"]
