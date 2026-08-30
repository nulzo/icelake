Bench matrix is a per-model OpenRouter param map (reasoning, temperature,
structured_outputs) instead of one global flag. LlmConfig accepts
`reasoning=none` to disable thinking on models that allow it.
