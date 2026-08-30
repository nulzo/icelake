Documented the name-in-prose lookup pattern ("what do you know about X?" with no
@mention): a new README section explains that `prompt_context` is mention-keyed by
design and shows the resolve-then-strict-fetch flow (`identity.resolve` →
`facts.list_for_subject`, never guessing on ambiguity) across three entry points —
slash command (zero LLM), a structured-output router, and native function calling.
Added `examples/name_lookup_tool.py`, a runnable LLM-free demo of the shared
handler, and wired the pattern into `examples/omni_style_bot.py` (router +
`/memory lookup`).
