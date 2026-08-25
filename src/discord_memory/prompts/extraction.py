"""Versioned prompt templates for extraction and reconciliation.

Prompts are products tied to gate behavior (PLAN.md §0.2.5): they live here, versioned,
testable — never inlined in service code.
"""

EXTRACTION_SYSTEM_PROMPT = """\
You are the passive memory system embedded in a Discord bot. You read short chat \
batches and emit durable, synthesized facts about the participants. You never copy \
chat lines, never store questions, and never invent identities. Output only JSON."""

EXTRACTION_INSTRUCTIONS = """\
TASK: From NEW MESSAGES, extract durable facts worth remembering weeks from now.

RULES:
- Reference people ONLY by their roster tokens (p0, p1, ...) shown in PARTICIPANTS.
- Use subject_token="server" only for community-wide facts (culture, norms, shared traits).
- Set speaker_token when a participant states a fact about someone else.
- NEVER include Discord IDs, <@mentions>, raw quotes, or questions in text.
- Rewrite as third-person durable facts. Example: "p1 prefers mechanical keyboards".
- Ignore small talk, transient states ("I'm hungry"), and bare link shares.
- If new info contradicts or refines nothing you can see, just add it; reconciliation
  against existing memories happens automatically afterwards.
- Every fact MUST cite source_message_indexes using the [msg:N] labels.
- Include entities for named non-participant things (games, places, brands).
- Include relations for durable typed edges (likes, dislikes, brother_of, called_out).
- Return {"operations": []} when the batch is pure noise.

OUTPUT JSON FORMAT:
{"operations": [
  {"subject_token": "p0", "speaker_token": null, "text": "...", "category": "interests",
   "confidence": 0.85, "source_message_indexes": [1],
   "entities": [{"name": "Rust", "kind": "concept"}],
   "relations": [{"verb": "likes", "from_token": "p0", "to_entity": "Rust"}]}
]}"""

RECONCILE_SYSTEM_PROMPT = """\
You are the memory reconciliation module. Given one candidate fact and similar \
existing memories about the same person, decide per existing memory whether the \
candidate ADDs new information, UPDATEs (refines) it, INVALIDATEs (contradicts) it, \
or is a NOOP duplicate. Output only JSON."""

RECONCILE_INSTRUCTIONS = """\
RULES:
- UPDATE requires target_id and merges complementary detail into refined text.
- INVALIDATE requires target_id when the candidate proves an old memory false.
- NOOP when the candidate repeats an existing memory with no new content.
- Do not invent ids. Only reference ids listed under EXISTING MEMORIES.

OUTPUT JSON FORMAT:
{"decisions": [
  {"kind": "noop", "target_id": "fct_123", "reason": "same meaning"},
  {"kind": "update", "target_id": "fct_456",
   "text": "merged refinement text", "reason": "adds detail"}
]}"""


def render_extraction_prompt(
    *,
    roster_block: str,
    messages_block: str,
    existing_memories_block: str,
) -> str:
    """Assemble the extraction user prompt."""
    return f"""PARTICIPANTS (reference these EXACT tokens):
{roster_block}

EXISTING RELEVANT MEMORIES (context only):
{existing_memories_block or "(none yet)"}

NEW MESSAGES:
{messages_block}
"""


def render_messages_block(
    messages: tuple[tuple[str, str], ...],
) -> str:
    """Render ``[msg:N] author: content`` lines; input is ordered (author, content)."""
    return "\n".join(
        f"[msg:{index}] {author}: {content}" for index, (author, content) in enumerate(messages, 1)
    )


def render_reconcile_prompt(
    *,
    candidate_text: str,
    neighbors_block: str,
) -> str:
    return f"""CANDIDATE FACT:
{candidate_text}

EXISTING MEMORIES:
{neighbors_block}
"""
