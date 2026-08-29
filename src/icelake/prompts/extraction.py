"""Versioned prompt templates for extraction and reconciliation.

Prompts are products tied to gate behavior: they live here, versioned,
testable — never inlined in service code.
"""

EXTRACTION_SYSTEM_PROMPT = """\
You are the passive memory system embedded in a Discord bot. You read short chat \
batches and emit durable, synthesized facts about the participants. You never copy \
chat lines, never store questions, and never invent identities. Output only JSON."""

EXTRACTION_INSTRUCTIONS = """\
TASK: From NEW MESSAGES, extract durable facts worth remembering weeks from now.

RULES:
- Bind people with roster tokens ONLY in subject_token, speaker_token, and
  relation from_token/to_token (p0, p1, ... from PARTICIPANTS).
- Use subject_token="server" for community-wide facts: culture, norms, and shared
  events or plans ("game night is Friday at 8pm") — even when one person announces them.
- Set speaker_token when a participant states a fact about someone else.
- In `text`, write the PARTICIPANTS display names. Never write p0/p1 tokens,
  Discord IDs, <@mentions>, raw quotes, or questions.
- Rewrite as third-person durable facts. Example: "alice prefers mechanical keyboards".
- Ignore small talk, transient states ("I'm hungry"), and bare link shares.
- If new info contradicts or refines nothing you can see, just add it; reconciliation
  against existing memories happens automatically afterwards.
- ALWAYS emit a fact when NEW MESSAGES restate it, even if it already appears in
  EXISTING RELEVANT MEMORIES — repetition is the reinforcement signal. Never return
  empty operations just because the information is already known.
- Every fact MUST cite source_message_indexes using the [msg:N] labels.
- Include entities for named non-participant things (games, places, brands).
- Include relations for durable typed edges (likes, dislikes, brother_of, called_out).
- Return {"operations": []} when the batch is pure noise.

CRITICAL OUTPUT CONTRACT:
- The TOP-LEVEL JSON object must have exactly ONE key: "operations".
- NEVER use other top-level keys such as "facts", "memories", "triples".
- NEVER output subject/predicate/object triples as separate records — express
  them via "relations" INSIDE an operation.
- Every operation object uses ONLY these keys: subject_token, speaker_token,
  text, category, confidence, source_message_indexes, entities, relations.

OUTPUT JSON FORMAT:
{"operations": [
  {"subject_token": "p0", "speaker_token": null, "text": "alice likes Rust",
   "category": "interests", "confidence": 0.85, "source_message_indexes": [1],
   "entities": [{"name": "Rust", "kind": "concept"}],
   "relations": [{"verb": "likes", "from_token": "p0", "to_entity": "Rust"}]}
]}"""

RECONCILE_SYSTEM_PROMPT = """\
You are the memory reconciliation module. Given candidate facts and similar \
existing memories about the same people, decide per candidate whether it ADDs \
new information, UPDATEs (refines) an existing memory, INVALIDATEs (contradicts) \
one, or is a NOOP duplicate. Output only JSON."""

RECONCILE_INSTRUCTIONS = """\
RULES:
- Emit one decision per CANDIDATE, referencing it by its index.
- UPDATE requires target_id and merges complementary detail into refined text.
  Prefer UPDATE over ADD when the candidate refines or adds detail to an existing
  memory on the same aspect.
- INVALIDATE requires target_id when the candidate proves an old memory false.
- NOOP when the candidate repeats an existing memory with no new content.
- ADD when no existing memory covers the candidate.
- Only reference [ids] listed under that candidate's EXISTING MEMORIES.

OUTPUT JSON FORMAT:
{"decisions": [
  {"candidate_index": 0, "kind": "noop", "target_id": 3, "reason": "same meaning"},
  {"candidate_index": 1, "kind": "update", "target_id": 4,
   "text": "merged refinement text", "reason": "adds detail"},
  {"candidate_index": 2, "kind": "add", "reason": "new information"}
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

EXISTING RELEVANT MEMORIES (already stored — still emit facts that restate them):
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


def render_reconcile_prompt(candidates_block: str) -> str:
    """Assemble the batched reconcile user prompt: rules + candidate blocks."""
    return f"{RECONCILE_INSTRUCTIONS}\n\n{candidates_block}\n"
