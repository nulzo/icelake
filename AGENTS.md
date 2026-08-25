# Core Principles (Ordered - Higher Wins on Conflict)

## Code is a liability, not an asset

- Before writing code, answer: (1) Does a dependency or existing module already do
  this? (2) What breaks upstream? (3) What depends on this downstream?
- Equal designs -> fewer lines wins. No speculative abstractions, no "just in case"
  helpers, no unused parameters.
- Deleting code is a feature. When a module loses its justification, say so.
- Never add code to appear thorough. Deliberate, minimal additions only.


## Adopt proven machinery; never rebuild it

- If a mature, maintained dependency solves the problem, use it — don't hand-roll
  a weaker version. Check the project's architecture decision records / plan docs
  for what has been adopted and what is explicitly out of scope.
- Conversely: if a dependency is deprecated or unmaintained, flag it rather than
  building deeper on top of it.

## One concern per module

- Each module owns one concern and exposes it through a narrow interface.
  Reaching into another module's internals is a violation — use its contract.

## Design for the swap you can't predict

- Every external dependency (database, LLM provider, library) is treated as
  replaceable. The test: "if we swapped X tomorrow, how many files change?"
  If the answer is more than the adapter plus config, the abstraction has leaked.


# Code Quality Standards

## Baseline

- Full type annotations on all public interfaces; strict type-checker clean.
- Linter + formatter clean before merge; no suppressions without a comment
  explaining why.
- All data crossing a boundary (API I/O, persistence, inter-process, LLM I/O)
  is validated by a schema. No raw dicts at boundaries.
- Fixed sets of values are enums, never raw strings.

## Clarity

- Names state intent and units (`timeout_seconds`, not `timeout`).
- Public capability entry points carry user-facing docstrings — treat them as
  product surface, not developer notes.
- Async on all I/O paths; never block the event loop.

## Size gates (need explicit justification to exceed)

- Function: ~40 lines. Module: ~300 lines. Justification means a comment or PR
  note explaining why splitting would be worse.

## Testing

- Business logic must be testable without transport, network, or live services —
  inject fakes at the seams. Core correctness logic (validation, verification,
  state transitions) gets exhaustive tests.
- Test behavior at interfaces, not implementation details.

## Every change passes this checklist

1. Does something that exists already do this?
2. What breaks upstream? What depends on this downstream?
3. Can this be fewer lines?
4. Is this logic duplicated elsewhere (or does it belong in a shared layer)?
5. Are new boundaries schema-validated?
6. Are new costs (API calls, storage growth, latency) metered and bounded?
7. Does this work with N processes, or only one?

## Hygiene

- No commented-out code (git is the history). No TODOs without a linked issue.
- No dead config knobs, no unused exports.
