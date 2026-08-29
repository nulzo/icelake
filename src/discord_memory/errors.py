"""Error taxonomy for discord-memory.

Consumer-facing contract (docs/API.md Part 13):

- ``observe`` raises only ``SchemaValidationError``.
- Retrieval methods never raise operational errors; they report degradation.
- CRUD/admin methods raise specific subclasses.
"""

from __future__ import annotations


class DiscordMemoryError(Exception):
    """Base class for every error raised by this library."""


class ConfigError(DiscordMemoryError):
    """Invalid or unknown configuration (URLs, knobs, provider schemes)."""


class SchemaValidationError(DiscordMemoryError):
    """Input failed boundary validation (malformed event, naive datetime, bad payload)."""


class SubjectNotAllowedError(DiscordMemoryError):
    """Target subject is a bot, opted out, or otherwise barred from memory."""


class FactNotFoundError(DiscordMemoryError):
    """Referenced fact id does not exist."""


class IdentityAmbiguousError(DiscordMemoryError):
    """Identifier matched multiple members and cannot be safely resolved."""

    def __init__(self, identifier: str, candidate_ids: tuple[str, ...]) -> None:
        self.identifier = identifier
        self.candidate_ids = candidate_ids
        super().__init__(
            f"identifier {identifier!r} matches {len(candidate_ids)} members; "
            "disambiguate before writing"
        )


class StorageUnavailableError(DiscordMemoryError):
    """Storage backend could not be reached or prepared."""


class LlmCapabilityError(DiscordMemoryError):
    """The LLM endpoint rejected the request's parameter set (HTTP 400/404/422).

    Raised instead of silently degrading: the message names the model, the
    provider's own error text, and the ``LlmConfig`` knobs
    (``temperature=None``, ``structured_outputs="json_object"``, ``params``)
    that resolve it. Fix the configuration; don't retry.
    """


class StructuredOutputError(DiscordMemoryError):
    """LLM JSON did not validate against the declared schema after one repair.

    Extraction treats this as a batch failure (dead-letter + ``ExtractionFailed``).
    Classify and reconcile may apply a documented domain default instead.
    """


class BudgetExceededError(DiscordMemoryError):
    """A hard-stop budget was exhausted for the requested operation."""


class WorkerNotRunningError(DiscordMemoryError):
    """Operation requires started workers but none are running."""
