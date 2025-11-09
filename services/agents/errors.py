"""
Error hierarchy for AI agents.

Provides structured, specific errors for better error handling and debugging.
"""


class AgentError(Exception):
    """Base error for all agent-related errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)


# Provider Errors
class ProviderError(AgentError):
    """Base error for LLM provider issues."""

    pass


class StructuredOutputError(ProviderError):
    """Failed to extract structured output."""

    pass


# State Management Errors
class StateError(AgentError):
    """Base error for state management issues."""

    pass


class StatePersistenceError(StateError):
    """Failed to persist state to database."""

    pass


class ContextLoadError(StateError):
    """Failed to load conversation context from database."""

    pass
