"""
Database enums for type-safe column values.

Defines all enum types used across the database schema.
"""

import enum


class UserRole(str, enum.Enum):
    """User roles in the system."""

    STUDENT = "student"
    EDUCATOR = "educator"
    ADMIN = "admin"

    def __str__(self) -> str:
        return self.value


class LearningStyle(str, enum.Enum):
    """Detected learning styles for students."""

    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


class MessageRole(str, enum.Enum):
    """Message roles in conversation."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    def __str__(self) -> str:
        return self.value


class SessionStatus(str, enum.Enum):
    """Learning session status."""

    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    ENDED = "ended"

    def __str__(self) -> str:
        return self.value


class ExitReason(str, enum.Enum):
    """Reason for session exit."""

    ACHIEVED_GOAL = "achieved_goal"
    MAX_TURNS = "max_turns"
    USER_EXIT = "user_exit"
    ERROR = "error"
    TIMEOUT = "timeout"

    def __str__(self) -> str:
        return self.value


class SourceType(str, enum.Enum):
    """Type of knowledge source."""

    DOCUMENT = "document"
    URL = "url"
    TEXT = "text"

    def __str__(self) -> str:
        return self.value


class SyncStatus(str, enum.Enum):
    """Bedrock Knowledge Base sync status."""

    PENDING = "pending"
    SYNCING = "syncing"
    SYNCED = "synced"
    FAILED = "failed"

    def __str__(self) -> str:
        return self.value