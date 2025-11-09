"""
Database package for Giro Agent.

Provides SQLAlchemy models, enums, and database utilities.
"""

from database.base import Base
from database.enums import (
    UserRole,
    LearningStyle,
    MessageRole,
    SessionStatus,
    ExitReason,
    SourceType,
    SyncStatus,
)
from database.models import (
    User,
    LearningProfile,
    Topic,
    KnowledgeSource,
    LearningSession,
    Message,
    SessionState,
    AgentConfig,
)

__all__ = [
    # Base
    "Base",
    # Enums
    "UserRole",
    "LearningStyle",
    "MessageRole",
    "SessionStatus",
    "ExitReason",
    "SourceType",
    "SyncStatus",
    # Models
    "User",
    "LearningProfile",
    "Topic",
    "KnowledgeSource",
    "LearningSession",
    "Message",
    "SessionState",
    "AgentConfig",
]