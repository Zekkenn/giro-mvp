"""
Database models for Giro Agent educational platform.

This module contains all SQLAlchemy ORM models for the application.
Models are organized by domain: User, Learning, Knowledge, Session, Agent.
"""

from typing import Optional, List
from datetime import datetime

from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Boolean,
    Float,
    ForeignKey,
    JSON,
    DateTime,
    Enum as SQLEnum,
    Index,
    UniqueConstraint,
    CheckConstraint,
)
from sqlalchemy.orm import relationship, Mapped
from sqlalchemy.sql import func

from database.base import Base
from database.mixins import TimestampMixin, SoftDeleteMixin
from database.enums import (
    UserRole,
    LearningStyle,
    MessageRole,
    SessionStatus,
    ExitReason,
    SourceType,
    SyncStatus,
)


# ============================================================================
# User Models
# ============================================================================


class User(Base, TimestampMixin):
    """
    User entity with OAuth integration.

    Represents both students and educators. Authentication is delegated to
    OAuth providers (Google, Microsoft, etc.) via external_id mapping.

    Attributes:
        id: Primary key
        external_id: OAuth provider's user ID (unique)
        email: User email (unique)
        username: Display username
        full_name: User's full name
        avatar_url: Profile avatar URL
        role: User role (student, educator, admin)
        is_active: Account status
        metadata: OAuth and additional data (JSON)
        last_login_at: Last login timestamp
    """

    __tablename__ = "users"

    # Primary key
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)

    # OAuth integration
    external_id: Mapped[str] = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="OAuth provider's user ID",
    )

    # User information
    email: Mapped[str] = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="User email address",
    )

    username: Mapped[Optional[str]] = Column(
        String(100),
        unique=True,
        nullable=True,
        comment="Display username",
    )

    full_name: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        comment="User's full name",
    )

    avatar_url: Mapped[Optional[str]] = Column(
        String(512),
        nullable=True,
        comment="Profile avatar URL",
    )

    # Role and status
    role: Mapped[UserRole] = Column(
        SQLEnum(UserRole),
        nullable=False,
        default=UserRole.STUDENT,
        comment="User role in the system",
    )

    is_active: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Account status",
    )

    # Metadata
    metadata_: Mapped[dict] = Column(
        "metadata",  # Column name in DB
        JSON,
        nullable=False,
        default=dict,
        comment="OAuth provider data and additional metadata",
    )

    last_login_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last login timestamp",
    )

    # Relationships
    learning_profile: Mapped[Optional["LearningProfile"]] = relationship(
        "LearningProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )

    learning_sessions: Mapped[List["LearningSession"]] = relationship(
        "LearningSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    created_topics: Mapped[List["Topic"]] = relationship(
        "Topic",
        back_populates="creator",
        foreign_keys="Topic.created_by",
    )

    uploaded_knowledge: Mapped[List["KnowledgeSource"]] = relationship(
        "KnowledgeSource",
        back_populates="uploader",
        foreign_keys="KnowledgeSource.uploaded_by",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', role={self.role})>"


# ============================================================================
# Learning Profile Models
# ============================================================================


class LearningProfile(Base, TimestampMixin, SoftDeleteMixin):
    """
    Student learning profile with AI-generated insights.

    Tracks learning style, preferences, progress, and AI-generated analysis
    of the student's learning patterns and orientation.

    Attributes:
        id: Primary key
        user_id: Foreign key to users table
        learning_style: Detected learning style
        learning_style_confidence: Confidence score (0.0-1.0)
        learning_style_detected_at: When style was detected
        learning_style_indicators: Evidence for detection (JSON array)
        ai_summary: AI-generated analysis of student
        ai_summary_updated_at: When AI summary was last updated
        total_sessions: Count of sessions completed
        total_interactions: Count of total interactions
        topics_studied: List of topic IDs studied (JSON array)
        metadata_: Additional profile data (JSON)
    """

    __tablename__ = "learning_profiles"

    # Primary key
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)

    # User relationship
    user_id: Mapped[int] = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        comment="User this profile belongs to",
    )

    # Learning style detection
    learning_style: Mapped[LearningStyle] = Column(
        SQLEnum(LearningStyle),
        nullable=False,
        default=LearningStyle.UNKNOWN,
        comment="Detected learning style",
    )

    learning_style_confidence: Mapped[float] = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="Confidence score for learning style (0.0-1.0)",
    )

    learning_style_detected_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When learning style was detected",
    )

    learning_style_indicators: Mapped[list] = Column(
        JSON,
        nullable=False,
        default=list,
        comment="Evidence for learning style detection",
    )

    # AI-generated summary
    ai_summary: Mapped[Optional[str]] = Column(
        Text,
        nullable=True,
        comment="AI-generated analysis of student's learning patterns and orientation",
    )

    ai_summary_updated_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="When AI summary was last updated",
    )

    # Progress tracking
    total_sessions: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of learning sessions",
    )

    total_interactions: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Total number of interactions across all sessions",
    )

    topics_studied: Mapped[list] = Column(
        JSON,
        nullable=False,
        default=list,
        comment="List of topic IDs studied",
    )

    # Metadata
    metadata_: Mapped[dict] = Column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        comment="Additional profile data",
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="learning_profile")

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "learning_style_confidence >= 0.0 AND learning_style_confidence <= 1.0",
            name="ck_learning_style_confidence_range",
        ),
        CheckConstraint(
            "total_sessions >= 0",
            name="ck_total_sessions_positive",
        ),
        CheckConstraint(
            "total_interactions >= 0",
            name="ck_total_interactions_positive",
        ),
    )

    def __repr__(self) -> str:
        return f"<LearningProfile(user_id={self.user_id}, style={self.learning_style})>"


# ============================================================================
# Topic Models
# ============================================================================


class Topic(Base, TimestampMixin, SoftDeleteMixin):
    """
    Educational topic/subject (hybrid: global + user-created).

    Topics can be global (admin-created, visible to all) or user-created
    (visible only to creator). Supports hierarchical organization.

    Attributes:
        id: Primary key
        name: Topic name
        slug: URL-friendly identifier
        description: Topic description
        learning_objectives: Markdown-formatted objectives
        icon: Icon name or emoji
        parent_id: Parent topic for hierarchy
        is_global: True if admin-created and visible to all
        created_by: User who created topic (NULL for global)
        is_active: Soft activation status
        metadata_: Additional topic data (JSON)
    """

    __tablename__ = "topics"

    # Primary key
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)

    # Topic information
    name: Mapped[str] = Column(
        String(255),
        nullable=False,
        index=True,
        comment="Topic name",
    )

    slug: Mapped[str] = Column(
        String(255),
        nullable=False,
        index=True,
        comment="URL-friendly slug",
    )

    description: Mapped[Optional[str]] = Column(
        Text,
        nullable=True,
        comment="Topic description",
    )

    learning_objectives: Mapped[Optional[str]] = Column(
        Text,
        nullable=True,
        comment="Markdown-formatted learning objectives",
    )

    icon: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        comment="Icon name or emoji",
    )

    # Hierarchy
    parent_id: Mapped[Optional[int]] = Column(
        Integer,
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
        comment="Parent topic for hierarchical organization",
    )

    # Global vs user-created
    is_global: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="True if admin-created and visible to all",
    )

    created_by: Mapped[Optional[int]] = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        comment="User who created topic (NULL for global topics)",
    )

    # Status
    is_active: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Topic activation status",
    )

    # Metadata
    metadata_: Mapped[dict] = Column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        comment="Additional topic data",
    )

    # Relationships
    creator: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="created_topics",
        foreign_keys=[created_by],
    )

    parent: Mapped[Optional["Topic"]] = relationship(
        "Topic",
        remote_side=[id],
        back_populates="children",
    )

    children: Mapped[List["Topic"]] = relationship(
        "Topic",
        back_populates="parent",
        cascade="all, delete-orphan",
    )

    knowledge_sources: Mapped[List["KnowledgeSource"]] = relationship(
        "KnowledgeSource",
        back_populates="topic",
    )

    learning_sessions: Mapped[List["LearningSession"]] = relationship(
        "LearningSession",
        back_populates="topic",
    )

    # Indexes and constraints
    __table_args__ = (
        # Unique constraint for user-scoped topics
        UniqueConstraint("name", "created_by", name="uq_topic_name_per_user"),
        UniqueConstraint("slug", "created_by", name="uq_topic_slug_per_user"),
        # Index for global/active topics
        Index("ix_topics_is_global_is_active", "is_global", "is_active"),
    )

    def __repr__(self) -> str:
        return f"<Topic(id={self.id}, name='{self.name}', is_global={self.is_global})>"


# ============================================================================
# Knowledge Source Models
# ============================================================================


class KnowledgeSource(Base, TimestampMixin, SoftDeleteMixin):
    """
    Documents/sources indexed in AWS Bedrock Knowledge Base.

    Multi-tenant with partitioning: user content is isolated via partition_key,
    global content is accessible to all users.

    Attributes:
        id: Primary key
        topic_id: Related topic
        title: Source title
        description: Source description
        source_type: Type of source (document, url, text)
        is_global: True if accessible to all users
        file_name: Original filename
        file_size: File size in bytes
        s3_bucket: AWS S3 bucket name
        s3_key: AWS S3 object key
        partition_key: Partitioning key for multi-tenancy
        bedrock_kb_id: Bedrock Knowledge Base ID
        bedrock_data_source_id: Bedrock Data Source ID
        vector_store_id: Vector store index ID
        sync_status: Sync status with Bedrock
        sync_error: Error message if sync failed
        last_synced_at: Last successful sync timestamp
        chunk_count: Number of chunks created
        metadata_: Additional source data (JSON)
        uploaded_by: User who uploaded the source
    """

    __tablename__ = "knowledge_sources"

    # Primary key
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)

    # Topic relationship
    topic_id: Mapped[Optional[int]] = Column(
        Integer,
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Related topic",
    )

    # Source information
    title: Mapped[str] = Column(
        String(512),
        nullable=False,
        comment="Source title",
    )

    description: Mapped[Optional[str]] = Column(
        Text,
        nullable=True,
        comment="Source description",
    )

    source_type: Mapped[SourceType] = Column(
        SQLEnum(SourceType),
        nullable=False,
        default=SourceType.DOCUMENT,
        comment="Type of source",
    )

    # Multi-tenancy
    is_global: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
        comment="True if accessible to all users",
    )

    # File information
    file_name: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        comment="Original filename",
    )

    file_size: Mapped[Optional[int]] = Column(
        Integer,
        nullable=True,
        comment="File size in bytes",
    )

    # AWS S3 storage
    s3_bucket: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        comment="AWS S3 bucket name",
    )

    s3_key: Mapped[Optional[str]] = Column(
        String(512),
        nullable=True,
        comment="AWS S3 object key",
    )

    # Multi-tenant partitioning
    partition_key: Mapped[str] = Column(
        String(255),
        nullable=False,
        index=True,
        comment="Partitioning key for multi-tenancy (user_<id> or 'global')",
    )

    # AWS Bedrock integration
    bedrock_kb_id: Mapped[str] = Column(
        String(255),
        nullable=False,
        index=True,
        comment="Bedrock Knowledge Base ID",
    )

    bedrock_data_source_id: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Bedrock Data Source ID",
    )

    vector_store_id: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        comment="Vector store index ID (OpenSearch/Kendra)",
    )

    # Sync status
    sync_status: Mapped[SyncStatus] = Column(
        SQLEnum(SyncStatus),
        nullable=False,
        default=SyncStatus.PENDING,
        index=True,
        comment="Sync status with Bedrock",
    )

    sync_error: Mapped[Optional[str]] = Column(
        Text,
        nullable=True,
        comment="Error message if sync failed",
    )

    last_synced_at: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Last successful sync timestamp",
    )

    chunk_count: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of chunks created",
    )

    # Metadata
    metadata_: Mapped[dict] = Column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        comment="Additional source data (tags, author, version, etc.)",
    )

    # Uploader
    uploaded_by: Mapped[Optional[int]] = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="User who uploaded the source",
    )

    # Relationships
    topic: Mapped[Optional["Topic"]] = relationship(
        "Topic",
        back_populates="knowledge_sources",
    )

    uploader: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="uploaded_knowledge",
        foreign_keys=[uploaded_by],
    )

    # Indexes and constraints
    __table_args__ = (
        Index("ix_knowledge_bedrock_kb_ds", "bedrock_kb_id", "bedrock_data_source_id"),
        Index("ix_knowledge_topic_sync", "topic_id", "sync_status"),
        Index("ix_knowledge_uploader_global", "uploaded_by", "is_global"),
        CheckConstraint(
            "chunk_count >= 0",
            name="ck_chunk_count_positive",
        ),
        CheckConstraint(
            "file_size IS NULL OR file_size >= 0",
            name="ck_file_size_positive",
        ),
    )

    def __repr__(self) -> str:
        return f"<KnowledgeSource(id={self.id}, title='{self.title}', sync_status={self.sync_status})>"


# ============================================================================
# Session Models
# ============================================================================


class LearningSession(Base, TimestampMixin):
    """
    Individual learning session between student and agent.

    Tracks conversation flow, learning outcomes, and session metrics.

    Attributes:
        id: Primary key
        user_id: Student user ID
        topic_id: Topic being studied
        agent_config_id: Agent configuration used
        session_id: External session identifier (unique)
        status: Session status
        interaction_count: Number of interactions
        start_time: Session start timestamp
        end_time: Session end timestamp
        duration_seconds: Session duration
        detected_learning_style: Learning style detected during session
        exit_reason: Reason for session exit
        metadata_: Session data (sources used, topics covered, etc.)
    """

    __tablename__ = "learning_sessions"

    # Primary key
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)

    # User relationship
    user_id: Mapped[int] = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Student user ID",
    )

    # Topic relationship
    topic_id: Mapped[Optional[int]] = Column(
        Integer,
        ForeignKey("topics.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Topic being studied",
    )

    # Agent configuration
    agent_config_id: Mapped[Optional[int]] = Column(
        Integer,
        ForeignKey("agent_configs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        comment="Agent configuration used for this session",
    )

    # Session identification
    session_id: Mapped[str] = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="External session identifier",
    )

    # Session status
    status: Mapped[SessionStatus] = Column(
        SQLEnum(SessionStatus),
        nullable=False,
        default=SessionStatus.ACTIVE,
        comment="Session status",
    )

    # Metrics
    interaction_count: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=0,
        comment="Number of interactions in session",
    )

    start_time: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Session start timestamp",
    )

    end_time: Mapped[Optional[datetime]] = Column(
        DateTime(timezone=True),
        nullable=True,
        comment="Session end timestamp",
    )

    duration_seconds: Mapped[Optional[int]] = Column(
        Integer,
        nullable=True,
        comment="Session duration in seconds",
    )

    # Learning outcomes
    detected_learning_style: Mapped[Optional[LearningStyle]] = Column(
        SQLEnum(LearningStyle),
        nullable=True,
        comment="Learning style detected during this session",
    )

    exit_reason: Mapped[Optional[ExitReason]] = Column(
        SQLEnum(ExitReason),
        nullable=True,
        comment="Reason for session exit",
    )

    # Metadata
    metadata_: Mapped[dict] = Column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        comment="Session data (sources used, topics covered, etc.)",
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="learning_sessions")

    topic: Mapped[Optional["Topic"]] = relationship(
        "Topic",
        back_populates="learning_sessions",
    )

    agent_config: Mapped[Optional["AgentConfig"]] = relationship(
        "AgentConfig",
        back_populates="learning_sessions",
    )

    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    state: Mapped[Optional["SessionState"]] = relationship(
        "SessionState",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )

    # Indexes and constraints
    __table_args__ = (
        Index("ix_learning_sessions_user_status", "user_id", "status"),
        Index("ix_learning_sessions_topic_status", "topic_id", "status"),
        Index("ix_learning_sessions_session_created", "session_id", "created_at"),
        CheckConstraint(
            "interaction_count >= 0",
            name="ck_interaction_count_positive",
        ),
        CheckConstraint(
            "duration_seconds IS NULL OR duration_seconds >= 0",
            name="ck_duration_positive",
        ),
    )

    def __repr__(self) -> str:
        return f"<LearningSession(id={self.id}, session_id='{self.session_id}', status={self.status})>"


class Message(Base):
    """
    Conversation message within a learning session.

    Stores all messages with Bedrock tracking for cost analysis.

    Attributes:
        id: Primary key
        session_id: Parent session ID
        role: Message role (user, assistant, system)
        content: Message content
        bedrock_invocation_id: Bedrock request ID
        model_used: Which Bedrock model was used
        input_tokens: Input token count
        output_tokens: Output token count
        latency_ms: Response latency in milliseconds
        metadata_: Message metadata (sources, citations, tool calls)
        created_at: Message timestamp
    """

    __tablename__ = "messages"

    # Primary key
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)

    # Session relationship
    session_id: Mapped[int] = Column(
        Integer,
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent session ID",
    )

    # Message data
    role: Mapped[MessageRole] = Column(
        SQLEnum(MessageRole),
        nullable=False,
        comment="Message role",
    )

    content: Mapped[str] = Column(
        Text,
        nullable=False,
        comment="Message content",
    )

    # Bedrock tracking
    bedrock_invocation_id: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        index=True,
        comment="Bedrock request ID",
    )

    model_used: Mapped[Optional[str]] = Column(
        String(100),
        nullable=True,
        comment="Which Bedrock model was used",
    )

    input_tokens: Mapped[Optional[int]] = Column(
        Integer,
        nullable=True,
        comment="Input token count",
    )

    output_tokens: Mapped[Optional[int]] = Column(
        Integer,
        nullable=True,
        comment="Output token count",
    )

    latency_ms: Mapped[Optional[int]] = Column(
        Integer,
        nullable=True,
        comment="Response latency in milliseconds",
    )

    # Metadata
    metadata_: Mapped[dict] = Column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        comment="Message metadata (sources, citations, tool calls)",
    )

    # Timestamp
    created_at: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Message timestamp",
    )

    # Relationships
    session: Mapped["LearningSession"] = relationship(
        "LearningSession",
        back_populates="messages",
    )

    # Indexes and constraints
    __table_args__ = (
        Index("ix_messages_session_created", "session_id", "created_at"),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_input_tokens_positive",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_output_tokens_positive",
        ),
        CheckConstraint(
            "latency_ms IS NULL OR latency_ms >= 0",
            name="ck_latency_positive",
        ),
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role}, session_id={self.session_id})>"


class SessionState(Base):
    """
    Agent state persistence per session.

    Stores agent memory, context, and planning state. Updated after each
    interaction to enable session recovery.

    Attributes:
        id: Primary key
        session_id: Parent session ID (unique)
        state_data: Agent state JSON blob
        state_summary: Summarized state (every 5 turns)
        last_updated: Last update timestamp
    """

    __tablename__ = "session_states"

    # Primary key
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)

    # Session relationship
    session_id: Mapped[int] = Column(
        Integer,
        ForeignKey("learning_sessions.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        comment="Parent session ID",
    )

    # State data
    state_data: Mapped[dict] = Column(
        JSON,
        nullable=False,
        default=dict,
        comment="Agent state JSON blob (memory, context, planning)",
    )

    state_summary: Mapped[Optional[str]] = Column(
        Text,
        nullable=True,
        comment="Summarized state generated every 5 turns",
    )

    # Timestamp
    last_updated: Mapped[datetime] = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="Last update timestamp",
    )

    # Relationships
    session: Mapped["LearningSession"] = relationship(
        "LearningSession",
        back_populates="state",
    )

    def __repr__(self) -> str:
        return f"<SessionState(session_id={self.session_id})>"


# ============================================================================
# Agent Configuration Models
# ============================================================================


class AgentConfig(Base, TimestampMixin):
    """
    Agent configuration presets (admin-managed).

    Defines agent behavior, model settings, and system prompts.
    Users cannot create custom configs (admin-only).

    Attributes:
        id: Primary key
        name: Configuration name (unique)
        description: Configuration description
        is_default: True if this is the default config
        is_active: Configuration status
        max_turns: Maximum conversation turns
        require_teacher_check_every: Turns between teacher reminders
        encourage_teacher_interaction: Enable teacher interaction prompts
        show_sources: Show source citations to students
        bedrock_model_id: Bedrock model ID to use
        bedrock_kb_id: Bedrock Knowledge Base ID
        temperature: Model temperature
        max_tokens: Maximum output tokens
        top_p: Top-p sampling parameter
        system_prompt: Base system instructions
        settings: Additional settings (JSON)
    """

    __tablename__ = "agent_configs"

    # Primary key
    id: Mapped[int] = Column(Integer, primary_key=True, autoincrement=True)

    # Configuration identification
    name: Mapped[str] = Column(
        String(255),
        unique=True,
        nullable=False,
        comment="Configuration name",
    )

    description: Mapped[Optional[str]] = Column(
        Text,
        nullable=True,
        comment="Configuration description",
    )

    # Status
    is_default: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=False,
        comment="True if this is the default configuration",
    )

    is_active: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Configuration status",
    )

    # Agent behavior
    max_turns: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=20,
        comment="Maximum conversation turns",
    )

    require_teacher_check_every: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=5,
        comment="Turns between teacher interaction reminders",
    )

    encourage_teacher_interaction: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Enable teacher interaction prompts",
    )

    show_sources: Mapped[bool] = Column(
        Boolean,
        nullable=False,
        default=True,
        comment="Show source citations to students",
    )

    # Model configuration
    bedrock_model_id: Mapped[str] = Column(
        String(255),
        nullable=False,
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        comment="Bedrock model ID to use",
    )

    bedrock_kb_id: Mapped[Optional[str]] = Column(
        String(255),
        nullable=True,
        comment="Bedrock Knowledge Base ID",
    )

    temperature: Mapped[float] = Column(
        Float,
        nullable=False,
        default=0.7,
        comment="Model temperature",
    )

    max_tokens: Mapped[int] = Column(
        Integer,
        nullable=False,
        default=2048,
        comment="Maximum output tokens",
    )

    top_p: Mapped[float] = Column(
        Float,
        nullable=False,
        default=0.9,
        comment="Top-p sampling parameter",
    )

    # System prompt
    system_prompt: Mapped[Optional[str]] = Column(
        Text,
        nullable=True,
        comment="Base system instructions",
    )

    # Additional settings
    settings: Mapped[dict] = Column(
        JSON,
        nullable=False,
        default=dict,
        comment="Additional configuration settings",
    )

    # Relationships
    learning_sessions: Mapped[List["LearningSession"]] = relationship(
        "LearningSession",
        back_populates="agent_config",
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "max_turns > 0",
            name="ck_max_turns_positive",
        ),
        CheckConstraint(
            "require_teacher_check_every > 0",
            name="ck_teacher_check_positive",
        ),
        CheckConstraint(
            "temperature >= 0.0 AND temperature <= 2.0",
            name="ck_temperature_range",
        ),
        CheckConstraint(
            "max_tokens > 0",
            name="ck_max_tokens_positive",
        ),
        CheckConstraint(
            "top_p >= 0.0 AND top_p <= 1.0",
            name="ck_top_p_range",
        ),
    )

    def __repr__(self) -> str:
        return f"<AgentConfig(id={self.id}, name='{self.name}', is_default={self.is_default})>"