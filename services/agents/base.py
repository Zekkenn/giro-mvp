"""
Core abstractions for AI agents.

This module defines the base interfaces and types that all AI agents must implement.
Following clean architecture principles:
- High-level policy (agents) doesn't depend on low-level details (providers)
- Dependencies point inward
- Easy to test and extend

Now integrated with database models and supports LangChain Expression Language (LCEL).
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import BaseModel, Field

# Import database enums (single source of truth)
from database.enums import MessageRole, ExitReason

# ============================================================================
# Data Models
# ============================================================================


class Message(BaseModel):
    """
    A message in the conversation (lightweight).

    This is used during pipeline execution. For persistence, use database.models.Message.
    """

    role: MessageRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationHistory(BaseModel):
    """Complete conversation history."""

    messages: list[Message] = Field(default_factory=list)

    def add_user_message(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add user message to history."""
        self.messages.append(
            Message(
                role=MessageRole.USER,
                content=content,
                metadata=metadata or {},
            )
        )

    def add_assistant_message(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add assistant message to history."""
        self.messages.append(
            Message(
                role=MessageRole.ASSISTANT,
                content=content,
                metadata=metadata or {},
            )
        )

    def get_user_messages(self) -> list[Message]:
        """Get only user messages."""
        return [msg for msg in self.messages if msg.role == MessageRole.USER]

    def get_last_user_message(self) -> Message | None:
        """Get the last user message."""
        user_msgs = self.get_user_messages()
        return user_msgs[-1] if user_msgs else None


class AgentContext(BaseModel):
    """
    Immutable context passed through the pipeline.

    This is the central data structure that flows through all pipeline steps.
    Each step receives a context and returns a NEW context (immutable pattern).

    IMPORTANT: Never mutate context directly. Use helper methods to create new instances.

    Fields aligned with new database schema:
    - user_id: maps to users.id
    - session_id: maps to learning_sessions.session_id (external identifier)
    - db_session_id: maps to learning_sessions.id (internal DB ID)
    """

    # Identity (updated to match new schema)
    user_id: int  # User ID from users table
    session_id: str  # External session identifier
    db_session_id: int | None = None  # Internal database ID (loaded by ContextBuilder)

    # Conversation
    conversation_history: ConversationHistory = Field(default_factory=ConversationHistory)
    user_message: str | None = None

    # Persisted state (saved to session_states table)
    persisted_state: dict[str, Any] = Field(default_factory=dict)

    # Processing data (ephemeral, used during pipeline execution)
    processing_data: dict[str, Any] = Field(default_factory=dict)

    # Control flow
    should_exit: bool = False
    exit_reason: ExitReason | None = None

    class Config:
        arbitrary_types_allowed = True
        frozen = False  # Allow model_copy to work

    def set_exit(self, reason: ExitReason) -> "AgentContext":
        """
        Signal that conversation should exit.

        Returns new context with exit set (immutable pattern).
        """
        return self.model_copy(
            update={
                "should_exit": True,
                "exit_reason": reason,
            }
        )

    def add_processing_data(self, key: str, value: Any) -> "AgentContext":
        """
        Add data to processing_data.

        Returns new context (immutable pattern).
        """
        return self.model_copy(
            update={
                "processing_data": {
                    **self.processing_data,
                    key: value,
                }
            }
        )

    def update_processing_data(self, updates: dict[str, Any]) -> "AgentContext":
        """
        Update multiple processing_data keys.

        Returns new context (immutable pattern).
        """
        return self.model_copy(
            update={
                "processing_data": {
                    **self.processing_data,
                    **updates,
                }
            }
        )

    def add_persisted_state(self, key: str, value: Any) -> "AgentContext":
        """
        Add data to persisted_state (will be saved to DB).

        Returns new context (immutable pattern).
        """
        return self.model_copy(
            update={
                "persisted_state": {
                    **self.persisted_state,
                    key: value,
                }
            }
        )

    def update_persisted_state(self, updates: dict[str, Any]) -> "AgentContext":
        """
        Update multiple persisted_state keys.

        Returns new context (immutable pattern).
        """
        return self.model_copy(
            update={
                "persisted_state": {
                    **self.persisted_state,
                    **updates,
                }
            }
        )


class AgentResponse(BaseModel):
    """Final response from agent."""

    content: str | None = None
    should_send: bool = True
    exit_path: str | None = None  # Maps to exit_reason for session tracking
    persisted_state_updates: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Pipeline Components
# ============================================================================


class PipelineStep(ABC):
    """
    Base class for pipeline steps.

    Pipeline steps are the building blocks of agent behavior. They can:
    - Analyze context (preprocessors)
    - Generate responses (providers)
    - Refine output (postprocessors)

    All steps must implement as_runnable() to work with LCEL chains.
    """

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def process(self, context: AgentContext) -> AgentContext:
        """
        Process the context and return updated context.

        Args:
            context: Current agent context

        Returns:
            Updated context
        """
        pass

    async def __call__(self, context: AgentContext) -> AgentContext:
        """Allow step to be called directly."""
        return await self.process(context)

    def as_runnable(self) -> Runnable:
        """
        Convert this step to an LCEL Runnable with meaningful name for tracing.

        This allows steps to be used in LangChain Expression Language chains.
        The Runnable accepts dict (serialized AgentContext) and returns dict.

        The step name is used for LangSmith/LangChain tracing, making it easy
        to debug and visualize the pipeline execution.

        Returns:
            RunnableLambda that wraps this step's process method
        """

        async def _wrapped_process(context_dict: dict) -> dict:
            """Wrapper that handles dict ↔ AgentContext conversion."""
            # Deserialize
            context = AgentContext(**context_dict)

            # Process
            result_context = await self.process(context)

            # Serialize
            return result_context.model_dump()

        # Use step name for LangSmith tracing visibility
        return RunnableLambda(_wrapped_process, name=self.name)


# ============================================================================
# Agent Base Class
# ============================================================================


class BaseAgentConfig(BaseModel):
    """Base configuration for agents."""

    name: str


TConfig = TypeVar("TConfig", bound=BaseAgentConfig)


def _build_response(context: AgentContext) -> AgentResponse:
    """
    Build agent response from pipeline context.

    Default implementation:
    - If should_exit, return exit response
    - Otherwise, return generated response from processing_data

    Override for custom response building logic.
    """
    if context.should_exit:
        return AgentResponse(
            content=None,
            should_send=False,
            exit_path=context.exit_reason.value if context.exit_reason else None,
            persisted_state_updates=context.persisted_state,
        )

    # Get generated response from pipeline
    generated_response = context.processing_data.get("final_response")

    return AgentResponse(
        content=generated_response,
        should_send=generated_response is not None,
        persisted_state_updates=context.persisted_state,
        metadata=context.processing_data,
    )


class BaseAgent(ABC, Generic[TConfig]):
    """
    Base class for all AI agents using LCEL.

    Agents use LangChain Expression Language (LCEL) for orchestration,
    providing powerful routing, branching, and retry capabilities.

    Responsibilities:
    - Define agent configuration
    - Build LCEL chain
    - Convert chain output to agent response
    """

    def __init__(self, config: TConfig):
        self.config = config
        self.chain = self._build_chain()

    @abstractmethod
    def _build_chain(self) -> Runnable:
        """
        Build the agent's LCEL chain.

        This is where you compose preprocessors, providers, and postprocessors
        using LangChain Expression Language.

        Example:
            def _build_chain(self):
                sentiment = SentimentAnalyzer(self.llm).as_runnable()
                goal = GoalChecker(self.llm, goal=self.config.objective).as_runnable()
                generate = RAGProvider(self.llm).as_runnable()

                # Simple sequence
                return sentiment | goal | generate

                # Or with branching
                from langchain_core.runnables import RunnableBranch

                sentiment_branch = RunnableBranch(
                    (lambda ctx: ctx["should_exit"], exit_handler),
                    goal  # default
                )

                return sentiment | sentiment_branch | generate

        Returns:
            Runnable chain
        """
        pass

    async def initialize_conversation(self, context: AgentContext) -> AgentResponse:
        """
        Initialize a new conversation.

        Default implementation runs the chain without a user message.
        Override if you need custom initialization logic.
        """
        result_dict = await self.chain.ainvoke(context.model_dump())
        result_context = AgentContext(**result_dict)
        return _build_response(result_context)

    async def process_message(self, context: AgentContext) -> AgentResponse:
        """
        Process a user message.

        Runs the chain and converts result to response.
        """
        result_dict = await self.chain.ainvoke(context.model_dump())
        result_context = AgentContext(**result_dict)
        return _build_response(result_context)

    def validate_config(self) -> None:
        """
        Validate agent configuration.

        Override to add custom validation logic.
        """
        pass