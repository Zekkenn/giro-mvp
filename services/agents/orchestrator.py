"""
Agent Orchestrator for Giro Agent.

Manages agent lifecycle and integrates with the database.
Refactored to use new PostgreSQL schema with SQLAlchemy models.
"""

import logging
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from services.agents.base import AgentContext, AgentResponse, BaseAgent, ExitReason
from database.models import LearningSession, Message, SessionState, LearningProfile
from database.enums import SessionStatus, MessageRole as DBMessageRole

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Orchestrates AI agent execution with database persistence.

    Responsibilities:
    - Manage agent instance lifecycle
    - Initialize and process conversations
    - Persist session state and messages to database
    - Track learning metrics
    - Handle agent exits and session completion

    Updated to use new database schema (learning_sessions, messages, session_states).
    """

    def __init__(self, db: Session, agent: BaseAgent):
        """
        Initialize orchestrator.

        Args:
            db: SQLAlchemy database session
            agent: Agent instance to orchestrate
        """
        self.db = db
        self.agent = agent

    async def initialize_conversation(
        self,
        user_id: int,
        session_id: str,
        topic_id: Optional[int] = None,
        agent_config_id: Optional[int] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Initialize a new conversation session.

        Args:
            user_id: User ID from users table
            session_id: External session identifier
            topic_id: Optional topic ID
            agent_config_id: Optional agent configuration ID

        Returns:
            (message_to_send, exit_path)
        """
        logger.info(
            f"Initializing conversation for agent {self.agent.__class__.__name__} "
            f"(user={user_id}, session={session_id})"
        )

        # Create session record
        db_session = LearningSession(
            user_id=user_id,
            session_id=session_id,
            topic_id=topic_id,
            agent_config_id=agent_config_id,
            status=SessionStatus.ACTIVE,
            start_time=datetime.now(timezone.utc),
        )
        self.db.add(db_session)
        self.db.flush()  # Get the ID

        # Build context
        context = AgentContext(
            user_id=user_id,
            session_id=session_id,
            db_session_id=db_session.id,
        )

        # Initialize agent
        response = await self.agent.initialize_conversation(context)

        # Handle response
        return await self._handle_response(
            response=response,
            db_session=db_session,
        )

    async def process_message(
        self,
        user_id: int,
        session_id: str,
        user_message: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Process a user message in an existing session.

        Args:
            user_id: User ID
            session_id: Session identifier
            user_message: User's message

        Returns:
            (response_message, exit_path)
        """
        logger.info(
            f"Processing message for {self.agent.__class__.__name__}: {user_message[:50]}..."
        )

        # Load session
        db_session = (
            self.db.query(LearningSession)
            .filter_by(session_id=session_id, user_id=user_id)
            .first()
        )

        if not db_session:
            raise ValueError(f"Session not found: {session_id} for user {user_id}")

        # Build context (ContextBuilder preprocessor will load history/state)
        context = AgentContext(
            user_id=user_id,
            session_id=session_id,
            db_session_id=db_session.id,
            user_message=user_message,
        )

        # Process message through agent
        response = await self.agent.process_message(context)

        # Handle response
        return await self._handle_response(
            response=response,
            db_session=db_session,
            user_message=user_message,
        )

    async def _handle_response(
        self,
        response: AgentResponse,
        db_session: LearningSession,
        user_message: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Handle agent response.

        1. Save user message (if provided)
        2. Save assistant response
        3. Update session state
        4. Update session metrics
        5. Update learning profile
        6. Handle session exit if needed

        Args:
            response: Agent response
            db_session: Database session record
            user_message: Optional user message

        Returns:
            (response_content, exit_path)
        """

        # Save user message
        if user_message:
            user_msg = Message(
                session_id=db_session.id,
                role=DBMessageRole.USER,
                content=user_message,
                metadata_={},
            )
            self.db.add(user_msg)
            db_session.interaction_count += 1

        # Save assistant response
        if response.should_send and response.content:
            assistant_msg = Message(
                session_id=db_session.id,
                role=DBMessageRole.ASSISTANT,
                content=response.content,
                metadata_=response.metadata,
            )
            self.db.add(assistant_msg)
            logger.info(f"Saved assistant message: {response.content[:50]}...")

        # Update or create session state
        if response.persisted_state_updates:
            self._persist_state(
                db_session=db_session,
                state_updates=response.persisted_state_updates,
            )

        # Handle exit
        if response.exit_path:
            self._end_session(
                db_session=db_session,
                exit_reason=response.exit_path,
            )

        # Update learning profile metrics
        self._update_learning_profile(db_session)

        # Commit transaction
        self.db.commit()

        return response.content, response.exit_path

    def _persist_state(
        self,
        db_session: LearningSession,
        state_updates: dict,
    ) -> None:
        """
        Persist agent state to session_states table.

        Args:
            db_session: Learning session record
            state_updates: State updates to persist
        """
        if not state_updates:
            return

        # Check if state exists
        state = (
            self.db.query(SessionState)
            .filter_by(session_id=db_session.id)
            .first()
        )

        if state:
            # Update existing state
            state.state_data = {
                **state.state_data,
                **state_updates,
            }
            state.last_updated = datetime.now(timezone.utc)
        else:
            # Create new state
            state = SessionState(
                session_id=db_session.id,
                state_data=state_updates,
            )
            self.db.add(state)

        logger.info(f"Persisted {len(state_updates)} state keys")

    def _end_session(
        self,
        db_session: LearningSession,
        exit_reason: str,
    ) -> None:
        """
        Mark session as ended.

        Args:
            db_session: Learning session record
            exit_reason: Reason for exit
        """
        db_session.status = SessionStatus.ENDED
        db_session.end_time = datetime.now(timezone.utc)

        # Calculate duration
        if db_session.start_time:
            duration = db_session.end_time - db_session.start_time
            db_session.duration_seconds = int(duration.total_seconds())

        # Map exit_reason string to ExitReason enum
        try:
            db_session.exit_reason = ExitReason(exit_reason)
        except ValueError:
            logger.warning(f"Unknown exit reason: {exit_reason}")
            db_session.exit_reason = ExitReason.ERROR

        logger.info(f"Session ended: {exit_reason}")

    def _update_learning_profile(self, db_session: LearningSession) -> None:
        """
        Update learning profile metrics.

        Args:
            db_session: Learning session record
        """
        # Get or create learning profile
        profile = (
            self.db.query(LearningProfile)
            .filter_by(user_id=db_session.user_id)
            .first()
        )

        if not profile:
            profile = LearningProfile(user_id=db_session.user_id)
            self.db.add(profile)

        # Update metrics
        profile.total_interactions = LearningProfile.total_interactions + 1

        # Update topics studied
        if db_session.topic_id and db_session.topic_id not in profile.topics_studied:
            topics = profile.topics_studied or []
            topics.append(db_session.topic_id)
            profile.topics_studied = topics

        # Update session count if session is ending
        if db_session.status in [SessionStatus.COMPLETED, SessionStatus.ENDED]:
            profile.total_sessions = LearningProfile.total_sessions + 1


# Exit reason mapping for backward compatibility
EXIT_REASON_TO_EDGE_TYPE = {
    ExitReason.ACHIEVED_GOAL: "achieved_goal",
    ExitReason.MAX_INTERACTIONS: "max_turns",
    ExitReason.USER_EXIT: "user_exit",
    ExitReason.ERROR: "error",
    ExitReason.TIMEOUT: "timeout",
}