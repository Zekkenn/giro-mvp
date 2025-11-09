"""
Context builder preprocessor.

Loads conversation history and state from database into context.
"""

import logging
from sqlalchemy.orm import Session

from services.agents.base import (
    AgentContext,
    ConversationHistory,
    MessageRole,
    PipelineStep,
    Message as AgentMessage,
)
from database.models import LearningSession, Message, SessionState

logger = logging.getLogger(__name__)


class ContextBuilder(PipelineStep):
    """
    Build conversation context from database.

    This preprocessor:
    1. Loads conversation history from DB
    2. Loads session state/variables from DB
    3. Populates the AgentContext with this data
    """

    def __init__(self, db_session: Session):
        super().__init__(name="ContextBuilder")
        self.db_session = db_session

    async def process(self, context: AgentContext) -> AgentContext:
        """Load context from database."""
        try:
            # Skip if context already loaded or no db_session_id
            if not context.db_session_id:
                logger.warning("No db_session_id in context, skipping context loading")
                return context

            # Load learning session
            learning_session = (
                self.db_session.query(LearningSession)
                .filter_by(id=context.db_session_id)
                .first()
            )

            if not learning_session:
                logger.warning(f"Learning session not found: {context.db_session_id}")
                return context

            # Load conversation history from messages table
            db_messages = (
                self.db_session.query(Message)
                .filter_by(session_id=learning_session.id)
                .order_by(Message.created_at)
                .all()
            )

            # Convert to our message format
            history = ConversationHistory()
            for db_msg in db_messages:
                if not db_msg.content:
                    continue

                history.messages.append(
                    AgentMessage(
                        role=db_msg.role,  # Already a MessageRole enum
                        content=db_msg.content,
                        metadata={
                            "message_id": db_msg.id,
                            "created_at": db_msg.created_at.isoformat() if db_msg.created_at else None,
                            "bedrock_invocation_id": db_msg.bedrock_invocation_id,
                        },
                    )
                )

            # Load session state from session_states table
            session_state = (
                self.db_session.query(SessionState)
                .filter_by(session_id=learning_session.id)
                .first()
            )

            state = session_state.state_data if session_state else {}

            logger.info(
                f"Loaded context for session {context.session_id}: "
                f"{len(history.messages)} messages, {len(state)} state keys"
            )

            # Mark context as loaded
            new_context = context.model_copy(
                update={
                    "conversation_history": history,
                    "persisted_state": state,
                }
            )

            return new_context.add_processing_data("conversation_loaded", True)

        except Exception as e:
            logger.error(f"Failed to load context from DB: {e}", exc_info=True)
            # Don't raise - return context as-is so pipeline can continue
            return context
