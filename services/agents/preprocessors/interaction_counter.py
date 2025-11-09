"""
Interaction counter preprocessor.

Tracks the number of interactions and triggers exit if max is reached.
"""

import logging

from services.agents.base import AgentContext, ExitReason, PipelineStep

logger = logging.getLogger(__name__)


class InteractionCounter(PipelineStep):
    """
    Track interaction count and enforce maximum.

    This preprocessor:
    1. Increments interaction count in state
    2. Checks if max interactions reached
    3. Triggers exit if limit exceeded
    """

    def __init__(self, max_interactions: int, state_key: str = "interaction_count"):
        super().__init__(name="InteractionCounter")
        self.max_interactions = max_interactions
        self.state_key = state_key

    async def process(self, context: AgentContext) -> AgentContext:
        """Track and check interaction count."""

        # Get current count
        current_count = int(context.persisted_state.get(self.state_key, 0))

        # Increment if user sent a message
        if context.user_message:
            current_count += 1

        logger.info(f"Interaction {current_count}/{self.max_interactions}")

        # Update context with new count (immutable)
        new_context = context.add_persisted_state(self.state_key, current_count)
        new_context = new_context.add_processing_data("interaction_count", current_count)

        # Check if max reached
        if current_count >= self.max_interactions:
            logger.info("Max interactions reached, triggering exit")
            new_context = new_context.set_exit(ExitReason.MAX_INTERACTIONS)

        return new_context
