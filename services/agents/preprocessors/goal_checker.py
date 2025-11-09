"""
Goal achievement checker.

Checks if the conversation objective has been achieved.
"""

import logging

from services.agents.base import AgentContext, ConversationHistory, ExitReason, PipelineStep
from services.agents.providers import BaseLLMProvider, LLMMessage, StructuredOutputExtractor

logger = logging.getLogger(__name__)


class GoalChecker(PipelineStep):
    """
    Check if conversation goal has been achieved.

    This preprocessor:
    1. Evaluates if the objective has been met based on conversation history
    2. Triggers exit if goal is achieved
    3. Stores goal status in pipeline_data
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        objective: str,
        exit_on_achieved: bool = True,
    ):
        super().__init__(name="GoalChecker")
        self.extractor = StructuredOutputExtractor(llm_provider)
        self.objective = objective
        self.exit_on_achieved = exit_on_achieved

    async def process(self, context: AgentContext) -> AgentContext:
        """Check if goal is achieved."""

        # Need at least some conversation to check
        if len(context.conversation_history.messages) < 2:
            return context.add_processing_data("goal_achieved", False)

        # Check goal achievement
        goal_result = await self._check_goal_achievement(
            objective=self.objective,
            conversation_history=context.conversation_history,
        )

        logger.info(
            f"Goal check: {'ACHIEVED' if goal_result['achieved'] else 'NOT ACHIEVED'} - {goal_result['reasoning']}"
        )

        # Store in processing data (immutable)
        new_context = context.update_processing_data(
            {
                "goal_achieved": goal_result["achieved"],
                "goal_reasoning": goal_result["reasoning"],
            }
        )

        # Trigger exit if achieved
        if self.exit_on_achieved and goal_result["achieved"]:
            logger.info("Goal achieved, triggering exit")
            new_context = new_context.set_exit(ExitReason.ACHIEVED_GOAL)

        return new_context

    async def _check_goal_achievement(
        self,
        objective: str,
        conversation_history: ConversationHistory,
    ) -> dict:
        """
        Check if objective has been achieved.

        Returns:
            {
                "achieved": bool,
                "reasoning": str,
                "confidence": float
            }
        """

        # Format conversation history
        conversation_text = "\n".join([f"{msg.role.value}: {msg.content}" for msg in conversation_history.messages])

        prompt = f"""Objective: {objective}

Conversation so far:
{conversation_text}

Has the objective been achieved in this conversation?

Consider:
- Has the user agreed to or completed the objective?
- Is there clear evidence the goal is met?
- Be conservative - only return true if you're confident

Respond in JSON format:
{{
    "achieved": true/false,
    "reasoning": "explanation",
    "confidence": 0.0 to 1.0
}}
"""

        schema = {
            "type": "object",
            "properties": {
                "achieved": {
                    "type": "boolean",
                    "description": "Whether the objective has been achieved",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the decision",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the decision",
                },
            },
            "required": ["achieved", "reasoning", "confidence"],
        }

        try:
            result = await self.extractor.extract(
                messages=[LLMMessage(role="user", content=prompt)],
                schema=schema,
                temperature=0,
            )
            return result

        except Exception as e:
            logger.error(f"Goal checking failed: {e}", exc_info=True)
            return {
                "achieved": False,
                "reasoning": f"Error during check: {str(e)}",
                "confidence": 0.0,
            }
