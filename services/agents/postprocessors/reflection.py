"""
Reflection refiner.

Uses self-critique to improve response quality.
"""

import logging

from services.agents.base import AgentContext, PipelineStep
from services.agents.providers import BaseLLMProvider, LLMMessage

logger = logging.getLogger(__name__)


class ReflectionRefiner(PipelineStep):
    """
    Refine responses using reflection pattern.

    This postprocessor:
    1. Checks if response needs refinement (flagged by previous steps)
    2. Uses LLM to critique and improve the response
    3. Replaces generated response with refined version
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        super().__init__(name="ReflectionRefiner")
        self.llm_provider = llm_provider

    async def process(self, context: AgentContext) -> AgentContext:
        """Refine response if needed."""

        # Check if reflection is needed
        if not context.processing_data.get("needs_reflection"):
            return context

        response = context.processing_data.get("generated_response")
        if not response:
            return context

        # Build refinement prompt
        issue = self._identify_issue(context)

        logger.info(f"Refining response due to: {issue}")

        # Refine response
        refined_response = await self._refine_response(
            original_response=response,
            user_message=context.user_message or "",
            retrieved_context=context.processing_data.get("retrieved_context"),
            issue=issue,
        )

        # Replace response (immutable)
        new_context = context.update_processing_data(
            {
                "original_response": response,
                "generated_response": refined_response,
                "was_refined": True,
            }
        )

        logger.info(f"Response refined: {response[:50]}... → {refined_response[:50]}...")

        return new_context

    def _identify_issue(self, context: AgentContext) -> str:
        """Identify what needs to be fixed."""
        issues = []

        if not context.processing_data.get("is_faithful"):
            reasoning = context.processing_data.get("faithfulness_reasoning", "unknown reason")
            issues.append(f"Response not faithful to context: {reasoning}")

        if context.processing_data.get("tone_issue"):
            issues.append("Tone needs adjustment")

        if context.processing_data.get("clarity_issue"):
            issues.append("Response lacks clarity")

        return "; ".join(issues) if issues else "General quality improvement needed"

    async def _refine_response(
        self,
        original_response: str,
        user_message: str,
        retrieved_context: str | None,
        issue: str,
    ) -> str:
        """
        Refine the response to fix identified issues.

        Returns:
            Refined response
        """

        context_section = ""
        if retrieved_context:
            context_section = f"""
Context from knowledge base:
{retrieved_context}
"""

        prompt = f"""The AI gave this response to the user's message:

User: {user_message}
AI: {original_response}

However, there's an issue: {issue}
{context_section}

Please rewrite the response to fix the issue while:
- Being helpful and natural
- Maintaining conversational tone
- Only using information from the context if provided
- Staying on topic

Provide ONLY the rewritten response, no explanations."""

        try:
            result = await self.llm_provider.generate(
                messages=[LLMMessage(role="user", content=prompt)],
                temperature=0.5,
            )
            return result.content

        except Exception as e:
            logger.error(f"Response refinement failed: {e}", exc_info=True)
            # Return original on error
            return original_response
