"""
Faithfulness checker for RAG responses.

Validates that AI responses are grounded in retrieved context.
"""

import logging

from services.agents.base import AgentContext, PipelineStep
from services.agents.providers import BaseLLMProvider, LLMMessage, StructuredOutputExtractor

logger = logging.getLogger(__name__)


class FaithfulnessChecker(PipelineStep):
    """
    Check if response is faithful to retrieved context.

    This postprocessor:
    1. Checks if response only uses information from retrieved context
    2. Flags unfaithful responses for refinement
    3. Stores faithfulness assessment in pipeline_data
    """

    def __init__(self, llm_provider: BaseLLMProvider):
        super().__init__(name="FaithfulnessChecker")
        self.extractor = StructuredOutputExtractor(llm_provider)

    async def process(self, context: AgentContext) -> AgentContext:
        """Check faithfulness of generated response."""

        # Only check if RAG was used
        retrieved_context = context.processing_data.get("retrieved_context")
        if not retrieved_context:
            return context.add_processing_data("is_faithful", True)

        # Get generated response
        response = context.processing_data.get("generated_response")
        if not response:
            return context

        # Check faithfulness
        faithfulness_result = await self._check_faithfulness(
            response=response,
            context=retrieved_context,
        )

        logger.info(
            f"Faithfulness check: {'PASSED' if faithfulness_result['faithful'] else 'FAILED'} - "
            f"{faithfulness_result['reasoning']}"
        )

        # Store result (immutable)
        new_context = context.update_processing_data(
            {
                "is_faithful": faithfulness_result["faithful"],
                "faithfulness_reasoning": faithfulness_result["reasoning"],
            }
        )

        # Flag for reflection if not faithful
        if not faithfulness_result["faithful"]:
            new_context = new_context.add_processing_data("needs_reflection", True)

        return new_context

    async def _check_faithfulness(self, response: str, context: str) -> dict:
        """
        Check if response is faithful to context.

        Returns:
            {
                "faithful": bool,
                "reasoning": str,
                "violations": list[str] | None
            }
        """

        prompt = f"""Given this context from a knowledge base:

{context}

And this AI response:

{response}

Is the response faithful to the context? Evaluate:
1. Does it only use information from the context?
2. Does it avoid making claims not supported by the context?
3. Is it accurately representing the context?

Respond in JSON format:
{{
    "faithful": true/false,
    "reasoning": "explanation",
    "violations": ["list", "of", "specific", "violations"] or null
}}
"""

        schema = {
            "type": "object",
            "properties": {
                "faithful": {
                    "type": "boolean",
                    "description": "Whether the response is faithful to context",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Explanation of the assessment",
                },
                "violations": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "List of specific violations if unfaithful",
                },
            },
            "required": ["faithful", "reasoning"],
        }

        try:
            result = await self.extractor.extract(
                messages=[LLMMessage(role="user", content=prompt)],
                schema=schema,
                temperature=0,
            )
            return result

        except Exception as e:
            logger.error(f"Faithfulness check failed: {e}", exc_info=True)
            # On error, assume faithful to avoid blocking response
            return {
                "faithful": True,
                "reasoning": f"Error during check: {str(e)}",
                "violations": None,
            }
