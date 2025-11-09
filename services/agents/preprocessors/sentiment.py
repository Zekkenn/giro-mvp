"""
Sentiment analysis preprocessor.

Analyzes user messages for negative sentiment and triggers exit if needed.
"""

import logging

from services.agents.base import AgentContext, ExitReason, PipelineStep
from services.agents.providers import BaseLLMProvider, LLMMessage, StructuredOutputExtractor

logger = logging.getLogger(__name__)


class SentimentAnalyzer(PipelineStep):
    """
    Analyze user sentiment and exit on negative sentiment.

    This preprocessor:
    1. Analyzes the sentiment of the latest user message
    2. If negative sentiment is detected above threshold, triggers exit
    3. Stores sentiment data in pipeline_data for later use
    """

    def __init__(
        self,
        llm_provider: BaseLLMProvider,
        threshold: float = 0.7,
        exit_on_negative: bool = True,
    ):
        super().__init__(name="SentimentAnalyzer")
        self.extractor = StructuredOutputExtractor(llm_provider)
        self.threshold = threshold
        self.exit_on_negative = exit_on_negative

    async def process(self, context: AgentContext) -> AgentContext:
        """Analyze sentiment of user message."""

        # Skip if no user message
        if not context.user_message:
            return context

        # Analyze sentiment
        sentiment_result = await self._analyze_sentiment(
            user_message=context.user_message,
        )

        logger.info(
            f"Sentiment analysis: {sentiment_result['classification']} "
            f"(confidence: {sentiment_result['confidence']:.2f})"
        )

        # Store in processing data (immutable)
        new_context = context.add_processing_data("sentiment", sentiment_result)

        # Check for exit condition
        if (
            self.exit_on_negative
            and sentiment_result["classification"] == "negative"
            and sentiment_result["confidence"] >= self.threshold
        ):
            logger.info("Negative sentiment detected, triggering exit")
            # Use USER_EXIT as closest match (NEGATIVE_SENTIMENT not in enum)
            new_context = new_context.set_exit(ExitReason.USER_EXIT)

        return new_context

    async def _analyze_sentiment(self, user_message: str) -> dict:
        """
        Analyze sentiment using LLM.

        Returns:
            {
                "classification": "positive" | "neutral" | "negative",
                "confidence": float,
                "reasoning": str
            }
        """

        prompt = f"""Analyze the sentiment of this user message:

"{user_message}"

Classify the sentiment as positive, neutral, or negative.

Respond in JSON format:
{{
    "classification": "positive" | "neutral" | "negative",
    "confidence": 0.0 to 1.0,
    "reasoning": "brief explanation"
}}
"""

        schema = {
            "type": "object",
            "properties": {
                "classification": {
                    "type": "string",
                    "enum": ["positive", "neutral", "negative"],
                    "description": "Sentiment classification",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence score",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the classification",
                },
            },
            "required": ["classification", "confidence", "reasoning"],
        }

        try:
            result = await self.extractor.extract(
                messages=[LLMMessage(role="user", content=prompt)],
                schema=schema,
                temperature=0,
            )
            return result

        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}", exc_info=True)
            # Default to neutral on error
            return {
                "classification": "neutral",
                "confidence": 0.5,
                "reasoning": f"Error during analysis: {str(e)}",
            }
