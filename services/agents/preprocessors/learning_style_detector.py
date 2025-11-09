"""
Learning Style Detection Preprocessor.

Analyzes student conversation patterns to detect their preferred learning style:
- Visual (diagrams, charts, spatial understanding)
- Auditory (discussions, verbal explanations, sounds)
- Kinesthetic (hands-on, physical examples, movement)
- Reading/Writing (textual information, note-taking, lists)

Uses LLM to analyze conversation history and infer learning style.
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from services.agents.base import AgentContext, PipelineStep


class LearningStyleAnalysis(BaseModel):
    """Structured output for learning style analysis."""

    detected_style: str  # "Visual", "Auditory", "Kinesthetic", "Reading/Writing", or "Unknown"
    confidence: float  # 0.0 to 1.0
    indicators: list[str]  # Evidence from conversation
    turn_detected: int  # Which turn the style was detected


class LearningStyleDetector(PipelineStep):
    """
    Detect student learning style from conversation patterns.

    This preprocessor:
    1. Analyzes conversation history after N turns (default: 2-3 interactions)
    2. Uses LLM to identify learning style indicators
    3. Stores detected style in persisted_state
    4. Adds learning_style to processing_data for downstream use

    Detection happens once and is cached in persisted_state.
    """

    STYLE_DETECTION_PROMPT = """You are an expert educational psychologist analyzing student learning patterns.

Based on the conversation history below, identify the student's preferred learning style.

Learning Styles:
- **Visual**: Prefers diagrams, charts, images, spatial understanding, visual metaphors
  Indicators: asks for diagrams, mentions "I see", talks about visualizing concepts, wants pictures

- **Auditory**: Prefers discussions, verbal explanations, listening, talking through problems
  Indicators: asks to explain verbally, mentions "sounds like", prefers discussions, talks aloud

- **Kinesthetic**: Prefers hands-on activities, physical examples, real-world applications, doing
  Indicators: asks for practical examples, wants to try it, mentions doing/building, physical metaphors

- **Reading/Writing**: Prefers textual information, note-taking, lists, written explanations
  Indicators: asks for written steps, takes notes, prefers text, makes lists, asks for reading materials

Analyze the conversation and respond with:
1. The most likely learning style (Visual, Auditory, Kinesthetic, or Reading/Writing)
2. Confidence level (0.0 to 1.0)
3. Specific indicators you observed

If there isn't enough information yet (less than 2-3 exchanges), respond with "Unknown" and confidence 0.0.

Conversation History:
{conversation}

Respond in JSON format:
{{
    "detected_style": "Visual|Auditory|Kinesthetic|Reading/Writing|Unknown",
    "confidence": 0.8,
    "indicators": ["specific evidence from conversation"]
}}
"""

    def __init__(
        self,
        llm: BaseChatModel,
        min_turns_for_detection: int = 2,
        confidence_threshold: float = 0.6,
        name: str = "LearningStyleDetector",
    ):
        """
        Initialize learning style detector.

        Args:
            llm: Language model for analysis
            min_turns_for_detection: Minimum conversation turns before attempting detection
            confidence_threshold: Minimum confidence to save detection result
            name: Step name for tracing
        """
        super().__init__(name=name)
        self.llm = llm
        self.min_turns_for_detection = min_turns_for_detection
        self.confidence_threshold = confidence_threshold

    async def process(self, context: AgentContext) -> AgentContext:
        """
        Detect learning style if not already detected.

        Args:
            context: Current agent context

        Returns:
            Updated context with learning_style in processing_data and persisted_state
        """
        # Skip if already detected
        existing_style = context.persisted_state.get("learning_style")
        if existing_style and existing_style != "Unknown":
            # Add to processing_data for current request
            return context.add_processing_data("learning_style", existing_style)

        # Check if we have enough conversation turns
        user_messages = context.conversation_history.get_user_messages()
        if len(user_messages) < self.min_turns_for_detection:
            # Not enough data yet
            return context.add_processing_data("learning_style", "Unknown")

        # Analyze conversation to detect style
        analysis = await self._analyze_learning_style(context)

        # Update context if confidence is high enough
        if analysis.confidence >= self.confidence_threshold:
            # Save to persisted_state (will be saved to DB)
            context = context.update_persisted_state({
                "learning_style": analysis.detected_style,
                "learning_style_confidence": analysis.confidence,
                "learning_style_indicators": analysis.indicators,
                "learning_style_detected_at_turn": len(user_messages),
            })

            # Add to processing_data for immediate use
            context = context.add_processing_data("learning_style", analysis.detected_style)
            context = context.add_processing_data("learning_style_confidence", analysis.confidence)
        else:
            # Not confident enough yet
            context = context.add_processing_data("learning_style", "Unknown")

        return context

    async def _analyze_learning_style(self, context: AgentContext) -> LearningStyleAnalysis:
        """
        Use LLM to analyze conversation and detect learning style.

        Args:
            context: Current agent context

        Returns:
            Learning style analysis
        """
        # Format conversation history for analysis
        conversation_text = self._format_conversation(context)

        # Create prompt
        prompt = self.STYLE_DETECTION_PROMPT.format(conversation=conversation_text)

        try:
            # Call LLM
            messages = [
                SystemMessage(content="You are an educational psychologist."),
                HumanMessage(content=prompt),
            ]

            response = await self.llm.ainvoke(messages)

            # Parse response (assuming JSON response)
            import json
            result = json.loads(response.content)

            return LearningStyleAnalysis(
                detected_style=result.get("detected_style", "Unknown"),
                confidence=float(result.get("confidence", 0.0)),
                indicators=result.get("indicators", []),
                turn_detected=len(context.conversation_history.get_user_messages()),
            )

        except Exception as e:
            # Fallback if LLM call fails
            return LearningStyleAnalysis(
                detected_style="Unknown",
                confidence=0.0,
                indicators=[f"Detection failed: {str(e)}"],
                turn_detected=0,
            )

    def _format_conversation(self, context: AgentContext) -> str:
        """
        Format conversation history for analysis.

        Args:
            context: Agent context

        Returns:
            Formatted conversation text
        """
        lines = []
        for msg in context.conversation_history.messages:
            role = "Student" if msg.role.value == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")

        return "\n".join(lines)