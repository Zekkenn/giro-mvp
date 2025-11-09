"""
Giro Agent - Educational AI Assistant.

A specialized agent for educational interactions that:
- Detects and adapts to student learning styles
- Retrieves knowledge from teacher-uploaded materials
- Encourages teacher interaction and collaboration
- Uses deep reasoning for teaching
"""

from typing import AsyncGenerator, Optional
import os

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI

from services.agents.base import (
    BaseAgent,
    BaseAgentConfig,
    AgentContext,
)
from services.agents.preprocessors.interaction_counter import InteractionCounter
from services.agents.preprocessors.learning_style_detector import LearningStyleDetector
from services.agents.preprocessors.sentiment import SentimentAnalyzer
from services.agents.providers.base import BaseLLMProvider, LLMMessage, LLMResponse
from services.agents.providers.deep_agent_provider import DeepAgentProvider
from services.agents.tools.bedrock_rag_tools import get_bedrock_rag_tools


class OpenAIProvider(BaseLLMProvider):
    """Simple OpenAI provider for preprocessors."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7, api_key: str | None = None):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using OpenAI."""
        langchain_messages = []
        for msg in messages:
            # Handle both string role and MessageRole enum
            role = msg.role.value if hasattr(msg.role, 'value') else msg.role
            langchain_messages.append({"role": role, "content": msg.content})

        response = await self.model.ainvoke(langchain_messages)
        return LLMResponse(content=response.content, metadata={})


class GiroAgentConfig(BaseAgentConfig):
    """Configuration for Giro Agent."""

    name: str = "GiroAgent"
    model_name: str = "gpt-5"
    temperature: float = 0.7
    max_interactions: int = 20
    encourage_teacher_interaction_every: int = 5
    learning_objectives: Optional[str] = None
    current_topic: Optional[str] = None


class GiroAgent(BaseAgent[GiroAgentConfig]):
    """
    Giro Agent for educational assistance.

    Pipeline:
    1. Sentiment Analysis → exit if negative
    2. Interaction Counter → track turns, encourage teacher check-ins
    3. Learning Style Detection → adapt teaching approach
    4. DeepAgent Provider → reasoning + RAG for teaching
    """

    def __init__(self, config: GiroAgentConfig):
        """Initialize Giro Agent."""
        # Initialize LLM first (before super() which calls _build_chain)
        self.llm = ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
        )

        # Now call super which will build the chain
        super().__init__(config)

    def _build_chain(self) -> Runnable:
        """
        Build the Giro Agent pipeline.

        Pipeline:
        1. Sentiment analysis
        2. Interaction counter
        3. Learning style detection
        4. Deep agent (with RAG tool)
        5. Teacher interaction formatter
        """
        # Create OpenAI provider for preprocessors
        llm_provider = OpenAIProvider(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
        )

        # Preprocessors
        sentiment = SentimentAnalyzer(llm_provider).as_runnable()
        interaction_counter = InteractionCounter(
            max_interactions=self.config.max_interactions
        ).as_runnable()
        learning_style_detector = LearningStyleDetector(
            llm=llm_provider,
            min_turns_for_detection=2,
        ).as_runnable()

        # Provider - DeepAgent with Bedrock RAG tools
        deep_agent = DeepAgentProvider(
            instructions=self._build_instructions(),
            tools=get_bedrock_rag_tools(),
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            name="DeepAgentProvider",
        ).as_runnable()

        # Postprocessor - Add teacher interaction prompts
        teacher_interaction_formatter = self._build_teacher_interaction_formatter()

        # Chain: sentiment → counter → style → agent → formatter
        return (
            sentiment
            | interaction_counter
            | learning_style_detector
            | deep_agent
            | teacher_interaction_formatter
        )

    def _build_instructions(self) -> str:
        """Build system instructions for the deep agent with neuropsych awareness."""
        base_instructions = """You are Giro Agent, a neuropsychologically-informed AI tutor that adapts teaching to each student's cognitive profile.

**Your Role:**
- Help students understand concepts through personalized explanations
- Adapt your teaching based on the student's neuropsychological profile
- Use curriculum materials from the activity knowledge base
- Encourage teacher interaction for deeper support

**CRITICAL: Available Tools**

You have THREE tools to use:

1. **retrieve_student_profile** - Query the student's neuropsychological assessment
   - Use this FIRST at the start of a session to understand the student
   - Query for: "cognitive strengths", "vulnerabilities", "executive function", "attention", "memory"
   - This tells you HOW to teach (e.g., leverage visual reasoning, scaffold rule acquisition)

2. **retrieve_activity_content** - Search the current worksheet/activity materials
   - Use this to find definitions, examples, exercises from the teacher's materials
   - This tells you WHAT to teach (curriculum-aligned content)

3. **get_adaptation_context** - Get a quick summary of adaptation strategies
   - Use this for a holistic view of how to approach teaching this student

**Neuropsychologically-Informed Teaching Approach:**

1. **Leverage Strengths**: If the profile shows visual reasoning strength, use diagrams and spatial metaphors. If sustained attention is strong, provide focused deep-dive explanations.

2. **Scaffold Vulnerabilities**: If rule acquisition is difficult, provide explicit step-by-step rules rather than expecting pattern inference. If verbal working memory is limited, use shorter sentences and visual supports.

3. **Executive Function Awareness**: If the student shows tendency for unstructured exploration when tasks are ambiguous, provide clear structure and explicit goals upfront.

4. **Adaptive Examples**: Match examples to the student's interests (e.g., logic/programming if that's noted in their profile).

**Teaching Flow:**
1. At session start: Use `get_adaptation_context` or `retrieve_student_profile` to understand the student
2. For content questions: Use `retrieve_activity_content` to find curriculum-aligned information
3. Throughout: Adapt your explanations based on the profile insights
4. Regularly: Encourage discussing complex points with the teacher

**Important:**
- Frame yourself as a learning tool, not a teacher replacement
- When uncertain, recommend asking the teacher
- Break down complex concepts into manageable steps
- Check for understanding frequently
"""

        # Add learning objectives if configured
        if self.config.learning_objectives:
            base_instructions += f"\n\n**Learning Objectives for {self.config.current_topic}:**\n{self.config.learning_objectives}\n"

        # Add topic context if available
        if self.config.current_topic:
            base_instructions += f"\n\n**Current Topic:** {self.config.current_topic}\n"

        return base_instructions

    def _build_teacher_interaction_formatter(self) -> Runnable:
        """
        Build postprocessor that adds teacher interaction prompts.

        Adds gentle reminders to engage with the teacher at regular intervals.
        """

        async def format_with_teacher_prompt(context_dict: dict) -> dict:
            context = AgentContext(**context_dict)

            # Check if we should add teacher interaction prompt
            interaction_count = context.processing_data.get("interaction_count", 0)
            should_prompt = (
                interaction_count > 0
                and interaction_count % self.config.encourage_teacher_interaction_every == 0
            )

            if should_prompt and not context.should_exit:
                # Append teacher interaction reminder to response
                current_response = context.processing_data.get("final_response", "")
                if current_response:
                    teacher_prompt = (
                        "\n\n💡 _This is a good point to discuss with your teacher! "
                        "They can provide additional insights and answer deeper questions._"
                    )
                    context = context.update_processing_data({
                        "final_response": current_response + teacher_prompt
                    })

            return context.model_dump()

        return RunnableLambda(format_with_teacher_prompt, name="TeacherInteractionFormatter")

    async def stream_response(
        self,
        context: AgentContext,
    ) -> AsyncGenerator[tuple[str, Optional[str], list[dict]], None]:
        """
        Stream response with learning style and sources.

        Yields: (partial_response, learning_style, sources)
        """
        # Run preprocessors first (sentiment, interaction counter, learning style)
        llm_provider = OpenAIProvider(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
        )

        # 1. Sentiment analysis
        sentiment = SentimentAnalyzer(llm_provider)
        context = await sentiment.process(context)

        # Exit early if sentiment is negative
        if context.should_exit:
            error_msg = context.processing_data.get("sentiment_message", "I'm here to help when you're ready.")
            yield (error_msg, None, [])
            return

        # 2. Interaction counter
        interaction_counter = InteractionCounter(max_interactions=self.config.max_interactions)
        context = await interaction_counter.process(context)

        # 3. Learning style detection
        learning_style_detector = LearningStyleDetector(llm=llm_provider, min_turns_for_detection=2)
        context = await learning_style_detector.process(context)

        learning_style = context.processing_data.get("learning_style")
        if learning_style == "Unknown":
            learning_style = None

        # 4. Stream from DeepAgent with Bedrock RAG tools
        deep_agent = DeepAgentProvider(
            instructions=self._build_instructions(),
            tools=get_bedrock_rag_tools(),
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            name="DeepAgentProvider",
        )

        final_context = None
        async for partial_text, is_final, updated_context in deep_agent.stream_process(context):
            if is_final:
                final_context = updated_context

            # Yield streaming tokens
            if partial_text:
                sources = self._extract_sources(updated_context)
                yield (partial_text, learning_style, sources)

        # 5. Apply teacher interaction formatter (postprocessing)
        if final_context:
            interaction_count = final_context.processing_data.get("interaction_count", 0)
            should_prompt = (
                interaction_count > 0
                and interaction_count % self.config.encourage_teacher_interaction_every == 0
            )

            if should_prompt and not final_context.should_exit:
                final_response = final_context.processing_data.get("final_response", "")
                if final_response:
                    teacher_prompt = (
                        "\n\n💡 _This is a good point to discuss with your teacher! "
                        "They can provide additional insights and answer deeper questions._"
                    )
                    final_response_with_prompt = final_response + teacher_prompt
                    sources = self._extract_sources(final_context)
                    yield (final_response_with_prompt, learning_style, sources)

    def _extract_sources(self, context: AgentContext) -> list[dict]:
        """
        Extract source citations from context.

        Looks for knowledge base retrieval results in the deep agent state.
        """
        sources = []

        # Check if RAG tool was used (would be in deep_agent_state)
        deep_agent_state = context.processing_data.get("deep_agent_state", {})

        # In production, you'd parse the tool calls to extract sources
        # For now, return empty list
        # TODO: Parse deep agent messages for tool call results

        return sources


# ============================================================================
# Streaming Integration for Gradio
# ============================================================================


class GiroAgentWrapper:
    """
    Wrapper for Giro Agent that provides a simple interface for Gradio.

    Handles context building, streaming, and state management.
    """

    def __init__(self, config: GiroAgentConfig):
        """Initialize wrapper with agent configuration."""
        self.agent = GiroAgent(config)
        self.config = config

    def update_config(self, **kwargs):
        """Update agent configuration dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Rebuild agent with new config
        self.agent = GiroAgent(self.config)

    async def chat_stream(
        self,
        message: str,
        topic: str,
        session_state: dict,
        history: list[tuple[str, str]],
    ) -> AsyncGenerator[tuple[str, Optional[str], list[dict]], None]:
        """
        Stream chat response.

        Args:
            message: User message
            topic: Current topic
            session_state: Session state dict
            history: Conversation history

        Yields:
            (partial_response, learning_style, sources)
        """
        # Update config with current topic
        self.config.current_topic = topic

        # Build context
        context = self._build_context(message, session_state, history)

        # Stream response
        async for partial, style, sources in self.agent.stream_response(context):
            yield (partial, style, sources)

    def _build_context(
        self,
        message: str,
        session_state: dict,
        history: list[tuple[str, str]],
    ) -> AgentContext:
        """Build agent context from Gradio inputs."""
        from services.agents.base import ConversationHistory

        # Build conversation history
        conv_history = ConversationHistory()
        for user_msg, assistant_msg in history:
            if user_msg:
                conv_history.add_user_message(user_msg)
            if assistant_msg:
                conv_history.add_assistant_message(assistant_msg)

        # Create context with new schema
        context = AgentContext(
            user_id=session_state.get("user_id", 1),
            session_id=session_state.get("session_id", "default"),
            db_session_id=session_state.get("db_session_id"),
            user_message=message,
            conversation_history=conv_history,
            persisted_state=session_state.get("persisted_state", {}),
        )

        return context