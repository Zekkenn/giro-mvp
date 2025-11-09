"""
Objective AI Agent Implementation.

Goal-oriented conversational agent with DeepAgents integration.
Uses LangChain's DeepAgents for advanced tool use and RAG.
"""

import os

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from services.agents.base import BaseAgent
from services.agents.postprocessors import WhatsAppFormatter
from services.agents.preprocessors import (
    ContextBuilder,
    GoalChecker,
    InteractionCounter,
    SentimentAnalyzer,
)
from services.agents.providers.base import BaseLLMProvider, LLMMessage, LLMResponse
from services.agents.providers.deep_agent_provider import DeepAgentProvider
from services.agents.tools import create_vector_store_search_tool


class OpenAIProvider(BaseLLMProvider):
    """Simple OpenAI provider for preprocessors."""

    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.7, api_key: str | None = None):
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
        langchain_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
        response = await self.model.ainvoke(langchain_messages)
        return LLMResponse(content=response.content, metadata={})


class ObjectiveAIConfig(BaseModel):
    """Configuration for Objective AI agent."""

    # Identity
    id: int
    name: str = "Objective AI"

    # Behavior
    objective: str = Field(
        ...,
        description="The goal to achieve (e.g., 'Schedule a demo appointment')",
    )
    initial_message: str = Field(
        ...,
        description="First message to send to user",
    )
    instructions: str = Field(
        default="",
        description="Additional instructions for the AI",
    )

    # Limits
    max_interactions: int = Field(
        default=10,
        description="Maximum conversation turns before giving up",
    )
    sentiment_threshold: float = Field(
        default=0.7,
        description="Threshold for negative sentiment detection (0.0-1.0)",
    )

    # RAG Configuration
    knowledge_base_id: str | None = Field(
        default=None,
        description="Bedrock Knowledge Base ID for RAG",
    )

    # Third party keys
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key (defaults to OPENAI_API_KEY env var)",
    )

    # Model Configuration
    model_name: str = Field(
        default="gpt-5-mini",
        description="LLM model to use",
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature",
    )

    # Features
    enable_rag: bool = Field(
        default=True,
        description="Enable RAG retrieval via search_knowledge_base tool",
    )
    enable_sentiment_check: bool = Field(
        default=True,
        description="Enable sentiment analysis preprocessor",
    )
    enable_goal_check: bool = Field(
        default=True,
        description="Enable goal achievement check preprocessor",
    )
    use_deep_agent: bool = Field(
        default=True,
        description="Use DeepAgent provider (requires OpenAI credentials). Set to False for demo mode.",
    )


class ObjectiveAIAgent(BaseAgent[ObjectiveAIConfig]):
    """
    Objective AI Agent with DeepAgents integration.

    Guides users toward completing a specific objective using:
    - Sentiment analysis (preprocessor)
    - Goal tracking (preprocessor)
    - RAG from knowledge base (via DeepAgents tool)
    - Advanced reasoning via DeepAgents
    """

    def __init__(
        self,
        config: ObjectiveAIConfig,
        db_session=None,
        llm_provider: BaseLLMProvider | None = None,
    ):
        self.db_session = db_session
        self.llm_provider = llm_provider
        super().__init__(config)

    async def initialize_conversation(self, context):
        """Initialize conversation with configured initial message."""
        from services.agents.base import AgentResponse

        return AgentResponse(
            content=self.config.initial_message,
            should_send=True,
            persisted_state_updates=context.persisted_state,
        )

    def _build_chain(self) -> Runnable:
        """
        Build the Objective AI LCEL chain with DeepAgents.

        Pipeline:
        1. [Preprocessor] Load context from DB
        2. [Preprocessor] Track interaction count
        3. [Preprocessor] Check sentiment → exit if negative
        4. [Preprocessor] Check goal → exit if achieved
        5. [DeepAgent] Process with tools and RAG
        6. [Postprocessor] Format for WhatsApp
        """

        # Build chain components
        chain_components = []

        # === PREPROCESSORS ===

        # Load conversation context from DB
        if self.db_session:
            chain_components.append(ContextBuilder(self.db_session).as_runnable())

        # Track interaction count
        chain_components.append(
            InteractionCounter(
                max_interactions=self.config.max_interactions,
            ).as_runnable()
        )

        # Use provided LLM provider or create default OpenAI provider
        llm_provider = self.llm_provider or OpenAIProvider(
            model_name=self.config.model_name,
            temperature=self.config.temperature,
        )

        # Check sentiment
        if self.config.enable_sentiment_check:
            chain_components.append(
                SentimentAnalyzer(
                    llm_provider=llm_provider,
                    threshold=self.config.sentiment_threshold,
                    exit_on_negative=True,
                ).as_runnable()
            )

        # Check goal achievement
        if self.config.enable_goal_check:
            chain_components.append(
                GoalChecker(
                    llm_provider=llm_provider,
                    objective=self.config.objective,
                    exit_on_achieved=True,
                ).as_runnable()
            )

        # === DEEP AGENT PROVIDER ===

        # Prepare tools for DeepAgent
        tools = []
        if self.config.enable_rag and self.config.knowledge_base_id:
            # Create file_search tool for the vector store
            tools.append(create_vector_store_search_tool(self.config.knowledge_base_id, self.config.openai_api_key))

        # Build comprehensive instructions
        instructions = f"""
            Remember to:
            - Guide the conversation toward achieving the objective
            - Use the file_search tool when you need information from the knowledge base
            - Be helpful, professional, and concise
            - Ask clarifying questions when needed
            - This is whatsapp conversation and you represent the company/product
            - Don't ask for any whatsapp location, we don't support it yet
            
            You are an AI assistant helping users achieve the following objective:
            
            {self.config.objective}
            
            {self.config.instructions}
            """

        # Create DeepAgent provider (no builtin tools)
        chain_components.append(
            DeepAgentProvider(
                instructions=instructions,
                tools=tools,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                builtin_tools=None,  # Don't use builtin tools
            ).as_runnable()
        )

        # === POSTPROCESSORS ===

        # Format for WhatsApp
        chain_components.append(WhatsAppFormatter().as_runnable())

        # Compose chain with LCEL pipe operator
        chain = chain_components[0]
        for component in chain_components[1:]:
            chain = chain | component

        # Add name to the overall chain for LangSmith tracing
        chain = chain.with_config({"run_name": f"ObjectiveAI: {self.config.name}"})

        return chain
