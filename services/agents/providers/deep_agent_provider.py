"""
DeepAgent Provider.

Wraps LangChain's DeepAgents library as a PipelineStep for use in agent chains.
"""

import os
from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from services.agents.base import AgentContext, PipelineStep


class DeepAgentProvider(PipelineStep):
    """
    DeepAgent provider with planning, tools, and sub-agents.

    Uses LangChain's DeepAgents library to create sophisticated agents with:
    - Task planning (write_todos tool)
    - Custom tools (passed as parameters)
    - File system operations (ls, read_file, write_file, edit_file)
    - Sub-agent spawning for specialized tasks

    This provider replaces traditional single-purpose providers with a
    multi-capability agent that can plan, retrieve, and execute.
    """

    def __init__(
        self,
        instructions: str,
        tools: list[BaseTool] | None = None,
        model_name: str = "gpt-5",
        temperature: float = 0.7,
        builtin_tools: list[str] | None = None,
        subagents: list[SubAgent] | None = None,
        interrupt_config: dict[str, Any] | None = None,
        api_key: str | None = None,
        name: str = "DeepAgentProvider",
    ):
        """
        Initialize deep agent provider.

        Args:
            instructions: System prompt/instructions for the agent
            tools: List of LangChain tools the agent can use
            model_name: OpenAI model to use
            temperature: Sampling temperature
            builtin_tools: List of builtin tool names (write_todos, ls, read_file, write_file, edit_file)
            subagents: List of sub-agent definitions
            interrupt_config: Human-in-the-loop configuration
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            name: Step name
        """
        super().__init__(name=name)
        self.instructions = instructions
        self.tools = tools or []
        self.model_name = model_name
        self.temperature = temperature
        self.builtin_tools = builtin_tools
        self.subagents = subagents
        self.interrupt_config = interrupt_config
        self.api_key = api_key

        # Build the deep agent
        self.agent_graph = self._build_agent()

    def _build_agent(self):
        """
        Build the deep agent using deepagents library.

        Returns:
            LangGraph CompiledGraph
        """
        # Create LLM
        model = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
        )

        # Create the deep agent
        graph = create_deep_agent(
            model=model,
            tools=self.tools,
            system_prompt=self.instructions,  # Changed from 'instructions'
            subagents=self.subagents,
            # Note: builtin_tools and interrupt_config are not in the current API
            # These would need to be handled differently if needed
        )

        return graph

    def _prepare_messages(self, context: AgentContext) -> list:
        """Prepare messages from context."""
        messages = []

        # Add recent conversation history (last 10 messages for context)
        for msg in context.conversation_history.messages[-10:]:
            messages.append(
                {
                    "role": msg.role.value,
                    "content": msg.content,
                }
            )

        # Add current user message
        messages.append(
            {
                "role": "user",
                "content": context.user_message,
            }
        )

        return messages

    async def process(self, context: AgentContext) -> AgentContext:
        """
        Process context through deep agent.

        Args:
            context: Agent context with user message

        Returns:
            Updated context with agent response
        """
        # Don't generate if should exit
        if context.should_exit:
            return context

        if not context.user_message:
            return context.update_processing_data({"error": "No user message provided"})

        try:
            messages = self._prepare_messages(context)

            # Run the agent
            result = await self.agent_graph.ainvoke({"messages": messages})

            # Extract response from result
            response_message = self._extract_response(result)

            # Update context with response
            return context.update_processing_data(
                {
                    "final_response": response_message,
                    "deep_agent_state": result,
                    "agent_type": "deep_agent",
                }
            )

        except Exception as e:
            return context.update_processing_data(
                {
                    "error": f"Deep agent failed: {str(e)}",
                    "error_type": type(e).__name__,
                }
            )

    async def stream_process(self, context: AgentContext):
        """
        Stream process context through deep agent with real-time token streaming.

        Args:
            context: Agent context with user message

        Yields:
            (partial_text, is_final, updated_context)
        """
        # Don't generate if should exit
        if context.should_exit:
            yield ("", True, context)
            return

        if not context.user_message:
            yield ("", True, context.update_processing_data({"error": "No user message provided"}))
            return

        try:
            messages = self._prepare_messages(context)

            # Stream events from the agent graph
            accumulated_text = ""
            final_state = None

            async for event in self.agent_graph.astream_events(
                {"messages": messages},
                version="v2"
            ):
                # Handle streaming chunks from LLM
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        accumulated_text += chunk.content
                        yield (accumulated_text, False, context)

                # Capture final state
                elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                    final_state = event["data"]["output"]

            # Extract final response
            if final_state:
                response_message = self._extract_response(final_state)
                updated_context = context.update_processing_data(
                    {
                        "final_response": response_message,
                        "deep_agent_state": final_state,
                        "agent_type": "deep_agent",
                    }
                )
                yield (response_message, True, updated_context)
            else:
                # Fallback if no final state captured
                updated_context = context.update_processing_data(
                    {
                        "final_response": accumulated_text,
                        "agent_type": "deep_agent",
                    }
                )
                yield (accumulated_text, True, updated_context)

        except Exception as e:
            error_context = context.update_processing_data(
                {
                    "error": f"Deep agent failed: {str(e)}",
                    "error_type": type(e).__name__,
                }
            )
            yield ("", True, error_context)

    def _extract_response(self, result: dict) -> str:
        """
        Extract response message from deep agent result.

        Args:
            result: Result from deep agent execution

        Returns:
            Response message content
        """
        # DeepAgents returns state with messages list
        messages = result.get("messages", [])

        if not messages:
            return "No response generated"

        # Get the last AI message
        for msg in reversed(messages):
            # Check if it's an AI message
            if hasattr(msg, "type") and msg.type == "ai":
                return msg.content
            elif isinstance(msg, dict) and msg.get("role") in ["assistant", "ai"]:
                return msg.get("content", "")

        # Fallback: return last message content
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            return last_msg.content
        elif isinstance(last_msg, dict):
            return last_msg.get("content", "No response")

        return "No response generated"
