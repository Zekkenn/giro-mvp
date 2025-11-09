"""
Base LLM provider interface.

All providers must implement this minimal interface.
This ensures preprocessors and postprocessors work with ANY provider.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel


class LLMMessage(BaseModel):
    """A message to send to the LLM."""

    role: str  # "user" | "assistant" | "system"
    content: str


class LLMResponse(BaseModel):
    """Response from LLM."""

    content: str
    usage: dict[str, int] | None = None
    model: str | None = None
    finish_reason: str | None = None
    raw_response: Any = None  # Provider-specific raw response


class ProviderCapability(str, Enum):
    """Capabilities that providers may support."""

    STRUCTURED_OUTPUT = "structured_output"  # Native JSON schema support
    FUNCTION_CALLING = "function_calling"  # Tool/function calling
    STREAMING = "streaming"  # Streaming responses
    VISION = "vision"  # Image understanding
    EMBEDDINGS = "embeddings"  # Generate embeddings


class BaseLLMProvider(ABC):
    """
    Minimal interface that ALL LLM providers must implement.

    Providers should only implement what they can guarantee to work.
    Advanced features are optional and declared via capabilities.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        timeout: float = 30.0,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a response from the LLM.

        This is the ONLY required method. All providers must implement this.

        Args:
            messages: Conversation messages
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Max tokens to generate
            timeout: Request timeout in seconds (default: 30s)
            **kwargs: Provider-specific parameters

        Returns:
            LLM response

        Raises:
            LLMTimeoutError: If request exceeds timeout
            LLMProviderError: If generation fails
        """
        pass

    def get_capabilities(self) -> set[ProviderCapability]:
        """
        Declare what this provider supports.

        Override to declare capabilities beyond basic generation.
        """
        return set()

    def has_capability(self, capability: ProviderCapability) -> bool:
        """Check if provider has a capability."""
        return capability in self.get_capabilities()


class StructuredOutputMixin:
    """
    Mixin for providers that support native structured output.

    Only add this if your provider has native JSON schema support (like OpenAI).
    """

    async def generate_structured(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        temperature: float = 0.0,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Generate structured output matching a JSON schema.

        Only implement this if your provider supports it natively.
        """
        raise NotImplementedError("Provider does not support native structured output")


class FunctionCallingMixin:
    """
    Mixin for providers that support function/tool calling.

    Only add this if your provider supports tool calling.
    """

    async def generate_with_tools(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        **kwargs,
    ) -> tuple[LLMResponse, list[dict[str, Any]]]:
        """
        Generate with tool calling support.

        Returns:
            (response, tool_calls)
        """
        raise NotImplementedError("Provider does not support function calling")
