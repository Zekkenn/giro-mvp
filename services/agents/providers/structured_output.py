"""
Structured output extraction.

This module provides a universal way to extract structured data from any LLM,
regardless of whether the provider supports native structured output.

Strategy:
1. If provider has native structured output → use it
2. Otherwise → use prompt engineering + JSON parsing
"""

import json
import logging
from typing import Any

from services.agents.providers.base import (
    BaseLLMProvider,
    LLMMessage,
    ProviderCapability,
    StructuredOutputMixin,
)

logger = logging.getLogger(__name__)


class StructuredOutputError(Exception):
    """Error during structured output extraction."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class StructuredOutputExtractor:
    """
    Extract structured data from any LLM provider.

    This class abstracts away provider differences, allowing preprocessors
    and postprocessors to work with any provider.
    """

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    async def extract(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        temperature: float = 0.0,
        max_retries: int = 2,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Extract structured data matching the schema.

        Args:
            messages: Messages to send to LLM
            schema: JSON schema for the output
            temperature: Sampling temperature
            max_retries: Number of retries on parsing failure
            timeout: Request timeout in seconds

        Returns:
            Parsed JSON object matching schema

        Raises:
            StructuredOutputError: If extraction fails after retries
        """

        try:
            # Strategy 1: Use native structured output if available
            if self.provider.has_capability(ProviderCapability.STRUCTURED_OUTPUT):
                return await self._extract_native(messages, schema, temperature, timeout)

            # Strategy 2: Use prompt engineering + parsing
            return await self._extract_with_prompting(messages, schema, temperature, max_retries, timeout)
        except Exception as e:
            raise StructuredOutputError(
                f"Failed to extract structured output: {str(e)}",
                {"schema": schema, "original_error": str(e)},
            ) from e

    async def _extract_native(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        temperature: float,
        timeout: float,
    ) -> dict[str, Any]:
        """Use provider's native structured output."""

        if not isinstance(self.provider, StructuredOutputMixin):
            raise RuntimeError("Provider claims structured output but doesn't implement mixin")

        return await self.provider.generate_structured(
            messages=messages,
            schema=schema,
            temperature=temperature,
            timeout=timeout,
        )

    async def _extract_with_prompting(
        self,
        messages: list[LLMMessage],
        schema: dict[str, Any],
        temperature: float,
        max_retries: int,
        timeout: float,
    ) -> dict[str, Any]:
        """Extract structured data using prompt engineering."""

        # Add schema to the last message
        schema_prompt = self._build_schema_prompt(schema)
        enhanced_messages = messages.copy()

        # Enhance last message with schema instructions
        if enhanced_messages:
            last_msg = enhanced_messages[-1]
            enhanced_messages[-1] = LLMMessage(
                role=last_msg.role,
                content=f"{last_msg.content}\n\n{schema_prompt}",
            )
        else:
            enhanced_messages.append(
                LLMMessage(
                    role="user",
                    content=schema_prompt,
                )
            )

        # Try to get valid JSON
        for attempt in range(max_retries + 1):
            try:
                response = await self.provider.generate(
                    messages=enhanced_messages,
                    temperature=temperature,
                    timeout=timeout,
                )

                # Extract JSON from response
                parsed = self._extract_json_from_text(response.content)

                # Validate against schema (basic validation)
                self._validate_schema(parsed, schema)

                return parsed

            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    f"JSON parsing failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                )

                if attempt < max_retries:
                    # Add correction message
                    enhanced_messages.append(
                        LLMMessage(
                            role="assistant",
                            content=response.content,
                        )
                    )
                    enhanced_messages.append(
                        LLMMessage(
                            role="user",
                            content=f"The response was not valid JSON. Error: {str(e)}\n\n"
                            f"Please provide a valid JSON response matching the schema.",
                        )
                    )
                else:
                    raise StructuredOutputError(
                        f"Failed to extract valid JSON after {max_retries + 1} attempts",
                        {"last_error": str(e), "attempts": max_retries + 1},
                    ) from e

    def _build_schema_prompt(self, schema: dict[str, Any]) -> str:
        """Build prompt instructions from JSON schema."""

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        properties_desc = []
        for prop_name, prop_schema in properties.items():
            prop_type = prop_schema.get("type", "any")
            prop_desc = prop_schema.get("description", "")
            is_required = prop_name in required

            properties_desc.append(f"  - {prop_name} ({prop_type}{'*' if is_required else ''}): {prop_desc}")

        return f"""Respond ONLY with a valid JSON object matching this schema:

Properties:
{chr(10).join(properties_desc)}

* = required field

Important:
- Return ONLY the JSON object, no other text
- Ensure all required fields are included
- Match the specified types exactly
"""

    def _extract_json_from_text(self, text: str) -> dict[str, Any]:
        """Extract JSON from text that might contain other content."""

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                json_text = text[start:end].strip()
                return json.loads(json_text)

        # Try to find JSON object in text
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            json_text = text[start:end]
            return json.loads(json_text)

        raise json.JSONDecodeError("No valid JSON found in response", text, 0)

    def _validate_schema(self, data: dict[str, Any], schema: dict[str, Any]) -> None:
        """Basic schema validation."""

        # Check required fields
        required = schema.get("required", [])
        for field in required:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # Check types (basic)
        properties = schema.get("properties", {})
        for field, value in data.items():
            if field in properties:
                expected_type = properties[field].get("type")
                if expected_type:
                    self._check_type(field, value, expected_type)

    def _check_type(self, field: str, value: Any, expected_type: str) -> None:
        """Check if value matches expected JSON schema type."""

        type_mapping = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list,
        }

        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type and not isinstance(value, expected_python_type):
            raise ValueError(f"Field '{field}' has type {type(value).__name__}, expected {expected_type}")
