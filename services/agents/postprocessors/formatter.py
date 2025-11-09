"""
WhatsApp message formatter.

Converts AI responses to WhatsApp-compatible format.
"""

import re
import logging

from services.agents.base import AgentContext, PipelineStep

logger = logging.getLogger(__name__)


def _convert_bold_and_italic(text):
    """Convert Markdown bold+italic (***text***) to WhatsApp bold+italic (*_text_*)."""
    return re.sub(r"\*\*\*(.+?)\*\*\*", r"*_\1_*", text)


def _convert_italic(text):
    """Convert Markdown italic (*text*) to WhatsApp italic (_text_)."""
    # (Avoid matching * adjacent to _ or *)
    return re.sub(r"(?<![*_])\*(?![*_])(.+?)(?<![*_])\*", r"_\1_", text)


def _convert_bold(text):
    """Convert Markdown bold (**text** or __text__) to WhatsApp bold (*text*)."""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text)
    return re.sub(r"__(.+?)__", r"*\1*", text)


def _convert_strikethrough(text):
    """Convert Markdown strikethrough (~~text~~) to WhatsApp strikethrough (~text~)."""
    return re.sub(r"~~(.+?)~~", r"~\1~", text)


def _convert_headers(text):
    """Convert Markdown headers (# Header) to WhatsApp bold (*Header*)."""
    text = re.sub(r"^# (.+)$", r"*\1*", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"*\1*", text, flags=re.MULTILINE)
    return re.sub(r"^### (.+)$", r"*\1*", text, flags=re.MULTILINE)


def _remove_unsupported_markdown(text):
    """Remove unsupported Markdown elements like links and code blocks."""
    text = re.sub(r"\[(.+?)]\((.+?)\)", r"\2", text)
    return re.sub(r"```[\s\S]+?```", lambda m: m.group(0).replace("```", ""), text)


def _handle_bullets(text):
    """Convert Markdown bullet points to WhatsApp-style bullets with indentation."""
    lines = text.split("\n")
    output = []
    for line in lines:
        match = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        if match:
            leading_spaces = match.group(1)
            content = match.group(2)
            level = len(leading_spaces) // 2
            indent = "  " * level
            output.append(f"{indent}• {content}")
        else:
            output.append(line)
    return "\n".join(output)


def _strip_internal_references(text):
    text = re.sub(r"【.*?†.*?】", "", text)
    return re.sub(r"【.*?:.*?】", "", text)


class WhatsAppFormatter(PipelineStep):
    """
    Format response for WhatsApp.

    This postprocessor:
    1. Converts markdown to WhatsApp format
    2. Handles special characters
    3. Ensures message length limits
    4. Sets final_response in pipeline_data
    """

    def __init__(self, max_length: int = 4096):
        super().__init__(name="WhatsAppFormatter")
        self.max_length = max_length

    async def process(self, context: AgentContext) -> AgentContext:
        """Format response for WhatsApp."""

        response = context.processing_data.get("generated_response")
        if not response:
            return context

        # Apply formatting
        formatted = self._format_for_whatsapp(response)

        # Ensure length limit
        if len(formatted) > self.max_length:
            formatted = formatted[: self.max_length - 3] + "..."
            logger.warning(f"Response truncated to {self.max_length} characters")

        # Set final response (immutable)
        return context.add_processing_data("final_response", formatted)

    def _format_for_whatsapp(self, text: str) -> str:
        """
        Convert markdown to WhatsApp format.
        """

        text = _convert_italic(text)
        text = _convert_bold(text)
        text = _convert_bold_and_italic(text)
        text = _convert_strikethrough(text)
        text = _convert_headers(text)
        text = _remove_unsupported_markdown(text)
        text = _handle_bullets(text)
        text = _strip_internal_references(text)
        return text
