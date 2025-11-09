"""
Tools for AI Agents.

LangChain-compatible tools that can be used by deepagents.
"""

from services.agents.tools.bedrock_rag_tools import (
    retrieve_student_profile,
    retrieve_activity_content,
    get_adaptation_context,
    get_bedrock_rag_tools,
)

__all__ = [
    # Bedrock RAG tools for neuropsych-informed tutoring
    "retrieve_student_profile",
    "retrieve_activity_content",
    "get_adaptation_context",
    "get_bedrock_rag_tools",
]
