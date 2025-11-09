"""
Bedrock RAG Tools for DeepAgents.

LangChain tools that integrate with BedrockKnowledgeManager for:
1. Neuropsychological profile queries (student-specific adaptations)
2. Activity content retrieval (curriculum materials)

These tools enable the GiroAgent to dynamically adapt teaching based on
the student's neuropsychological profile and retrieve relevant activity content.
"""

from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from services.bedrock_knowledge_manager import (
    get_bedrock_knowledge_manager,
    DocumentType,
)


# ============================================================================
# Tool Input Schemas
# ============================================================================


class StudentProfileQueryInput(BaseModel):
    """Input schema for student profile retrieval."""

    query: str = Field(
        description=(
            "The specific aspect of the student's neuropsychological profile to retrieve. "
            "Examples: 'cognitive strengths', 'learning vulnerabilities', "
            "'executive function characteristics', 'attention profile', "
            "'working memory capacity', 'visual-spatial abilities', "
            "'recommendations for teaching'"
        )
    )


class ActivityContentQueryInput(BaseModel):
    """Input schema for activity content retrieval."""

    query: str = Field(
        description=(
            "IMPORTANT: Use SIMPLE Spanish queries to find exercises. "
            "The activity uses Spanish exercise names. "
            "Examples of GOOD queries: 'Ejercicio 1', 'Ejercicio 2', 'Ejercicio 3', 'funciones', 'máquina de números'. "
            "Examples of BAD queries (won't work): 'Exercise 3 about functions with pattern +3', long English descriptions."
        )
    )


# ============================================================================
# LangChain Tools
# ============================================================================


@tool("retrieve_student_profile", args_schema=StudentProfileQueryInput)
def retrieve_student_profile(query: str) -> str:
    """
    Retrieve relevant information from the student's neuropsychological profile.

    Use this tool to understand the student's:
    - Cognitive strengths (e.g., visual reasoning, sustained attention, logic/programming interest)
    - Learning vulnerabilities (e.g., rule acquisition difficulties, verbal working memory limitations)
    - Executive function characteristics (e.g., tendency for unstructured exploration when tasks are ambiguous)
    - Attention and memory profiles
    - Processing speed and recommendations

    IMPORTANT: Use this information to adapt your teaching approach:
    - Leverage identified strengths in explanations (e.g., use visual diagrams for visual reasoners)
    - Provide extra scaffolding for vulnerability areas (e.g., explicit rules for rule acquisition issues)
    - Structure tasks appropriately based on executive function profile

    Args:
        query: What aspect of the profile to retrieve

    Returns:
        Relevant profile information with adaptation guidance
    """
    try:
        manager = get_bedrock_knowledge_manager()
    except ValueError as e:
        return f"Knowledge base not configured: {e}. Proceeding with general teaching approach."

    results = manager.retrieve_student_profile(query=query, num_results=5)

    if not results:
        return (
            f"No specific information found for: '{query}'. "
            "Consider asking about: 'cognitive strengths', 'vulnerabilities', "
            "'executive function', 'attention', 'memory', or 'recommendations'."
        )

    # Format results for the agent
    formatted = ["**Student Neuropsychological Profile:**\n"]

    for i, result in enumerate(results, 1):
        score = result.get("score", 0)
        content = result.get("content", "")

        # Include relevance indicator
        relevance = "High" if score > 0.7 else "Medium" if score > 0.5 else "Low"

        formatted.append(
            f"\n**[{i}] Relevance: {relevance}**\n"
            f"{content}\n"
        )

    formatted.append(
        "\n---\n"
        "**Use this information to:**\n"
        "- Leverage strengths when explaining concepts\n"
        "- Provide scaffolding for identified vulnerabilities\n"
        "- Structure tasks according to executive function profile\n"
    )

    return "\n".join(formatted)


@tool("retrieve_activity_content", args_schema=ActivityContentQueryInput)
def retrieve_activity_content(query: str) -> str:
    """
    Retrieve relevant content from the current activity worksheet/materials.

    IMPORTANT: Use simple Spanish queries like "Ejercicio 1", "Ejercicio 2", etc.
    DO NOT use long English descriptions - they won't find anything!

    Use this tool to find:
    - Definitions and explanations from the teacher-provided activity
    - Examples and exercises from the worksheet
    - Step-by-step instructions

    Args:
        query: Simple Spanish query (e.g., "Ejercicio 1", "Ejercicio 2", "funciones")

    Returns:
        Relevant activity content
    """
    try:
        manager = get_bedrock_knowledge_manager()
    except ValueError as e:
        return f"Knowledge base not configured: {e}. Providing general educational guidance."

    # Don't use topic filter - just semantic search
    results = manager.retrieve_activity_content(query=query, topic=None, num_results=5)

    if not results:
        return (
            f"No content found in activity materials for: '{query}'. "
            "The topic may not be covered in the current worksheet. "
            "Consider rephrasing or asking about a related concept."
        )

    # Format results
    formatted = ["**From Activity Materials:**\n"]

    for i, result in enumerate(results, 1):
        content = result.get("content", "")
        metadata = result.get("metadata", {})
        score = result.get("score", 0)

        # Extract location info
        location = result.get("location", {})
        s3_loc = location.get("s3Location", {})

        formatted.append(
            f"\n**[{i}]** (relevance: {score:.2f})\n"
            f"{content}\n"
        )

    return "\n".join(formatted)


class AdaptationContextInput(BaseModel):
    """Input schema for adaptation context retrieval."""

    situation: str = Field(
        description=(
            "Describe the current teaching situation or challenge. "
            "Examples: 'student is struggling with abstract concept', "
            "'student seems confused by the instructions', "
            "'need to explain a new mathematical rule', "
            "'student is losing focus', "
            "'starting a new exercise'. "
            "This helps retrieve the most relevant profile information."
        )
    )


@tool("get_adaptation_context", args_schema=AdaptationContextInput)
def get_adaptation_context(situation: str) -> str:
    """
    Get relevant student profile information for a specific teaching situation.

    Use this tool when you need to adapt your teaching approach based on
    the student's neuropsychological profile for a SPECIFIC situation.

    The situation you describe will determine what profile information is retrieved:
    - "struggling with abstract concept" → retrieves info about abstract reasoning
    - "confused by instructions" → retrieves info about verbal processing, working memory
    - "explaining a new rule" → retrieves info about rule acquisition
    - "losing focus" → retrieves info about attention profile
    - "starting exercise" → retrieves general strengths and teaching recommendations

    Args:
        situation: Description of the current teaching situation or challenge

    Returns:
        Relevant profile information for adapting to this situation
    """
    try:
        manager = get_bedrock_knowledge_manager()
    except ValueError as e:
        return f"Knowledge base not configured: {e}. Using general teaching strategies."

    # Map situations to relevant profile queries
    situation_lower = situation.lower()

    queries = []

    # Determine relevant queries based on situation
    if any(word in situation_lower for word in ["abstract", "concept", "understand", "comprende"]):
        queries.append("abstract reasoning conceptual understanding")
        queries.append("visual reasoning spatial abilities")

    if any(word in situation_lower for word in ["instruction", "confused", "confundido", "directions"]):
        queries.append("verbal working memory language processing")
        queries.append("executive function task instructions")

    if any(word in situation_lower for word in ["rule", "regla", "pattern", "patrón"]):
        queries.append("rule acquisition pattern learning")
        queries.append("explicit instruction scaffolding")

    if any(word in situation_lower for word in ["focus", "attention", "distract", "atención"]):
        queries.append("attention sustained focus concentration")
        queries.append("executive function self-regulation")

    if any(word in situation_lower for word in ["struggling", "difficult", "difícil", "hard"]):
        queries.append("learning vulnerabilities difficulties")
        queries.append("scaffolding support recommendations")

    if any(word in situation_lower for word in ["start", "begin", "new", "nuevo", "exercise"]):
        queries.append("cognitive strengths abilities")
        queries.append("teaching recommendations adaptations")

    # Default queries if no specific match
    if not queries:
        queries = [
            "cognitive strengths abilities",
            "learning vulnerabilities challenges",
            "teaching recommendations"
        ]

    # Retrieve relevant information
    all_results = []
    for query in queries[:3]:  # Limit to 3 queries
        results = manager.retrieve_student_profile(query=query, num_results=2)
        all_results.extend(results)

    if not all_results:
        return f"No profile information found for situation: '{situation}'. Using general teaching approach."

    # Build response
    summary = [f"**Profile Context for: '{situation}'**\n"]

    seen_content = set()
    for i, result in enumerate(all_results[:4], 1):  # Top 4 unique results
        content = result.get("content", "")
        content_key = content[:100]  # Dedup by first 100 chars

        if content_key not in seen_content:
            seen_content.add(content_key)
            score = result.get("score", 0)
            summary.append(f"\n**[{i}]** (relevance: {score:.2f})")
            summary.append(f"{content[:400]}...")

    summary.append(
        "\n---\n"
        f"**Recommendation for this situation:** "
        "Use the above profile information to adapt your response."
    )

    return "\n".join(summary)


# ============================================================================
# Tool Collection
# ============================================================================


def get_bedrock_rag_tools():
    """
    Get all Bedrock RAG tools for use in DeepAgents.

    Returns:
        List of LangChain tools for neuropsych-aware tutoring
    """
    return [
        retrieve_student_profile,
        retrieve_activity_content,
        get_adaptation_context,
    ]