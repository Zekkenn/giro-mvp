"""
Neuropsychologically-Informed Tutoring Agent.

Supports two conditions for A/B testing:
1. BASELINE: Activity guidance only (no profile adaptation)
2. ADAPTIVE: Profile-aware teaching with difficulty adaptation

Uses DeepAgentProvider with Bedrock RAG tools for:
- Activity content retrieval
- Student profile retrieval (ADAPTIVE mode only)

Tracks:
- Current step in activity
- Completed steps
- Student difficulty indicators
- Interaction logs for research analysis
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, Optional

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from services.agents.base import (
    BaseAgent,
    BaseAgentConfig,
    AgentContext,
    ConversationHistory,
)
from services.agents.providers.base import BaseLLMProvider, LLMMessage, LLMResponse
from services.agents.providers.deep_agent_provider import DeepAgentProvider
from services.agents.tools.bedrock_rag_tools import (
    retrieve_student_profile,
    retrieve_activity_content,
    get_adaptation_context,
    get_bedrock_rag_tools,
)


class TutoringCondition(str, Enum):
    """Experimental conditions for A/B testing."""
    BASELINE = "baseline"      # No profile, just activity guidance
    ADAPTIVE = "adaptive"      # Profile-aware with difficulty adaptation


class DifficultyLevel(str, Enum):
    """Difficulty levels for adaptive scaffolding."""
    EASY = "easy"              # More scaffolding, simpler language
    STANDARD = "standard"      # Normal level
    CHALLENGING = "challenging" # Less scaffolding, more independence


@dataclass
class StepProgress:
    """Tracks progress through activity steps."""
    step_number: int
    started_at: str
    completed_at: Optional[str] = None
    attempts: int = 0
    hints_given: int = 0
    difficulty_adjustments: int = 0
    student_responses: list = field(default_factory=list)


@dataclass
class SessionState:
    """Persistent state for a tutoring session."""
    session_id: str
    student_id: str
    condition: TutoringCondition
    current_step: int = 1
    total_steps: int = 5  # Adjust based on activity
    difficulty_level: DifficultyLevel = DifficultyLevel.STANDARD
    completed_steps: list = field(default_factory=list)
    step_progress: dict = field(default_factory=dict)  # step_num -> StepProgress
    profile_retrieved: bool = False
    profile_summary: Optional[str] = None
    interaction_count: int = 0
    session_started: bool = False  # Track if welcome message was shown

    def to_dict(self) -> dict:
        """Convert to dictionary for persistence."""
        data = asdict(self)
        data["condition"] = self.condition.value
        data["difficulty_level"] = self.difficulty_level.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        """Create from dictionary."""
        data["condition"] = TutoringCondition(data["condition"])
        data["difficulty_level"] = DifficultyLevel(data["difficulty_level"])
        return cls(**data)


@dataclass
class InteractionLog:
    """Log entry for research analysis."""
    timestamp: str
    session_id: str
    condition: str
    interaction_number: int
    current_step: int
    difficulty_level: str
    student_message: str
    agent_response: str
    profile_used: bool
    hints_given: int
    step_completed: bool
    difficulty_adjusted: bool
    profile_state: Optional[dict] = None


class NeuropsychAgentConfig(BaseAgentConfig):
    """Configuration for Neuropsych Agent."""
    name: str = "NeuropsychAgent"
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    condition: TutoringCondition = TutoringCondition.ADAPTIVE
    log_interactions: bool = True
    log_path: str = "interaction_logs"
    max_hints_per_step: int = 3


class OpenAIProvider(BaseLLMProvider):
    """Simple OpenAI provider for the agent."""

    def __init__(self, model_name: str = "gpt-5", temperature: float = 0.7):
        self.model = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY"),
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
            role = msg.role.value if hasattr(msg.role, 'value') else msg.role
            langchain_messages.append({"role": role, "content": msg.content})

        response = await self.model.ainvoke(langchain_messages)
        return LLMResponse(content=response.content, metadata={})


class NeuropsychAgent:
    """
    Neuropsychologically-informed tutoring agent using DeepAgent pipeline.

    Two modes:
    - BASELINE: Guides through activity without profile adaptation
    - ADAPTIVE: Uses student profile to adapt teaching approach
    """

    def __init__(self, config: NeuropsychAgentConfig):
        self.config = config

        # Setup logging directory
        if config.log_interactions:
            self.log_dir = Path(config.log_path)
            self.log_dir.mkdir(exist_ok=True)

    def _get_tools_for_condition(self, condition: TutoringCondition) -> list:
        """Get appropriate tools based on condition."""
        if condition == TutoringCondition.ADAPTIVE:
            # Full toolkit with profile access
            return get_bedrock_rag_tools()
        else:
            # Baseline: only activity content, no profile
            return [retrieve_activity_content]

    def _build_instructions(self, state: SessionState) -> str:
        """Build system instructions based on condition and state."""

        # Welcome/intro section
        intro = """You are GIRO, an AI tutor guiding a student through a mathematics activity about functions.

## IMPORTANT: Session Introduction
If this is the START of a session (first message from student), you MUST:
1. Greet the student warmly
2. Briefly explain what you'll be working on (functions activity with 5 exercises)
3. Explain your role as a supportive tutor
4. Ask if they're ready to begin Exercise 1

"""

        # Core instructions
        base_instructions = f"""## Current Progress
- **YOU ARE ON EXERCISE {state.current_step}** - Use query "Ejercicio {state.current_step}" to get its content
- **Completed exercises:** {state.completed_steps or 'None yet'}
- **Total exercises:** {state.total_steps}
- **Difficulty Level:** {state.difficulty_level.value}

## CRITICAL: Exercise Progression Rules

**CURRENT EXERCISE: You MUST work on Ejercicio {state.current_step}**
- To get content: use retrieve_activity_content with query "Ejercicio {state.current_step}"
- DO NOT skip ahead or go back to previous exercises

**WHEN TO MOVE TO NEXT EXERCISE:**
An exercise is COMPLETE when the student:
1. Correctly identifies the pattern/rule (e.g., "se suma 3", "the output is input plus 3")
2. Can apply it to a new example (e.g., correctly predicts what f(10) would be)
3. Explains the concept in their own words

**HOW TO TRANSITION:**
When exercise {state.current_step} is complete, say something like:
- "¡Excelente! Has entendido el Ejercicio {state.current_step}. Pasemos al Ejercicio {state.current_step + 1}."
Then use retrieve_activity_content with query "Ejercicio {state.current_step + 1}"

## CRITICAL: Socratic Teaching Method
You are a GUIDE, not an answer-giver. Your role is to help the student DISCOVER the answer themselves.

**NEVER DO THESE:**
- NEVER give the answer directly
- NEVER show the mathematical operation (like "4-1=3")
- NEVER fill in blanks for them
- NEVER give more than ONE small hint at a time
- NEVER present the full exercise content all at once - introduce it gradually

**ALWAYS DO THESE:**
- Ask ONE simple question at a time
- Wait for the student to respond before giving more guidance
- When they're stuck, ask a simpler related question
- Celebrate their own discoveries, no matter how small
- Let them struggle a bit - that's where learning happens

## Activity Guidance Rules
1. To get exercise content: query "Ejercicio {state.current_step}" (current exercise)
2. Present the exercise GRADUALLY - start with just the basic setup
3. Ask the student what they notice or think BEFORE giving any hints
4. If they ask "what does X mean?", explain the concept simply but DON'T solve it for them
5. Focus on ONE question at a time

## Example of GOOD vs BAD responses:

**Student says:** "No entiendo"

**BAD (too explicit):**
"La entrada es 1 y la salida es 4. Para encontrar la regla, resta: 4-1=3, 5-2=3, 6-3=3. Siempre sumas 3!"

**GOOD (Socratic):**
"Está bien no entender al principio. Miremos solo el primer ejemplo: entra un 1 y sale un 4. ¿Qué crees que le pasó al 1 para convertirse en 4?"

## Exercise Completion Flow
1. **Present**: Get content with "Ejercicio {state.current_step}", show basic setup
2. **Ask**: "¿Qué observas?" or "¿Qué crees que está pasando?"
3. **Guide**: Based on their answer, ask ONE follow-up question
4. **Verify**: When they give a correct answer, ask them to explain WHY or try another example
5. **Complete**: Once verified, celebrate and announce "Pasemos al Ejercicio {state.current_step + 1}"
6. **Next**: Query "Ejercicio {state.current_step + 1}" and start the new exercise
"""

        # Condition-specific instructions
        if state.condition == TutoringCondition.ADAPTIVE:
            base_instructions += """
## ADAPTIVE MODE - Profile-Aware Teaching
You have access to the student's neuropsychological profile via tools.

**At session start, use get_adaptation_context or retrieve_student_profile to understand:**
- Cognitive strengths to leverage (e.g., visual reasoning, sustained attention)
- Vulnerabilities to scaffold (e.g., rule acquisition, verbal working memory)
- Executive function considerations

**Teaching Adaptations:**
- LEVERAGE strengths: Use visual diagrams if visual reasoning is strong
- SCAFFOLD vulnerabilities: Provide explicit step-by-step rules
- STRUCTURE clearly: Clear goals and expectations upfront
- Use VISUAL representations when possible
- Keep instructions EXPLICIT and concise
"""
        else:
            base_instructions += """
## BASELINE MODE - Standard Teaching
Provide clear, standard explanations without specific cognitive adaptations.
Use a balanced approach suitable for any learner.
Do NOT use the profile retrieval tools - focus only on activity content.
"""

        # Difficulty-specific instructions
        if state.difficulty_level == DifficultyLevel.EASY:
            base_instructions += """
## Difficulty: EASY
- Provide more scaffolding and hints
- Break down concepts into smaller steps
- Use simpler language
- Give examples before asking questions
"""
        elif state.difficulty_level == DifficultyLevel.CHALLENGING:
            base_instructions += """
## Difficulty: CHALLENGING
- Encourage more independent thinking
- Ask open-ended questions
- Reduce scaffolding
- Challenge the student to explain their reasoning
"""

        return intro + base_instructions

    async def process_message(
        self,
        message: str,
        session_state: SessionState,
        history: list[tuple[str, str]],
    ) -> tuple[str, SessionState, InteractionLog]:
        """
        Process a student message using DeepAgent pipeline.

        Args:
            message: Student's message
            session_state: Current session state
            history: Conversation history

        Returns:
            (response, updated_state, interaction_log)
        """
        session_state.interaction_count += 1
        is_first_message = not session_state.session_started

        if is_first_message:
            session_state.session_started = True

        # Build agent context
        context = self._build_context(message, session_state, history)

        # Create DeepAgent with appropriate tools
        tools = self._get_tools_for_condition(session_state.condition)
        instructions = self._build_instructions(session_state)

        deep_agent = DeepAgentProvider(
            instructions=instructions,
            tools=tools,
            model_name=self.config.model_name,
            temperature=self.config.temperature,
            name="NeuropsychDeepAgent",
        )

        # Process through the agent
        result_context = await deep_agent.process(context)

        # Extract response
        agent_response = result_context.processing_data.get("final_response", "")

        # Analyze for step completion and difficulty adjustment
        step_completed, difficulty_adjusted = self._analyze_interaction(
            message, agent_response, session_state
        )

        # Track profile usage
        profile_used = (
            session_state.condition == TutoringCondition.ADAPTIVE
            and session_state.profile_retrieved
        )

        # Update session state
        if step_completed:
            session_state.completed_steps.append(session_state.current_step)
            session_state.current_step = min(
                session_state.current_step + 1,
                session_state.total_steps
            )

        # Create interaction log
        log_entry = InteractionLog(
            timestamp=datetime.now().isoformat(),
            session_id=session_state.session_id,
            condition=session_state.condition.value,
            interaction_number=session_state.interaction_count,
            current_step=session_state.current_step,
            difficulty_level=session_state.difficulty_level.value,
            student_message=message,
            agent_response=agent_response,
            profile_used=profile_used,
            hints_given=session_state.step_progress.get(
                session_state.current_step, StepProgress(session_state.current_step, "")
            ).hints_given,
            step_completed=step_completed,
            difficulty_adjusted=difficulty_adjusted,
            profile_state={"summary": session_state.profile_summary} if profile_used else None,
        )

        # Save log if enabled
        if self.config.log_interactions:
            self._save_log(log_entry)

        return agent_response, session_state, log_entry

    def _build_context(
        self,
        message: str,
        session_state: SessionState,
        history: list[tuple[str, str]],
    ) -> AgentContext:
        """Build agent context from inputs."""
        conv_history = ConversationHistory()
        for user_msg, assistant_msg in history:
            if user_msg:
                conv_history.add_user_message(user_msg)
            if assistant_msg:
                conv_history.add_assistant_message(assistant_msg)

        return AgentContext(
            user_id=1,
            session_id=session_state.session_id,
            user_message=message,
            conversation_history=conv_history,
            persisted_state={
                "condition": session_state.condition.value,
                "current_step": session_state.current_step,
                "difficulty_level": session_state.difficulty_level.value,
            },
        )

    def _analyze_interaction(
        self,
        student_message: str,
        agent_response: str,
        state: SessionState,
    ) -> tuple[bool, bool]:
        """
        Analyze interaction to determine:
        - If current step was completed
        - If difficulty should be adjusted

        Returns: (step_completed, difficulty_adjusted)
        """
        step_completed = False
        difficulty_adjusted = False

        # Initialize step progress if not exists
        if state.current_step not in state.step_progress:
            state.step_progress[state.current_step] = StepProgress(
                step_number=state.current_step,
                started_at=datetime.now().isoformat(),
            )

        # Record student response
        current_progress = state.step_progress[state.current_step]
        current_progress.student_responses.append(student_message)
        current_progress.attempts += 1

        # More robust completion detection - require explicit transition announcement
        # AND check that we're not already on this step
        next_step = state.current_step + 1
        completion_patterns = [
            f"pasemos al ejercicio {next_step}",
            f"vamos al ejercicio {next_step}",
            f"ejercicio {next_step}",
            f"moving to exercise {next_step}",
            f"next exercise",  # Only if it's a clear transition
        ]

        response_lower = agent_response.lower()

        # Only mark as complete if:
        # 1. Agent explicitly mentions moving to next exercise
        # 2. Current step is not already completed
        if state.current_step not in state.completed_steps:
            for pattern in completion_patterns:
                if pattern in response_lower:
                    # Additional validation: ensure it's actually a transition statement
                    # Look for context words like "pasemos", "vamos", "moving"
                    transition_words = ["pasemos", "vamos", "moving", "let's move", "empecemos con"]
                    has_transition = any(word in response_lower for word in transition_words)

                    if has_transition or f"ejercicio {next_step}" in response_lower:
                        step_completed = True
                        current_progress.completed_at = datetime.now().isoformat()
                        break

        # Count hints given in agent response
        hint_indicators = ["pista:", "hint:", "te ayudo", "piensa en", "recuerda que", "considera"]
        for indicator in hint_indicators:
            if indicator in response_lower:
                current_progress.hints_given += 1
                break

        # Check for difficulty indicators in student message
        struggle_phrases = [
            "no entiendo", "confundido", "ayuda", "difícil",
            "don't understand", "confused", "help", "difficult", "hard",
            "no sé", "no puedo"
        ]

        message_lower = student_message.lower()
        for phrase in struggle_phrases:
            if phrase in message_lower:
                # Consider lowering difficulty
                if state.difficulty_level != DifficultyLevel.EASY:
                    state.difficulty_level = DifficultyLevel.EASY
                    current_progress.difficulty_adjustments += 1
                    difficulty_adjusted = True
                break

        # Check for mastery indicators
        mastery_phrases = [
            "fácil", "ya entendí", "obvio", "simple",
            "easy", "got it", "obvious", "simple", "understood",
            "muy fácil", "pan comido"
        ]

        for phrase in mastery_phrases:
            if phrase in message_lower:
                # Consider raising difficulty
                if state.difficulty_level != DifficultyLevel.CHALLENGING:
                    state.difficulty_level = DifficultyLevel.CHALLENGING
                    current_progress.difficulty_adjustments += 1
                    difficulty_adjusted = True
                break

        return step_completed, difficulty_adjusted

    def _save_log(self, log: InteractionLog):
        """Save interaction log to file."""
        log_file = self.log_dir / f"{log.session_id}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(asdict(log)) + "\n")


# =============================================================================
# Wrapper for Gradio Integration
# =============================================================================


class NeuropsychAgentWrapper:
    """
    Wrapper for Gradio UI integration.

    Manages session state and provides simple interface.
    """

    def __init__(
        self,
        condition: TutoringCondition = TutoringCondition.ADAPTIVE,
        model_name: str = "gpt-4o",
    ):
        self.config = NeuropsychAgentConfig(
            condition=condition,
            model_name=model_name,
        )
        self.agent = NeuropsychAgent(self.config)
        self.sessions: dict[str, SessionState] = {}

    def create_session(
        self,
        session_id: str,
        student_id: str,
        condition: Optional[TutoringCondition] = None,
    ) -> SessionState:
        """Create a new tutoring session."""
        state = SessionState(
            session_id=session_id,
            student_id=student_id,
            condition=condition or self.config.condition,
        )
        self.sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get existing session state."""
        return self.sessions.get(session_id)

    async def chat(
        self,
        message: str,
        session_id: str,
        history: list[tuple[str, str]],
    ) -> tuple[str, dict]:
        """
        Process chat message.

        Args:
            message: Student message
            session_id: Session identifier
            history: Conversation history

        Returns:
            (response, state_dict)
        """
        # Get or create session
        state = self.sessions.get(session_id)
        if not state:
            state = self.create_session(session_id, "default_student")

        # Process message
        response, updated_state, log = await self.agent.process_message(
            message=message,
            session_state=state,
            history=history,
        )

        # Update stored state
        self.sessions[session_id] = updated_state

        return response, updated_state.to_dict()

    def get_progress(self, session_id: str) -> dict:
        """Get session progress for UI display."""
        state = self.sessions.get(session_id)
        if not state:
            return {}

        return {
            "current_step": state.current_step,
            "total_steps": state.total_steps,
            "completed_steps": state.completed_steps,
            "difficulty_level": state.difficulty_level.value,
            "condition": state.condition.value,
            "interaction_count": state.interaction_count,
            "profile_retrieved": state.profile_retrieved,
        }

    def switch_condition(self, session_id: str, condition: TutoringCondition):
        """Switch tutoring condition mid-session (for testing)."""
        state = self.sessions.get(session_id)
        if state:
            state.condition = condition
            state.profile_retrieved = False  # Reset to fetch profile in new condition