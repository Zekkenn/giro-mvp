"""
Giro Agent Orchestrator - Integrates LangChain agent with Gradio frontend.

Provides a bridge between the Gradio UI and the NeuropsychAgent implementation
for A/B testing (baseline vs adaptive neuropsych-informed tutoring).
"""

import os
import asyncio
from typing import Generator, List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
import uuid

from dotenv import load_dotenv

from services.agents.implementations.neuropsych_agent import (
    NeuropsychAgentWrapper,
    NeuropsychAgentConfig,
    TutoringCondition,
    DifficultyLevel,
    SessionState,
)

load_dotenv()


@dataclass
class ChatMessage:
    """Simple message format for orchestrator."""
    role: str
    content: str


class GiroOrchestrator:
    """
    Orchestrator for Neuropsych-informed Agent with A/B testing support.

    Two conditions:
    - BASELINE: Activity guidance without profile adaptation
    - ADAPTIVE: Profile-aware teaching with difficulty adaptation

    This bridges the gap between Gradio's synchronous interface and
    our async LangChain agent pipeline.
    """

    def __init__(self, condition: TutoringCondition = TutoringCondition.ADAPTIVE):
        """Initialize Neuropsych Agent with default configuration."""
        self.condition = condition
        self.api_key = os.getenv("OPENAI_API_KEY", None)
        self._image_client = None

        # Create agent wrapper with specified condition
        self.agent_wrapper = NeuropsychAgentWrapper(
            condition=condition,
            model_name=os.getenv("OPENAI_MODEL", "gpt-5"),
        )

        # Initialize image generation client if API key available
        if self.api_key:
            try:
                from openai import OpenAI
                self._image_client = OpenAI(api_key=self.api_key)
            except Exception:
                self._image_client = None

    def create_session(
        self,
        session_id: str,
        student_id: str = "default_student",
        condition: Optional[TutoringCondition] = None,
    ) -> SessionState:
        """Create a new tutoring session."""
        return self.agent_wrapper.create_session(
            session_id=session_id,
            student_id=student_id,
            condition=condition or self.condition,
        )

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get existing session state."""
        return self.agent_wrapper.get_session(session_id)

    def get_progress(self, session_id: str) -> Dict[str, Any]:
        """Get session progress for UI display."""
        return self.agent_wrapper.get_progress(session_id)

    def switch_condition(self, session_id: str, condition: TutoringCondition):
        """Switch tutoring condition for a session."""
        self.agent_wrapper.switch_condition(session_id, condition)

    def chat_stream(
        self,
        subject: str,
        topic_facts: str,
        profile: dict,
        history: List[ChatMessage],
        message: str,
        session_id: str,
    ) -> Generator[str, None, None]:
        """
        Stream chat responses from Neuropsych Agent.

        Args:
            subject: Current topic
            topic_facts: Activity and material context (unused - we use Bedrock KB)
            profile: User profile settings (unused - we use Bedrock KB)
            history: Conversation history
            message: User's message
            session_id: Session identifier

        Yields:
            Response strings for display
        """
        # Ensure session exists
        if not self.agent_wrapper.get_session(session_id):
            self.create_session(session_id, student_id="student1")

        # Convert history to tuples
        history_tuples = []
        for i in range(0, len(history), 2):
            user_msg = history[i].content if i < len(history) else ""
            assistant_msg = history[i + 1].content if i + 1 < len(history) else ""
            if user_msg or assistant_msg:
                history_tuples.append((user_msg, assistant_msg))

        # Run async agent in sync context
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Get response from agent
                response, state_dict = loop.run_until_complete(
                    self.agent_wrapper.chat(
                        message=message,
                        session_id=session_id,
                        history=history_tuples,
                    )
                )

                # Yield the full response (NeuropsychAgent doesn't stream yet)
                yield response

            finally:
                loop.close()

        except Exception as e:
            # Fallback to simple response on error
            import traceback
            print(f"Error in chat_stream: {e}")
            traceback.print_exc()

            yield f"I'm having trouble processing your message right now. Error: {str(e)}\n\n"
            yield "Could you rephrase your question or ask your teacher for help?"

    def generate_image(
        self,
        prompt: str,
        size: str = "768x768",
        steps: int = 28,
        guidance: float = 7.5,
        seed: Optional[int] = -1
    ) -> List[str]:
        """
        Generate educational images.

        Args:
            prompt: Image description
            size: Image size (e.g., "768x768")
            steps: Generation steps (unused for OpenAI)
            guidance: Guidance scale (unused for OpenAI)
            seed: Random seed (unused for OpenAI)

        Returns:
            List of image file paths
        """
        if self._image_client:
            try:
                import base64
                import io
                from PIL import Image

                # Generate using OpenAI DALL-E
                result = self._image_client.images.generate(
                    model="dall-e-3",
                    prompt=prompt,
                    size="1024x1024",  # DALL-E 3 only supports specific sizes
                    n=1,
                    quality="standard",
                )

                # Get image URL (DALL-E returns URLs, not base64)
                image_url = result.data[0].url

                # Download and save image
                import requests
                response = requests.get(image_url)
                img = Image.open(io.BytesIO(response.content))

                # Save to temp directory
                os.makedirs("generated_images", exist_ok=True)
                out_path = f"generated_images/edu_{abs(hash(prompt))%999999}.png"
                img.save(out_path, format="PNG")

                return [out_path]

            except Exception as e:
                print(f"Image generation error: {e}")
                # Fall through to stub

        # Stub fallback
        try:
            from PIL import Image, ImageDraw

            w, h = [int(x) for x in size.split("x")]
            img = Image.new("RGB", (w, h), color=(245, 246, 250))
            d = ImageDraw.Draw(img)
            txt = f"[Educational Diagram]\n\n{prompt[:120]}\n\n(Add OPENAI_API_KEY to .env for real image generation)"
            d.multiline_text((24, 24), txt, fill=(50, 50, 60), spacing=8)

            os.makedirs("generated_images", exist_ok=True)
            out_path = f"generated_images/stub_{abs(hash(prompt))%999999}.png"
            img.save(out_path, format="PNG")

            return [out_path]

        except Exception as e:
            print(f"Stub image generation error: {e}")
            return []