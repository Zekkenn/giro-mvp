"""
Giro Agent - Educational AI Assistant
A modern interface for AI-assisted learning with teacher supervision.

Supports A/B testing with two conditions:
- BASELINE: Activity guidance without neuropsych profile adaptation
- ADAPTIVE: Profile-aware teaching with difficulty adaptation
"""

import os
import json
import uuid
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI
from dotenv import load_dotenv
import gradio as gr

from orchestrator_giro import GiroOrchestrator, ChatMessage
from services.agents.implementations.neuropsych_agent import TutoringCondition

load_dotenv()

# -------------------------
# State Management
# -------------------------
class SessionManager:
    """Manages learning sessions and uploaded materials."""

    def __init__(self):
        self.materials_dir = Path("uploaded_materials")
        self.activities_dir = Path("resources/activities")
        self.materials_dir.mkdir(exist_ok=True)
        self.activities_dir.mkdir(exist_ok=True)

        # Load or initialize metadata
        self.materials_db = self._load_db("materials_db.json")
        self.activities_db = self._load_db("activities_db.json")

    def _load_db(self, filename: str) -> dict:
        """Load database from JSON file."""
        path = Path(filename)
        if path.exists():
            return json.loads(path.read_text())
        return {"items": []}

    def _save_db(self, filename: str, data: dict):
        """Save database to JSON file."""
        Path(filename).write_text(json.dumps(data, indent=2))

    def add_material(self, file_path: str, topic: str, title: str, description: str) -> str:
        """Add uploaded material to database."""
        material = {
            "id": len(self.materials_db["items"]) + 1,
            "file_path": file_path,
            "topic": topic,
            "title": title,
            "description": description,
            "uploaded_at": datetime.now().isoformat(),
        }
        self.materials_db["items"].append(material)
        self._save_db("materials_db.json", self.materials_db)
        return f"✅ Material '{title}' added to {topic}"

    def add_activity(self, file_path: str, topic: str, title: str, instructions: str) -> str:
        """Add activity for students to follow."""
        activity = {
            "id": len(self.activities_db["items"]) + 1,
            "file_path": file_path,
            "topic": topic,
            "title": title,
            "instructions": instructions,
            "created_at": datetime.now().isoformat(),
        }
        self.activities_db["items"].append(activity)
        self._save_db("activities_db.json", self.activities_db)
        return f"✅ Activity '{title}' created for {topic}"

    def get_materials_by_topic(self, topic: str) -> List[Dict]:
        """Get all materials for a specific topic."""
        return [m for m in self.materials_db["items"] if m["topic"] == topic]

    def get_activities_by_topic(self, topic: str) -> List[Dict]:
        """Get all activities for a specific topic."""
        return [a for a in self.activities_db["items"] if a["topic"] == topic]

    def get_all_materials(self) -> str:
        """Get formatted list of all materials."""
        if not self.materials_db["items"]:
            return "No materials uploaded yet."

        output = []
        for m in self.materials_db["items"]:
            output.append(f"**{m['title']}** ({m['topic']})\n{m['description']}")
        return "\n\n".join(output)

    def get_all_activities(self) -> str:
        """Get formatted list of all activities."""
        if not self.activities_db["items"]:
            return "No activities created yet."

        output = []
        for a in self.activities_db["items"]:
            output.append(f"**{a['title']}** ({a['topic']})\n{a['instructions']}")
        return "\n\n".join(output)

SESSION_MANAGER = SessionManager()

# Create orchestrators for each condition
ORCHESTRATORS = {
    "adaptive": GiroOrchestrator(condition=TutoringCondition.ADAPTIVE),
    "baseline": GiroOrchestrator(condition=TutoringCondition.BASELINE),
}

# Available topics
TOPICS = ["Mathematics", "Physics", "Chemistry", "Biology", "History", "Computer Science"]

# Session tracking for persistent session IDs
ACTIVE_SESSIONS: Dict[str, str] = {}  # Maps condition -> session_id


def get_or_create_session_id(condition: str) -> str:
    """Get existing session ID or create new one for the condition."""
    if condition not in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS[condition] = f"{condition}_{uuid.uuid4().hex[:8]}"
    return ACTIVE_SESSIONS[condition]


def reset_session(condition: str) -> str:
    """Reset the session for a condition."""
    ACTIVE_SESSIONS[condition] = f"{condition}_{uuid.uuid4().hex[:8]}"
    return f"Session reset. New session: {ACTIVE_SESSIONS[condition]}"


def get_progress_display(condition: str, session_id: str) -> str:
    """Get formatted progress display for the UI."""
    orchestrator = ORCHESTRATORS.get(condition.lower(), ORCHESTRATORS["adaptive"])
    progress = orchestrator.get_progress(session_id)

    if not progress:
        return "**Session not started**\nSend a message to begin the activity."

    current = progress.get("current_step", 1)
    total = progress.get("total_steps", 5)
    completed = progress.get("completed_steps", [])
    difficulty = progress.get("difficulty_level", "standard")
    interaction_count = progress.get("interaction_count", 0)
    profile_retrieved = progress.get("profile_retrieved", False)
    condition_str = progress.get("condition", condition)

    # Build progress bar
    progress_bar = ""
    for i in range(1, total + 1):
        if i in completed:
            progress_bar += "🟢"
        elif i == current:
            progress_bar += "🔵"
        else:
            progress_bar += "⚪"

    return f"""**Condition:** `{condition_str.upper()}`
**Progress:** {progress_bar} ({len(completed)}/{total} completed)
**Current Exercise:** {current}
**Difficulty:** {difficulty.capitalize()}
**Interactions:** {interaction_count}
**Profile Used:** {"Yes" if profile_retrieved else "No"}"""


# -------------------------
# Student Chat Functions
# -------------------------
def stream_student_chat(
    message: str,
    history: List[Tuple[str, str]],
    topic: str,
    condition: str,
    current_activity: Optional[str] = None,
):
    """
    Stream chat responses for students.

    Args:
        message: Student message
        history: Conversation history
        topic: Current topic
        condition: Tutoring condition (adaptive or baseline)
        current_activity: Selected activity (unused - we use Bedrock KB)
    """
    # Get the right orchestrator for the condition
    orchestrator = ORCHESTRATORS.get(condition.lower(), ORCHESTRATORS["adaptive"])
    session_id = get_or_create_session_id(condition.lower())

    # Prepare context (mostly unused now - Bedrock KB handles this)
    topic_materials = SESSION_MANAGER.get_materials_by_topic(topic)
    topic_context = f"Topic: {topic}\n\nAvailable materials: {len(topic_materials)}"

    # Convert history
    prior_messages = []
    for user_msg, assistant_msg in history:
        if user_msg:
            prior_messages.append(ChatMessage(role="user", content=user_msg))
        if assistant_msg:
            prior_messages.append(ChatMessage(role="assistant", content=assistant_msg))

    # Stream response
    stream = orchestrator.chat_stream(
        subject=topic,
        topic_facts=topic_context,
        profile={"pace": "normal", "verbosity": "balanced"},
        history=prior_messages,
        message=message,
        session_id=session_id,
    )

    # Yield responses
    for chunk in stream:
        yield chunk

def generate_learning_image(prompt: str, topic: str):
    """Generate educational images to enhance learning."""
    if not prompt.strip():
        return None

    # Add educational context to prompt
    enhanced_prompt = f"Educational diagram for {topic}: {prompt}"

    # Generate image using default orchestrator
    try:
        images = ORCHESTRATORS["adaptive"].generate_image(
            prompt=enhanced_prompt,
            size="768x768",
            steps=28,
            guidance=7.5,
            seed=-1
        )
        return images
    except Exception as e:
        print(f"Image generation error: {e}")
        return None

def get_activity_list(topic: str) -> List[str]:
    """Get list of activities for topic dropdown."""
    activities = SESSION_MANAGER.get_activities_by_topic(topic)
    if not activities:
        return ["No activity selected"]
    return ["No activity selected"] + [a["title"] for a in activities]

# -------------------------
# Teacher Functions
# -------------------------
def upload_material(file, topic: str, title: str, description: str):
    """Handle material upload from teacher."""
    if file is None:
        return "⚠️ Please select a file to upload."

    if not title.strip():
        return "⚠️ Please provide a title for the material."

    # Save file
    file_path = SESSION_MANAGER.materials_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())

    # Add to database
    result = SESSION_MANAGER.add_material(
        file_path=str(file_path),
        topic=topic,
        title=title,
        description=description or "No description provided"
    )

    return result

def create_activity(topic: str, title: str, instructions: str, activity_file):
    """Create a new activity for students."""
    if not title.strip():
        return "⚠️ Please provide a title for the activity."

    if not instructions.strip():
        return "⚠️ Please provide instructions for the activity."

    # Save activity file if provided
    file_path = None
    if activity_file:
        file_path = SESSION_MANAGER.activities_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{activity_file.name}"
        with open(file_path, "wb") as f:
            f.write(activity_file.read())

    # Add to database
    result = SESSION_MANAGER.add_activity(
        file_path=str(file_path) if file_path else "",
        topic=topic,
        title=title,
        instructions=instructions
    )

    return result

# -------------------------
# UI Theme
# -------------------------
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=["Inter", "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_label_text_weight="600",
    block_title_text_weight="600",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

# -------------------------
# Gradio Interface
# -------------------------
with gr.Blocks(theme=custom_theme, title="Giro Agent - AI Learning Assistant") as demo:
    gr.Markdown(
        """
        # 🎓 Giro Agent
        ### AI-Powered Learning with Teacher Supervision
        """
    )

    with gr.Tabs() as tabs:
        # ==================== STUDENT TAB ====================
        with gr.Tab("💬 Student Chat", id=0):
            with gr.Row():
                with gr.Column(scale=2):
                    # A/B Testing Condition Selector
                    gr.Markdown("#### Research Condition")
                    condition_selector = gr.Radio(
                        choices=["Adaptive", "Baseline"],
                        value="Adaptive",
                        label="Tutoring Mode",
                        info="Adaptive uses neuropsych profile; Baseline does not",
                        interactive=True
                    )

                    # Progress Tracking Panel
                    with gr.Accordion("Activity Progress", open=True):
                        progress_display = gr.Markdown(
                            "**Session not started**\nSend a message to begin the activity."
                        )
                        refresh_progress_btn = gr.Button("Refresh Progress", size="sm")

                        def update_progress(condition):
                            session_id = get_or_create_session_id(condition.lower())
                            return get_progress_display(condition.lower(), session_id)

                        refresh_progress_btn.click(
                            fn=update_progress,
                            inputs=[condition_selector],
                            outputs=[progress_display]
                        )

                        # Also update when condition changes
                        condition_selector.change(
                            fn=update_progress,
                            inputs=[condition_selector],
                            outputs=[progress_display]
                        )

                    # Reset Session Button
                    reset_btn = gr.Button("Reset Session", variant="secondary", size="sm")
                    reset_status = gr.Markdown("")

                    def handle_reset(condition):
                        result = reset_session(condition.lower())
                        progress = get_progress_display(condition.lower(), get_or_create_session_id(condition.lower()))
                        return result, progress

                    reset_btn.click(
                        fn=handle_reset,
                        inputs=[condition_selector],
                        outputs=[reset_status, progress_display]
                    )

                    # Topic selection
                    student_topic = gr.Dropdown(
                        choices=TOPICS,
                        value="Mathematics",
                        label="Select Topic",
                        interactive=True
                    )

                    # Activity selection (dynamic based on topic)
                    student_activity = gr.Dropdown(
                        choices=["No activity selected"],
                        value="No activity selected",
                        label="Current Activity",
                        interactive=True,
                        visible=False  # Hidden - we use Bedrock KB now
                    )

                    # Update activity list when topic changes
                    student_topic.change(
                        fn=get_activity_list,
                        inputs=[student_topic],
                        outputs=[student_activity]
                    )

                    # Source citations panel
                    with gr.Accordion("Sources & References", open=False):
                        sources_display = gr.Markdown("Using AWS Bedrock Knowledge Base for content retrieval.")

                        def update_sources(topic):
                            return "Content is retrieved from AWS Bedrock Knowledge Base:\n- Student neuropsych profile\n- Activity worksheet (Ejercicios 1-5)"

                        student_topic.change(
                            fn=update_sources,
                            inputs=[student_topic],
                            outputs=[sources_display]
                        )

                    # Image generation panel
                    with gr.Accordion("Generate Learning Visual", open=False):
                        img_prompt = gr.Textbox(
                            label="Describe what you want to visualize",
                            placeholder="e.g., 'A diagram showing photosynthesis process with labels'"
                        )
                        img_btn = gr.Button("Generate Image", variant="primary")
                        img_output = gr.Gallery(
                            label="Generated Images",
                            columns=1,
                            height=300,
                            show_label=False
                        )

                        img_btn.click(
                            fn=generate_learning_image,
                            inputs=[img_prompt, student_topic],
                            outputs=[img_output]
                        )

                with gr.Column(scale=3):
                    # Main chat interface with condition input
                    chat = gr.ChatInterface(
                        fn=stream_student_chat,
                        type="tuples",
                        additional_inputs=[student_topic, condition_selector, student_activity],
                        title=None,
                    )

        # ==================== TEACHER TAB ====================
        with gr.Tab("👨‍🏫 Teacher Panel", id=1):
            gr.Markdown("### Upload Learning Materials & Create Activities")

            with gr.Row():
                # Material Upload Section
                with gr.Column():
                    gr.Markdown("#### 📤 Upload Learning Material")

                    material_file = gr.File(
                        label="Select File (PDF, DOCX, TXT, etc.)",
                        file_types=[".pdf", ".docx", ".txt", ".md"]
                    )
                    material_topic = gr.Dropdown(
                        choices=TOPICS,
                        value="Mathematics",
                        label="Topic"
                    )
                    material_title = gr.Textbox(
                        label="Material Title",
                        placeholder="e.g., 'Quadratic Equations - Chapter 5'"
                    )
                    material_description = gr.TextArea(
                        label="Description",
                        placeholder="Brief description of the material content...",
                        lines=3
                    )
                    upload_btn = gr.Button("Upload Material", variant="primary")
                    upload_status = gr.Markdown("")

                    upload_btn.click(
                        fn=upload_material,
                        inputs=[material_file, material_topic, material_title, material_description],
                        outputs=[upload_status]
                    )

                # Activity Creation Section
                with gr.Column():
                    gr.Markdown("#### 📝 Create Student Activity")

                    activity_file = gr.File(
                        label="Activity File (Optional - Worksheet, Assignment, etc.)",
                        file_types=[".pdf", ".docx", ".txt", ".md"]
                    )
                    activity_topic = gr.Dropdown(
                        choices=TOPICS,
                        value="Mathematics",
                        label="Topic"
                    )
                    activity_title = gr.Textbox(
                        label="Activity Title",
                        placeholder="e.g., 'Solving Linear Equations Practice'"
                    )
                    activity_instructions = gr.TextArea(
                        label="Instructions for Students",
                        placeholder="Describe what students should do, step by step...",
                        lines=5
                    )
                    create_activity_btn = gr.Button("Create Activity", variant="primary")
                    activity_status = gr.Markdown("")

                    create_activity_btn.click(
                        fn=create_activity,
                        inputs=[activity_topic, activity_title, activity_instructions, activity_file],
                        outputs=[activity_status]
                    )

            # View uploaded materials and activities
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📚 All Uploaded Materials")
                    all_materials = gr.Markdown(SESSION_MANAGER.get_all_materials())
                    refresh_materials_btn = gr.Button("Refresh", size="sm")
                    refresh_materials_btn.click(
                        fn=lambda: SESSION_MANAGER.get_all_materials(),
                        inputs=[],
                        outputs=[all_materials]
                    )

                with gr.Column():
                    gr.Markdown("#### 📝 All Activities")
                    all_activities = gr.Markdown(SESSION_MANAGER.get_all_activities())
                    refresh_activities_btn = gr.Button("Refresh", size="sm")
                    refresh_activities_btn.click(
                        fn=lambda: SESSION_MANAGER.get_all_activities(),
                        inputs=[],
                        outputs=[all_activities]
                    )

# -------------------------
# FastAPI Integration
# -------------------------
app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"🚀 Starting Giro Agent on http://localhost:{port}")
    print(f"📚 Student Chat: http://localhost:{port}")
    print(f"👨‍🏫 Teacher Panel: http://localhost:{port}/?tab=1")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)