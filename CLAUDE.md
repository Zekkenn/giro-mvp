# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Giro Agent** is a teacher-supervised, AI-powered learning assistant designed for classroom collaboration. It features automatic learning style detection, knowledge retrieval from teacher-uploaded materials, and encourages student-teacher interaction during class.

The project has two implementations:
- **Demo** (`app.py`, `orchestrator.py`): Simple Gradio UI with OpenAI direct integration for quick testing
- **Production** (`app_v2.py`, `services/agents/`): Full-featured system with agent pipeline architecture

## Development Commands

### Environment Setup
```bash
# Create virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Or using uv (preferred)
uv sync
```

### Running the Application
```bash
# Demo version (simple OpenAI integration)
python app.py

# Production version (full agent pipeline)
python app_v2.py

# Both open at http://localhost:8000
```

### Environment Configuration
```bash
# Copy example env file and configure
cp .env.example .env
# Add OPENAI_API_KEY for live LLM/image generation
# Set OPENAI_MODEL (default: gpt-4o-mini)
# Set PORT (default: 8000)
```

### Docker
```bash
# Build and run with Docker
docker build -t edu-agent-gradio-pro .
docker run -it --rm -p 8000:8000 --env-file .env edu-agent-gradio-pro
```

## Architecture

### Two-Layer Design

**Demo Layer** (`app.py`, `orchestrator.py`):
- Simple Gradio UI showcasing streaming chat and image generation
- `Orchestrator` class with two main hooks: `chat_stream()` and `generate_image()`
- File-based topic facts stored in `knowledge/<subject>.md`
- Learner profile controls (pace, verbosity, examples, temperature)
- Transcript export (Markdown/JSON)

**Production Agent System** (`app_v2.py`, `services/agents/`):
- Full Gradio UI with student and teacher interfaces
- Giro Agent: Specialized educational agent with learning style detection and knowledge retrieval
- LCEL (LangChain Expression Language) based pipeline architecture
- Vector-based knowledge management (ChromaDB + OpenAI embeddings)
- Teacher configuration: upload materials, set objectives, configure guardrails
- Automatic learning style adaptation
- Source citation and transparency

### Giro Agent Features (Production)

**`services/agents/giro_agent.py`**: The core educational agent implementation

#### Pipeline Architecture
```python
Sentiment → InteractionCounter → LearningStyleDetector → DeepAgent (with RAG) → TeacherInteractionFormatter
```

1. **Learning Style Detection** (`LearningStyleDetector`):
   - Analyzes conversation after 2-3 turns
   - Detects: Visual, Auditory, Kinesthetic, or Reading/Writing style
   - Stored in `persisted_state` for session continuity
   - Agent adapts teaching approach automatically

2. **Knowledge Retrieval** (`services/knowledge_manager.py`):
   - Teachers upload PDFs, DOCX, TXT, MD files via UI
   - Documents processed and chunked (1000 chars, 200 overlap)
   - Indexed in ChromaDB with OpenAI embeddings
   - Retrieved via `search_knowledge_base` tool (available to DeepAgent)
   - Sources cited in responses for transparency

3. **Teacher Interaction Prompting**:
   - Configured via guardrails: `encourage_teacher_interaction_every` (default: 5 turns)
   - Appends reminder after N interactions
   - Agent instructions emphasize teacher collaboration
   - Bot positions itself as a learning tool, not teacher replacement

4. **Streaming Support**:
   - `GiroAgentWrapper.chat_stream()` yields `(partial_response, learning_style, sources)`
   - Gradio UI updates in real-time
   - Session state tracks learning_style, interaction_count

#### Teacher Configuration (UI: "Teacher Configuration" tab)
- **Materials Upload**: Upload documents, auto-indexed to vector store
- **Topics**: Add/manage subject topics
- **Learning Objectives**: Set per-topic objectives (included in agent instructions)
- **Guardrails**:
  - Max conversation turns
  - Teacher check-in frequency
  - Enable/disable source citations
  - Enable/disable teacher interaction prompts

#### Student Interface (UI: "Student Chat" tab)
- Topic selector (synced with teacher config)
- Chat with streaming responses
- Learning style badge (appears after detection)
- Source citations panel (shows documents used)
- Teacher interaction reminders

### Core Agent Components

#### Base Abstractions (`services/agents/base.py`)
- `AgentContext`: Immutable context passed through pipeline (session_id, user_id, conversation_history, persisted_state, processing_data)
- `PipelineStep`: Base class for all pipeline components with `process()` and `as_runnable()` methods
- `BaseAgent`: Uses LCEL chains via `_build_chain()` to compose preprocessors → providers → postprocessors
- `AgentResponse`: Final response with content, exit_path, and persisted_state_updates

#### Pipeline Stages

**Preprocessors** (`services/agents/preprocessors/`):
- `ContextBuilder`: Loads conversation history and session state from database
- `SentimentAnalyzer`: Detects negative sentiment and triggers exit
- `GoalChecker`: Evaluates if objective is achieved
- `InteractionCounter`: Tracks turns and enforces max_interactions limit
- `LearningStyleDetector`: Analyzes student responses to detect learning style (Visual, Auditory, Kinesthetic, Reading/Writing)

**Providers** (`services/agents/providers/`):
- `DeepAgentProvider`: Wraps LangChain's DeepAgents with planning, tools, sub-agents
- `StructuredOutputProvider`: Generates structured responses with schema validation
- All providers check `context.should_exit` before generating

**Postprocessors** (`services/agents/postprocessors/`):
- `FaithfulnessChecker`: Validates response accuracy against source documents
- `ReflectionStep`: Self-critique and improvement loop
- `ResponseFormatter`: Formats output for delivery

**Tools** (`services/agents/tools/`):
- `RAGTool`: Retrieve from vector stores
- `VectorStoreTool`: Query embeddings

### Orchestration Flow

```
AgentOrchestrator.initialize_conversation() or .process_message()
  → Build AgentContext
  → Run agent.chain (LCEL pipeline)
  → _handle_response()
    → Persist state to DB
    → Send message
    → Map exit_reason to next step
```

### Key Patterns

**Immutability**: Always use `context.model_copy(update={...})` or helper methods like `context.add_processing_data()`, never mutate context directly.

**LCEL Composition**: Agents build chains by piping steps:
```python
def _build_chain(self):
    return (
        ContextBuilder(db).as_runnable() |
        SentimentAnalyzer().as_runnable() |
        GoalChecker().as_runnable() |
        DeepAgentProvider(...).as_runnable() |
        ResponseFormatter().as_runnable()
    )
```

**Exit Reasons**: `ExitReason` enum maps to edge types for conversation routing:
- `ACHIEVED_GOAL` → "achieved_goal"
- `NEGATIVE_SENTIMENT` → "critical_conversation"
- `MAX_INTERACTIONS` → "not_achieved_goal"
- `USER_REQUEST` → "user_exit"
- `ERROR` → "error"

## File Structure

- `app.py`: Demo Gradio UI with chat streaming, topic facts, image generation
- `app_v2.py`: **Production Gradio UI** with student/teacher interfaces, full Giro Agent integration
- `orchestrator.py`: Drop-in stub for LLM integration (demo only)
- `services/agents/`: Production-grade agent framework
  - `base.py`: Core abstractions and interfaces
  - **`giro_agent.py`**: Giro Agent implementation (educational agent)
  - `orchestrator.py`: Agent lifecycle manager (DB integration, for enterprise use)
  - `preprocessors/`:
    - **`learning_style_detector.py`**: Detects student learning preferences
    - `sentiment.py`, `goal_checker.py`, `interaction_counter.py`, `context_builder.py`
  - `providers/`:
    - `deep_agent_provider.py`: DeepAgents integration
    - `structured_output.py`, `base.py`
  - `postprocessors/`:
    - `faithfulness.py`, `reflection.py`, `formatter.py`
  - `tools/`:
    - **`rag_tool.py`**: Knowledge base retrieval (integrated with KnowledgeManager)
    - `vector_store_tool.py`
- **`services/knowledge_manager.py`**: Document upload, processing, vector storage (ChromaDB)
- `ui/theme.css`: Gradio custom styling
- `utils/transcript.py`: Export utilities
- `knowledge_base/`: Uploaded teacher materials (created at runtime)
- `vector_store/`: ChromaDB persistent storage (created at runtime)
- `teacher_config.json`: Teacher settings (topics, objectives, guardrails)

## Integration Points

### Wiring Your Orchestrator

To replace the demo orchestrator with production agents:

1. In `app.py`, replace `ORCHESTRATOR = Orchestrator()` with your agent
2. Implement `chat_stream(subject, topic_facts, profile, history, message, session_id) -> Generator[str]`
3. Implement `generate_image(prompt, size, steps, guidance, seed) -> List[str]`

Or bypass `orchestrator.py` entirely and call `AgentOrchestrator` from `services/agents/` with your custom `BaseAgent` subclass.

### State Persistence

- Session state is stored in a key-value store with key `giro_agent_{agent_name}`
- Conversation history stored in the messages table
- `ContextBuilder` loads both into `AgentContext` at pipeline start
- `AgentOrchestrator._persist_state()` saves `persisted_state_updates` after each turn

## Notes

- No PII collection by default in demo mode
- Configure data retention before production pilots
- LangSmith tracing enabled via step names in LCEL (set `LANGCHAIN_TRACING_V2=true`)
- Deep agents require `OPENAI_API_KEY` (or configure `api_key` parameter)