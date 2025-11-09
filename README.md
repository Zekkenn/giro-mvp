# Giro Agent — AI Learning Assistant

A modern **Gradio + FastAPI** interface for AI-assisted learning with teacher supervision. Features a clean two-tab design: one for students to learn interactively, and one for teachers to manage materials and activities.

## ✨ Features

### Student Experience (Tab 1: 💬 Student Chat)
- **Topic Selection**: Choose from Mathematics, Physics, Chemistry, Biology, History, Computer Science
- **Activity Guidance**: Follow teacher-created activities with AI assistance
- **Interactive Chat**: Streaming responses with context awareness
- **Source Citations**: View materials and references for the current topic
- **Visual Learning**: Generate educational diagrams and visualizations on-demand

### Teacher Panel (Tab 2: 👨‍🏫 Teacher Panel)
- **Upload Learning Materials**: Add PDFs, documents, and resources organized by topic
- **Create Activities**: Design guided learning activities for students to follow
- **Material Management**: View and organize all uploaded materials
- **Activity Tracking**: Monitor available activities across all topics

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY for image generation
```

### 2. Setup Database
```bash
# Start PostgreSQL with Docker
docker-compose up -d postgres

# Run migrations
alembic upgrade head

# Seed initial data (topics, admin user, default config)
python scripts/db_manager.py seed
```

### 3. Run the Application
```bash
python app.py

# Access the interface:
# 📚 Student Chat: http://localhost:8000
# 👨‍🏫 Teacher Panel: http://localhost:8000/?tab=1
```

## 🧩 How It Works

The application uses a simple `Orchestrator` that you can replace with your production agent:

**orchestrator.py**: Handles chat streaming and image generation
- `chat_stream()`: Yields text tokens for streaming responses
- `generate_image()`: Creates educational visualizations

To integrate with the refactored database agents, update the orchestrator to use:
- `AgentOrchestrator` from `services/agents/orchestrator.py`
- `GiroAgent` or your custom agent implementation
- SQLAlchemy session for persistence

## 🐳 Docker
```bash
docker build -t edu-agent-gradio-pro .
docker run -it --rm -p 8000:8000 --env-file .env edu-agent-gradio-pro
```

## 🛠️ Extending ideas
- Add a **Teacher Review** tab that writes to Redis and an approval queue.
- Swap Topic Facts with a RAG retriever (e.g., Supabase pgvector).
- Add auth at the API layer (Clerk/Supabase/JWT) and keep the UI thin.
- Attach analytics (sessions, approval rate, learning gains).

## 🔐 Notes
No PII collection by default. Configure data retention and logs before pilots.

## 🤖 Development
This project was developed with guidance from Claude (Anthropic AI). The neurological profile and learning style detection system was designed during person-to-person sessions. Claude assisted in building the core architecture abstraction (pipeline components, agent framework, and state management) and linking the AI agent framework with the Giro educational assistant implementation. 
LangChain and AWS integrations (including Bedrock Knowledge Bases) were orchestrated by hand with Claude's guidance. Infrastructure was deployed by hand using the clients of each vendor. 
