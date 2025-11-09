# Giro Agent - Database Design Proposal (Revised)

## Overview

PostgreSQL database schema for the Giro Agent educational platform with:
- **Agent-centric configuration** (not teacher-specific)
- **AWS Bedrock & AWS Vector Store** integration for knowledge base
- **OAuth-ready** user model (no built-in auth)
- Learning session tracking with conversation history
- Learning style detection and adaptation

---

## Design Principles

1. **Agent-first**: Configuration tied to agent instances, not teachers
2. **Cloud-native**: AWS Bedrock for LLMs, AWS Vector stores for knowledge
3. **Auth-agnostic**: No passwords, OAuth integration point defined
4. **Scalability**: Multi-tenant, multi-agent support
5. **Flexibility**: JSON for extensible metadata

---

## Entity Relationship Diagram

```
┌─────────────┐
│    User     │
│ (OAuth ID)  │
└──────┬──────┘
       │
       │
       ▼
┌──────────────┐       ┌─────────────┐
│  Learning    │◄─────►│   Topic     │
│  Session     │       └──────┬──────┘
└──────┬───────┘              │
       │                      │
       ├──────────┐           ▼
       │          │    ┌──────────────┐
       ▼          ▼    │  Knowledge   │
┌──────────┐ ┌────────┤   Source     │
│ Message  │ │Session │              │
│          │ │ State  │              │
└──────────┘ └────────┘              │
                                     │
       ┌─────────────┐               ▼
       │    Agent    │        [AWS Vector Store]
       │  Config     │        (OpenSearch/Kendra)
       └─────────────┘               │
                                     ▼
                              [AWS Bedrock]
                              (Knowledge Base)
```

---

## Core Tables

### 1. **users**
User entity with OAuth integration point.

**Columns:**
- `id` (PK): Integer, auto-increment
- `external_id`: String(255), unique, indexed (OAuth provider ID)
- `email`: String(255), unique, indexed
- `username`: String(100), unique, nullable
- `full_name`: String(255), nullable
- `avatar_url`: String(512), nullable
- `role`: Enum('student', 'educator', 'admin'), default='student'
- `is_active`: Boolean, default true
- `metadata`: JSON (OAuth provider, scopes, etc.)
- `created_at`: Timestamp with timezone
- `updated_at`: Timestamp with timezone
- `last_login_at`: Timestamp with timezone, nullable

**Relationships:**
- 1:Many → `learning_sessions`
- 1:1 → `learning_profiles` (optional)

**Design Notes:**
- `external_id` maps to OAuth provider's user ID (Google, Microsoft, etc.)
- No password storage - authentication delegated to OAuth
- `metadata` stores OAuth-specific data (provider, refresh tokens if needed)
- Role-based access control at application layer

**Future OAuth Integration:**
```python
# Example OAuth flow
user = User.query.filter_by(external_id=oauth_user_id).first()
if not user:
    user = User(
        external_id=oauth_user_id,
        email=oauth_email,
        full_name=oauth_name,
        metadata={"provider": "google", "oauth_data": {...}}
    )
```

---

### 2. **LEt's **
Student learning preferences and progress (optional, auto-created).

**Columns:**
- `id` (PK): Integer
- `user_id` (FK → users.id, unique): Integer
- `learning_style`: Enum('visual', 'auditory', 'kinesthetic', 'reading_writing', 'unknown'), default='unknown'
- `learning_style_confidence`: Float, default 0.0
- `learning_style_detected_at`: Timestamp, nullable
- `learning_style_indicators`: JSON, default=[] (evidence strings)
- **`ai_summary`**: Text, nullable (AI-generated analysis of student's learning patterns, strengths, areas for improvement)
- **`ai_summary_updated_at`**: Timestamp, nullable
- `total_sessions`: Integer, default 0
- `total_interactions`: Integer, default 0
- `topics_studied`: JSON, default=[] (topic IDs)
- `metadata`: JSON, default={}
- `created_at`: Timestamp
- `updated_at`: Timestamp
- **`deleted_at`**: Timestamp, nullable (soft delete)

**Relationships:**
- 1:1 ← `users`

**Design Notes:**
- Auto-created on first session
- AI generates summary of student's learning orientation, behavior patterns
- Summary updated periodically (e.g., every 10 sessions)
- Used by agent to personalize responses
- Soft delete support

---

### 3. **topics**
Educational subjects/topics (hybrid: global + user-created).

**Columns:**
- `id` (PK): Integer
- `name`: String(255), indexed
- `slug`: String(255), indexed (URL-friendly)
- `description`: Text, nullable
- `learning_objectives`: Text, nullable (markdown)
- `icon`: String(255), nullable (emoji or icon name)
- `parent_id` (FK → topics.id): Integer, nullable (for hierarchical topics)
- **`is_global`**: Boolean, default=false (true = admin-created, false = user-created)
- **`created_by`** (FK → users.id): Integer, nullable (null = system/admin)
- `is_active`: Boolean, default=true
- `metadata`: JSON, default={}
- `created_at`: Timestamp
- `updated_at`: Timestamp
- **`deleted_at`**: Timestamp, nullable (soft delete)

**Indexes:**
- (name, created_by) - unique constraint for user-scoped topics
- (slug, created_by) - unique constraint for user-scoped slugs
- (is_global, is_active)

**Relationships:**
- 1:Many → `knowledge_sources`
- 1:Many → `learning_sessions`
- Many:1 → `users` (creator, nullable)
- Self-referencing → `topics` (parent_id for hierarchy)

**Design Notes:**
- **Global topics** (`is_global=true, created_by=null`): Visible to all users, admin-managed
- **User topics** (`is_global=false, created_by=X`): Visible only to creator
- Hierarchical support (e.g., Math → Calculus → Derivatives)
- Soft delete support

**Example hierarchy:**
```
Mathematics (parent_id=null)
  ├─ Algebra (parent_id=1)
  ├─ Calculus (parent_id=1)
  └─ Statistics (parent_id=1)
```

---

### 4. **knowledge_sources**
Documents/sources indexed in AWS Bedrock Knowledge Base (multi-tenant with partitioning).

**Columns:**
- `id` (PK): Integer
- `topic_id` (FK → topics.id): Integer, nullable
- `title`: String(512)
- `description`: Text, nullable
- `source_type`: Enum('document', 'url', 'text'), default='document'
- **`is_global`**: Boolean, default=false (true = accessible to all, false = user-only)
- `file_name`: String(255), nullable
- `file_size`: Integer, nullable (bytes)
- `s3_bucket`: String(255), nullable (AWS S3 bucket)
- `s3_key`: String(512), nullable (AWS S3 object key)
- **`partition_key`**: String(255), indexed (user_id for partitioning in Bedrock KB)
- `bedrock_kb_id`: String(255), indexed (Bedrock Knowledge Base ID)
- `bedrock_data_source_id`: String(255), indexed (Bedrock Data Source ID)
- `vector_store_id`: String(255), nullable (OpenSearch index or Kendra index)
- `sync_status`: Enum('pending', 'syncing', 'synced', 'failed'), default='pending'
- `sync_error`: Text, nullable
- `last_synced_at`: Timestamp, nullable
- `chunk_count`: Integer, default=0
- `metadata`: JSON, default={} (tags, author, version, etc.)
- `uploaded_by` (FK → users.id): Integer, nullable
- `created_at`: Timestamp
- `updated_at`: Timestamp
- **`deleted_at`**: Timestamp, nullable (soft delete)

**Indexes:**
- (bedrock_kb_id, bedrock_data_source_id)
- (topic_id, sync_status)
- (uploaded_by, is_global)
- (partition_key)

**Relationships:**
- Many:1 → `topics`
- Many:1 → `users` (uploader)

**Design Notes:**
- **Multi-tenant partitioning**: `partition_key=user_{user_id}` for user content, `partition_key=global` for admin content
- **Global sources** (`is_global=true`): Visible to all users (admin-uploaded textbooks, curriculum)
- **User sources** (`is_global=false`): Only retrievable by uploader (private notes, custom materials)
- Bedrock KB uses metadata filters: `{"partition_key": ["user_123", "global"]}` for retrieval
- S3 storage for original files
- Sync status tracking for async ingestion
- Soft delete support

**AWS Integration Flow (with partitioning):**
```
1. User uploads PDF → S3 (key includes partition: user_123/document.pdf)
2. Create knowledge_sources (status=pending, partition_key="user_123")
3. Trigger Bedrock KB sync with metadata: {"partition_key": "user_123"}
4. Bedrock chunks, embeds, stores in OpenSearch with partition metadata
5. Update status to synced
6. On retrieval: filter by partition_key=["user_123", "global"]
```

---

### 5. **learning_sessions**
Individual learning sessions.

**Columns:**
- `id` (PK): Integer
- `user_id` (FK → users.id): Integer, indexed
- `topic_id` (FK → topics.id): Integer, nullable
- `agent_config_id` (FK → agent_configs.id): Integer, nullable
- `session_id`: String(255), unique, indexed (external session identifier)
- `status`: Enum('active', 'completed', 'paused', 'ended'), default='active'
- `interaction_count`: Integer, default=0
- `start_time`: Timestamp, default=now()
- `end_time`: Timestamp, nullable
- `duration_seconds`: Integer, nullable (calculated on end)
- `detected_learning_style`: Enum('visual', 'auditory', 'kinesthetic', 'reading_writing', 'unknown'), nullable
- `exit_reason`: Enum('achieved_goal', 'max_turns', 'user_exit', 'error', 'timeout'), nullable
- `metadata`: JSON, default={} (sources used, topics covered, etc.)
- `created_at`: Timestamp
- `updated_at`: Timestamp

**Indexes:**
- (user_id, status)
- (topic_id, status)
- (agent_config_id)
- (session_id, created_at)

**Relationships:**
- Many:1 → `users`
- Many:1 → `topics`
- Many:1 → `agent_configs`
- 1:Many → `messages`
- 1:1 → `session_states`

**Design Notes:**
- Links to agent config used for this session
- Tracks exit reason for analytics
- Duration calculated on session end

---

### 6. **messages**
Conversation messages within sessions.

**Columns:**
- `id` (PK): Integer
- `session_id` (FK → learning_sessions.id): Integer, indexed
- `role`: Enum('user', 'assistant', 'system'), not null
- `content`: Text, not null
- `bedrock_invocation_id`: String(255), nullable (Bedrock request ID)
- `model_used`: String(100), nullable (which Bedrock model)
- `input_tokens`: Integer, nullable
- `output_tokens`: Integer, nullable
- `latency_ms`: Integer, nullable
- `metadata`: JSON, default={} (sources, citations, tool calls)
- `created_at`: Timestamp

**Indexes:**
- (session_id, created_at)
- (bedrock_invocation_id)

**Relationships:**
- Many:1 → `learning_sessions`

**Design Notes:**
- Stores Bedrock-specific metadata (model, tokens, invocation ID)
- Citations/sources in metadata JSON
- Token tracking for cost analysis

---

### 7. **session_states**
Agent state persistence per session.

**Columns:**
- `id` (PK): Integer
- `session_id` (FK → learning_sessions.id, unique): Integer
- `state_data`: JSON, not null, default={}
- `last_updated`: Timestamp, default=now(), onupdate=now()

**Relationships:**
- 1:1 → `learning_sessions`

**Design Notes:**
- Stores agent memory, context, planning state
- Updated after each interaction
- Enables session recovery

---

### 8. **agent_configs**
Agent configuration presets (not user-specific).

**Columns:**
- `id` (PK): Integer
- `name`: String(255), unique, not null
- `description`: Text, nullable
- `is_default`: Boolean, default=false
- `is_active`: Boolean, default=true
-
- **Agent Behavior:**
- `max_turns`: Integer, default=20
- `require_teacher_check_every`: Integer, default=5
- `encourage_teacher_interaction`: Boolean, default=true
- `show_sources`: Boolean, default=true
-
- **Model Configuration:**
- `bedrock_model_id`: String(255), default='anthropic.claude-3-sonnet-20240229-v1:0'
- `bedrock_kb_id`: String(255), nullable (Knowledge Base ID)
- `temperature`: Float, default=0.7
- `max_tokens`: Integer, default=2048
- `top_p`: Float, default=0.9
-
- **System Prompt:**
- `system_prompt`: Text, nullable (base instructions)
-
- **Additional Settings:**
- `settings`: JSON, default={} (extensible)
-
- `created_at`: Timestamp
- `updated_at`: Timestamp

**Relationships:**
- 1:Many → `learning_sessions`

**Design Notes:**
- Pre-configured agent presets (e.g., "Math Tutor", "Science Guide")
- Multiple configs can exist, selected at session creation
- One default config for new sessions
- Not tied to specific users - reusable across platform

**Example configs:**
```sql
-- Default general learning agent
INSERT INTO agent_configs (name, is_default, bedrock_model_id) VALUES
('General Learning Assistant', true, 'anthropic.claude-3-sonnet-20240229-v1:0');

-- Specialized math tutor
INSERT INTO agent_configs (name, description, max_turns, bedrock_model_id) VALUES
('Math Tutor Pro', 'Deep reasoning for mathematics', 30, 'anthropic.claude-3-opus-20240229-v1:0');
```

---

## AWS Integration Points

### 1. **AWS Bedrock (LLM Provider)**
- Model invocation tracked in `messages.bedrock_invocation_id`
- Model ID stored in `agent_configs.bedrock_model_id`
- Supports Claude, Titan, Jurassic models

### 2. **AWS Bedrock Knowledge Bases**
- Knowledge Base ID in `agent_configs.bedrock_kb_id`
- Data sources tracked in `knowledge_sources.bedrock_data_source_id`
- Automatic chunking and embedding by Bedrock

### 3. **AWS Vector Stores (OpenSearch or Kendra)**
- Index ID stored in `knowledge_sources.vector_store_id`
- Used by Bedrock KB for retrieval
- Managed separately from PostgreSQL

### 4. **AWS S3 (File Storage)**
- Original files stored in S3
- Bucket/key tracked in `knowledge_sources.s3_bucket/s3_key`
- Accessed by Bedrock for ingestion

---

## Data Flow Example

### Knowledge Upload Flow:
```
1. User uploads PDF → S3 bucket
2. Create knowledge_sources record (status=pending)
3. Trigger Bedrock KB data source sync (async)
4. Bedrock chunks, embeds, stores in OpenSearch
5. Update knowledge_sources (status=synced, bedrock_data_source_id=xxx)
```

### Chat Flow:
```
1. User sends message → create message record (role=user)
2. Load session_states for context
3. Invoke Bedrock with:
   - model_id from agent_configs
   - knowledge_base_id from agent_configs
   - conversation history from messages
4. Bedrock retrieves from Knowledge Base (if needed)
5. Store response → messages (role=assistant, bedrock_invocation_id, citations)
6. Update session_states
7. Update learning_profiles (interaction_count, learning_style if detected)
```

---

## Migration Strategy

### Phase 1: Core Schema (Initial)
- users, learning_profiles, topics
- learning_sessions, messages, session_states
- agent_configs

### Phase 2: Knowledge Integration
- knowledge_sources
- AWS Bedrock KB setup
- S3 bucket configuration

### Phase 3: OAuth Integration
- Add OAuth provider tables if needed
- Implement OAuth flow
- Migrate external_id mapping

---

## ✅ Finalized Decisions

1. **Agent Configs**: ✅ Admin-only for now. Users cannot create custom configs (future iteration)

2. **Topics**: ✅ Hybrid approach - global taxonomy (admin-created) + user-created topics
   - Add `created_by` field to topics
   - Add `is_global` boolean flag

3. **Knowledge Sources**: ✅ Multi-tenant with partitioning
   - One global Bedrock KB but partitioned by `user_id` to avoid information leakage
   - User-created content isolated from other users' content
   - Global content accessible to all

4. **Session State**: ✅ Save every turn + summary every 5 turns
   - Add `state_summary` field to session_states
   - Background job generates summaries

5. **Analytics**: ✅ Not now - use SQL views on existing tables later

6. **Soft Deletes**: ✅ Add `deleted_at` timestamp to all major tables

7. **Multi-region**: ✅ Not for now - single region deployment

8. **Usage Tracking**: ✅ YES - Add `api_usage` table for cost tracking

---

## Next Steps

1. ✅ Review and finalize schema design
2. Create SQLAlchemy models
3. Set up Alembic migrations
4. Create database initialization script
5. Build AWS Bedrock integration layer
6. Refactor agents to use new DB models
7. Add OAuth integration (future)

---

## Estimated Storage (1 year, 100 users)

| Table              | Rows      | Size Est.  |
|--------------------|-----------|------------|
| users              | ~100      | <1 MB      |
| learning_profiles  | ~100      | <1 MB      |
| topics             | ~50       | <1 MB      |
| knowledge_sources  | ~200      | ~5 MB      |
| agent_configs      | ~10       | <1 MB      |
| learning_sessions  | ~10,000   | ~50 MB     |
| messages           | ~200,000  | ~500 MB    |
| session_states     | ~10,000   | ~100 MB    |
| **Total**          |           | **~656 MB**|

**S3 Storage**: ~5-10 GB (original documents)
**Vector Store**: Managed by AWS (OpenSearch/Kendra)