# Giro Agent - Database Setup Guide

Complete guide to setting up PostgreSQL database with Alembic migrations for local development.

---

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ with virtual environment
- AWS credentials (for Bedrock integration)

---

## Quick Start

### 1. Setup Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and configure:
# - AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
# - Bedrock settings (BEDROCK_KB_ID, S3_BUCKET_NAME)
# - Database credentials (optional - defaults work for local dev)
```

### 2. Start PostgreSQL with Docker Compose

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Verify it's running
docker-compose ps

# Check logs
docker-compose logs postgres
```

### 3. Install Python Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate  # or: .venv\Scripts\activate on Windows

# Install dependencies (including alembic, psycopg2, pydantic-settings)
pip install -r requirements.txt
```

### 4. Run Alembic Migrations

```bash
# Create initial migration
alembic revision --autogenerate -m "Initial schema"

# Apply migrations
alembic upgrade head

# Verify
python scripts/db_manager.py check
```

### 5. Seed Initial Data

```bash
# Add admin user, default agent config, and global topics
python scripts/db_manager.py seed
```

---

## Database Management Commands

### Check Connection
```bash
python scripts/db_manager.py check
```

### Initialize Database (creates tables - use Alembic instead in production)
```bash
python scripts/db_manager.py init
```

### Seed Data (admin user, default config, global topics)
```bash
python scripts/db_manager.py seed
```

### Reset Database (WARNING: deletes all data!)
```bash
python scripts/db_manager.py reset
```

---

## Alembic Migration Commands

### Create New Migration (autogenerate from model changes)
```bash
alembic revision --autogenerate -m "Description of changes"
```

### Apply Migrations
```bash
# Upgrade to latest
alembic upgrade head

# Upgrade one version
alembic upgrade +1

# Downgrade one version
alembic downgrade -1

# Downgrade to specific revision
alembic downgrade <revision_id>
```

### View Migration History
```bash
# Show current revision
alembic current

# Show migration history
alembic history --verbose

# Show pending migrations
alembic heads
```

---

## Docker Compose Services

### PostgreSQL Database
```bash
# Start
docker-compose up -d postgres

# Stop
docker-compose stop postgres

# View logs
docker-compose logs -f postgres

# Connect with psql
docker-compose exec postgres psql -U giro -d giro_agent
```

### PgAdmin (Database UI - optional)
```bash
# Start PgAdmin
docker-compose --profile tools up -d pgadmin

# Access at: http://localhost:5050
# Login: admin@giro.local / admin (from .env)

# Add server connection:
# - Host: postgres (Docker network)
# - Port: 5432
# - Database: giro_agent
# - Username: giro
# - Password: giro_dev_password
```

### Stop All Services
```bash
docker-compose down

# Stop and remove volumes (deletes data!)
docker-compose down -v
```

---

## Environment Variables

### Required for AWS Bedrock
```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here

BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
BEDROCK_KB_ID=your_kb_id_here

S3_BUCKET_NAME=giro-agent-knowledge-sources
```

### Database Configuration
```env
# Default values work for Docker Compose setup
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=giro_agent
POSTGRES_USER=giro
POSTGRES_PASSWORD=giro_dev_password
```

---

## Troubleshooting

### PostgreSQL won't start
```bash
# Check if port 5432 is already in use
lsof -i :5432

# View detailed logs
docker-compose logs postgres

# Remove and recreate
docker-compose down -v
docker-compose up -d postgres
```

### Alembic migration errors
```bash
# Check database connection
python scripts/db_manager.py check

# View current state
alembic current

# If migrations are out of sync, you may need to stamp
alembic stamp head  # Mark current DB as up-to-date

# Or reset and re-run
alembic downgrade base
alembic upgrade head
```

### Connection refused errors
```bash
# Make sure PostgreSQL is running
docker-compose ps

# Check connection in .env
cat .env | grep POSTGRES

# Test connection
python -c "from database.session import engine; engine.connect()"
```

---

## Database Schema

See [DATABASE_DESIGN_PROPOSAL.md](DATABASE_DESIGN_PROPOSAL.md) for complete schema documentation.

### Tables Created
- `users` - OAuth-integrated user accounts
- `learning_profiles` - Student learning preferences (with AI summary)
- `topics` - Educational topics (global + user-created)
- `knowledge_sources` - Bedrock Knowledge Base documents
- `learning_sessions` - Learning session tracking
- `messages` - Conversation messages (with Bedrock metrics)
- `session_states` - Agent state persistence
- `agent_configs` - Agent configuration presets

---

## Production Deployment

### Using Managed PostgreSQL (RDS, etc.)

1. Update `.env` with production database URL:
   ```env
   DATABASE_URL=postgresql://user:pass@your-rds-endpoint:5432/dbname
   ```

2. Run migrations:
   ```bash
   alembic upgrade head
   ```

3. Seed initial data:
   ```bash
   python scripts/db_manager.py seed
   ```

### Security Checklist

- [ ] Use strong passwords (not defaults)
- [ ] Enable SSL for database connections
- [ ] Set up database backups
- [ ] Configure AWS IAM roles (not access keys)
- [ ] Enable CloudWatch logging
- [ ] Set up monitoring/alerts
- [ ] Review security groups/firewall rules

---

## Next Steps

1. ✅ Database setup complete
2. Configure AWS Bedrock Knowledge Base
3. Upload learning materials to S3
4. Test agent integration
5. Set up OAuth authentication
6. Deploy to staging environment

For questions or issues, see the main [CLAUDE.md](../CLAUDE.md) documentation.