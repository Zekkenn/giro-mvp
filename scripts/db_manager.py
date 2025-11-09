#!/usr/bin/env python3
"""
Database management CLI script.

Provides commands for common database operations:
- Initialize database
- Run migrations
- Create seed data
- Reset database
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.session import engine, create_all_tables, drop_all_tables, get_db_context
from database.models import User, AgentConfig, Topic
from database.enums import UserRole


def init_db():
    """Initialize database tables (use Alembic migrations instead in production)."""
    print("🔧 Creating all database tables...")
    create_all_tables()
    print("✅ Database initialized successfully!")


def reset_db():
    """Reset database (drop and recreate all tables)."""
    print("⚠️  WARNING: This will delete all data!")
    confirm = input("Type 'yes' to confirm: ")

    if confirm.lower() != "yes":
        print("❌ Reset cancelled")
        return

    print("🗑️  Dropping all tables...")
    drop_all_tables()

    print("🔧 Creating all tables...")
    create_all_tables()

    print("✅ Database reset complete!")


def seed_data():
    """Seed database with initial data."""
    print("🌱 Seeding database...")

    with get_db_context() as db:
        # Create admin user
        admin = User(
            external_id="admin_local_dev",
            email="admin@giro.local",
            username="admin",
            full_name="Giro Admin",
            role=UserRole.ADMIN,
            is_active=True,
            metadata_={"provider": "local", "dev_account": True},
        )
        db.add(admin)

        # Create default agent config
        default_config = AgentConfig(
            name="General Learning Assistant",
            description="Default configuration for general learning assistance",
            is_default=True,
            is_active=True,
            max_turns=20,
            require_teacher_check_every=5,
            encourage_teacher_interaction=True,
            show_sources=True,
            bedrock_model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9,
            system_prompt="You are Giro Agent, an AI learning assistant...",
        )
        db.add(default_config)

        # Create global topics
        topics_data = [
            ("Mathematics", "mathematics", "📐", "Core mathematical concepts and problem solving"),
            ("Physics", "physics", "🔬", "Physical sciences and natural laws"),
            ("Chemistry", "chemistry", "⚗️", "Chemical reactions and molecular structures"),
            ("Biology", "biology", "🧬", "Life sciences and living organisms"),
            ("History", "history", "📚", "World history and historical events"),
            ("Computer Science", "computer-science", "💻", "Programming and computational thinking"),
        ]

        for name, slug, icon, description in topics_data:
            topic = Topic(
                name=name,
                slug=slug,
                icon=icon,
                description=description,
                is_global=True,
                created_by=None,
                is_active=True,
            )
            db.add(topic)

        db.commit()

    print("✅ Seed data created successfully!")
    print("   - Admin user: admin@giro.local")
    print("   - Default agent config: General Learning Assistant")
    print("   - 6 global topics created")


def check_connection():
    """Check database connection."""
    print("🔍 Checking database connection...")
    try:
        with engine.connect() as conn:
            result = conn.execute(sa.text("SELECT version()"))
            version = result.scalar()
            print(f"✅ Connected to PostgreSQL: {version}")
            return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Giro Agent Database Manager")
    parser.add_argument(
        "command",
        choices=["init", "reset", "seed", "check"],
        help="Command to execute",
    )

    args = parser.parse_args()

    if args.command == "init":
        init_db()
    elif args.command == "reset":
        reset_db()
    elif args.command == "seed":
        seed_data()
    elif args.command == "check":
        check_connection()


if __name__ == "__main__":
    # Import here to avoid circular imports
    import sqlalchemy as sa
    main()