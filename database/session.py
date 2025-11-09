"""
Database session management.

Provides database engine, session factory, and session context managers.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.engine import Engine

from database.config import settings


# Create database engine
engine = create_engine(
    settings.get_database_url(),
    echo=settings.echo_sql,
    pool_size=settings.pool_size,
    max_overflow=settings.max_overflow,
    pool_timeout=settings.pool_timeout,
    pool_recycle=settings.pool_recycle,
    pool_pre_ping=True,  # Verify connections before using
)


# Event listener to set session timezone to UTC
@event.listens_for(Engine, "connect")
def set_timezone(dbapi_conn, connection_record):
    """Set PostgreSQL session timezone to UTC."""
    cursor = dbapi_conn.cursor()
    cursor.execute("SET timezone='UTC'")
    cursor.close()


# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)


def get_db() -> Generator[Session, None, None]:
    """
    Get database session (dependency injection for FastAPI).

    Yields:
        Database session

    Example:
        ```python
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """
    Get database session as context manager.

    Yields:
        Database session

    Example:
        ```python
        with get_db_context() as db:
            user = db.query(User).first()
            print(user.email)
        ```
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_all_tables():
    """
    Create all database tables.

    WARNING: Use Alembic migrations in production!
    This is only for development/testing.
    """
    from database.base import Base

    Base.metadata.create_all(bind=engine)
    print("✅ All tables created successfully")


def drop_all_tables():
    """
    Drop all database tables.

    WARNING: This will delete all data!
    Only use in development/testing.
    """
    from database.base import Base

    Base.metadata.drop_all(bind=engine)
    print("⚠️  All tables dropped")
