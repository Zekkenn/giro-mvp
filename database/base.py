"""
Database base configuration.

Provides the declarative base for all SQLAlchemy models.
"""

from sqlalchemy.orm import declarative_base

# Declarative base for all models
Base = declarative_base()

# Naming convention for constraints (helps with Alembic migrations)
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

Base.metadata.naming_convention = NAMING_CONVENTION