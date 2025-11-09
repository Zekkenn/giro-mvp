"""
Database configuration and settings.

Loads configuration from environment variables.
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, PostgresDsn


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    # PostgreSQL connection
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="giro_agent", alias="POSTGRES_DB")
    postgres_user: str = Field(default="giro", alias="POSTGRES_USER")
    postgres_password: str = Field(default="giro_dev_password", alias="POSTGRES_PASSWORD")

    # Database URL (can override individual settings)
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # Connection pool settings
    pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, alias="DB_POOL_TIMEOUT")
    pool_recycle: int = Field(default=3600, alias="DB_POOL_RECYCLE")

    # Debug mode
    echo_sql: bool = Field(default=False, alias="DEBUG")

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields from .env (like OPENAI_API_KEY, etc.)

    def get_database_url(self) -> str:
        """
        Get database URL for SQLAlchemy.

        Returns:
            Database connection URL
        """
        if self.database_url:
            return self.database_url

        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Singleton instance
settings = DatabaseSettings()
