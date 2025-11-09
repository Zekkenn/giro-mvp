"""
Application configuration and settings.

Loads configuration from environment variables.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class AppSettings(BaseSettings):
    """Application-level configuration settings."""

    # OpenAI configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")

    # LangChain/LangSmith configuration
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_project: str = Field(default="giro-teacher", alias="LANGCHAIN_PROJECT")
    langchain_endpoint: str = Field(default="https://api.smith.langchain.com", alias="LANGCHAIN_ENDPOINT")
    langsmith_workspace_id: str = Field(default="default", alias="LANGSMITH_WORKSPACE_ID")

    # Server configuration
    port: int = Field(default=8000, alias="PORT")

    # AWS Bedrock configuration (optional)
    aws_region: Optional[str] = Field(default=None, alias="AWS_REGION")
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    bedrock_model_id: Optional[str] = Field(default=None, alias="BEDROCK_MODEL_ID")
    bedrock_kb_id: Optional[str] = Field(default=None, alias="BEDROCK_KB_ID")

    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton instance
app_settings = AppSettings()