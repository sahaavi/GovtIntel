"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Database
    database_url: str = "postgresql+asyncpg://govintel:govintel@localhost:5432/govintel"

    # USAspending API
    usaspending_base_url: str = "https://api.usaspending.gov/api/v2"

    # Gemini
    gemini_api_key: str = ""

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "govintel"

    # HuggingFace
    hf_api_token: str = ""
    hf_model_id: str = ""

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"

    # App
    app_env: str = "development"
    log_level: str = "INFO"


def get_settings() -> Settings:
    """Return application settings (cached via lru_cache if needed)."""
    return Settings()
