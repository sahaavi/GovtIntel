"""Application configuration via environment variables."""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # Database
    database_url: str = "postgresql+asyncpg://govintel:govintel@localhost:5432/govintel"

    # USAspending API
    usaspending_base_url: str = "https://api.usaspending.gov/api/v2"
    ingestion_naics_code: str = "541512"
    ingestion_max_pages: int = 2
    ingestion_page_size: int = 100

    # Gemini
    gemini_api_key: str = ""

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "govintel"

    # Retrieval
    chromadb_path: str = "./data/chromadb"
    chroma_collection_name: str = "contracts"
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # HuggingFace
    hf_api_token: str = ""
    hf_model_id: str = ""

    # Generation
    generation_provider: str = "gemini"
    prompt_version: str = "v1"
    retrieval_corpus_limit: int = Field(default=5_000, ge=1, le=25_000)

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
