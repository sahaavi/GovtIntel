"""FastAPI dependency injection."""

from functools import lru_cache

from govintel.config import Settings, get_settings


@lru_cache
def get_cached_settings() -> Settings:
    """Cached settings instance for dependency injection."""
    return get_settings()
