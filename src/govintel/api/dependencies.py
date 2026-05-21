"""FastAPI dependency injection."""

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from govintel.analysis.engine import AnalyticsEngine
from govintel.config import Settings, get_settings
from govintel.generation.gemini import GeminiClient
from govintel.generation.mistral import MistralClient
from govintel.generation.report import DatabaseContractRetriever, ReportGenerator
from govintel.models import SearchResult
from govintel.retrieval.reranker import CrossEncoderReranker
from govintel.retrieval.vector import ChromaVectorStore

_ENGINES: dict[str, AsyncEngine] = {}


class LazyChromaVectorStore:
    """Create the Chroma vector backend only when retrieval actually runs."""

    def __init__(
        self,
        *,
        chromadb_path: str,
        collection_name: str,
        model_name: str,
    ) -> None:
        self._chromadb_path = chromadb_path
        self._collection_name = collection_name
        self._model_name = model_name
        self._store: ChromaVectorStore | None = None

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Search the configured Chroma collection."""

        if self._store is None:
            self._store = ChromaVectorStore(
                chromadb_path=self._chromadb_path,
                collection_name=self._collection_name,
                model_name=self._model_name,
            )
        return self._store.search(query, top_k)


@lru_cache
def get_cached_settings() -> Settings:
    """Cached settings instance for dependency injection."""
    return get_settings()


def get_async_engine(database_url: str) -> AsyncEngine:
    """Cached async SQLAlchemy engine for API dependencies."""

    if database_url not in _ENGINES:
        _ENGINES[database_url] = create_async_engine(database_url, echo=False)
    return _ENGINES[database_url]


async def dispose_async_engines() -> None:
    """Dispose cached database engines during application shutdown."""

    for engine in list(_ENGINES.values()):
        await engine.dispose()
    _ENGINES.clear()


def get_report_generator(
    settings: Settings = Depends(get_cached_settings),
) -> ReportGenerator:
    """Build the report generation pipeline from application settings."""

    engine = get_async_engine(settings.database_url)
    provider = settings.generation_provider.lower()
    llm = (
        MistralClient(settings=settings)
        if provider == "mistral"
        else GeminiClient(settings=settings)
    )
    return ReportGenerator(
        llm=llm,
        retriever=DatabaseContractRetriever(
            engine=engine,
            vector_store=LazyChromaVectorStore(
                chromadb_path=settings.chromadb_path,
                collection_name=settings.chroma_collection_name,
                model_name=settings.embedding_model_name,
            ),
            reranker=CrossEncoderReranker(),
            corpus_limit=settings.retrieval_corpus_limit,
        ),
        analytics=AnalyticsEngine(engine=engine),
        prompt_version=settings.prompt_version,
    )
