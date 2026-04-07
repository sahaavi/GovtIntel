"""API route definitions."""

from typing import Any

from fastapi import APIRouter, Depends

from govintel.api.dependencies import get_cached_settings
from govintel.config import Settings
from govintel.models import AnalysisQuery

router = APIRouter(prefix="/api/v1")


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/analyze")
async def analyze(
    query: AnalysisQuery,
    settings: Settings = Depends(get_cached_settings),
) -> dict[str, Any]:
    """Generate a procurement intelligence brief.

    Stub implementation — will be wired to the full RAG pipeline in Phase 5.
    """
    return {
        "status": "stub",
        "query": query.question,
        "message": "Analysis pipeline not yet implemented",
    }
