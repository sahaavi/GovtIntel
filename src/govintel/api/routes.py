"""API route definitions."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from govintel.api.dependencies import get_report_generator
from govintel.generation.gemini import LLMGenerationError
from govintel.generation.report import (
    CitationValidationError,
    InsufficientEvidenceError,
    ReportGenerator,
    ReportParsingError,
)
from govintel.models import AnalysisQuery, IntelligenceBrief

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@router.post("/analyze")
async def analyze(
    query: AnalysisQuery,
    generator: ReportGenerator = Depends(get_report_generator),
) -> IntelligenceBrief:
    """Generate a procurement intelligence brief from retrieved contract evidence."""

    try:
        return await generator.generate(query, strategy="zero_shot")
    except (CitationValidationError, InsufficientEvidenceError, ReportParsingError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        logger.warning("LLM provider failure", exc_info=exc)
        raise HTTPException(status_code=502, detail="Generation provider failed") from exc
