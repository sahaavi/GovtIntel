"""Pydantic schemas for GovIntel data models."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field


class ContractAward(BaseModel):
    """A single federal contract award from USAspending.gov."""

    award_id: str
    recipient_name: str
    awarding_agency: str
    award_amount: float
    start_date: date
    end_date: date | None = None
    naics_code: str = ""
    description: str = ""
    place_of_performance_state: str = ""
    award_type: str = ""

    @classmethod
    def from_usaspending(cls, raw: dict[str, Any]) -> ContractAward:
        """Parse a USAspending API response into a ContractAward.

        Handles nested field paths from the USAspending /awards/ endpoint.
        """
        return cls(
            award_id=raw.get("generated_internal_id", raw.get("id", "")),
            recipient_name=_nested_get(raw, ["recipient", "recipient_name"], "Unknown"),
            awarding_agency=_nested_get(
                raw, ["awarding_agency", "toptier_agency", "name"], "Unknown"
            ),
            award_amount=float(raw.get("total_obligation", 0) or 0),
            start_date=raw.get("period_of_performance_start_date", "2000-01-01"),
            end_date=raw.get("period_of_performance_current_end_date"),
            naics_code=raw.get("naics_code", "") or "",
            description=raw.get("description", "") or "",
            place_of_performance_state=_nested_get(raw, ["place_of_performance", "state_code"], ""),
            award_type=raw.get("type_description", "") or "",
        )


class AnalysisQuery(BaseModel):
    """User query for procurement intelligence analysis."""

    question: str = Field(min_length=5, description="The analysis question")
    agency_filter: str | None = Field(default=None, description="Filter by awarding agency name")
    date_range_years: int = Field(
        default=3, ge=1, le=10, description="How many years of data to analyze"
    )
    naics_filter: str | None = Field(default=None, description="Filter by NAICS code")


class ContractorSummary(BaseModel):
    """Aggregated summary for a single contractor."""

    name: str
    total_award_value: float
    contract_count: int
    win_rate: float = Field(ge=0.0, le=1.0, description="Win rate as a fraction")
    primary_agencies: list[str] = Field(default_factory=list)
    avg_contract_value: float = 0.0


class IntelligenceBrief(BaseModel):
    """Structured procurement intelligence report."""

    query: str
    executive_summary: str
    competitive_landscape: str
    top_contractors: list[ContractorSummary]
    spend_trends: str
    key_contracts: list[ContractAward]
    strategic_implications: str
    citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single search result from any retrieval method."""

    text: str
    score: float
    doc_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


def _nested_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Safely traverse nested dicts. Returns default if any key is missing."""
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current
