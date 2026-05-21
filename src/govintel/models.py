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

        Handles both nested award-detail responses and flat
        /search/spending_by_award rows.
        """
        return cls(
            award_id=_first_present(raw, "generated_internal_id", "Award ID", "id", default=""),
            recipient_name=_first_present(
                raw,
                "Recipient Name",
                default=_nested_get(raw, ["recipient", "recipient_name"], "Unknown"),
            ),
            awarding_agency=_first_present(
                raw,
                "Awarding Agency",
                default=_nested_get(raw, ["awarding_agency", "toptier_agency", "name"], "Unknown"),
            ),
            award_amount=float(
                _first_present(raw, "Award Amount", "total_obligation", default=0) or 0
            ),
            start_date=_first_present(
                raw,
                "Start Date",
                "Period of Performance Start Date",
                "period_of_performance_start_date",
                default="2000-01-01",
            ),
            end_date=_first_present(
                raw,
                "End Date",
                "Period of Performance Current End Date",
                "period_of_performance_current_end_date",
            ),
            naics_code=_naics_code(_first_present(raw, "NAICS", "naics_code", default="")),
            description=_first_present(raw, "Description", "description", default=""),
            place_of_performance_state=_first_present(
                raw,
                "Place of Performance State Code",
                default=_nested_get(raw, ["place_of_performance", "state_code"], ""),
            ),
            award_type=_first_present(
                raw,
                "Contract Award Type",
                "Award Type",
                "type_description",
                default="",
            ),
        )


class AnalysisQuery(BaseModel):
    """User query for procurement intelligence analysis."""

    question: str = Field(min_length=5, max_length=500, description="The analysis question")
    agency_filter: str | None = Field(
        default=None,
        max_length=120,
        description="Filter by awarding agency name",
    )
    date_range_years: int = Field(
        default=3, ge=1, le=10, description="How many years of data to analyze"
    )
    naics_filter: str | None = Field(
        default=None,
        pattern=r"^\d{2,6}$",
        description="Filter by NAICS code",
    )


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


def _first_present(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first non-empty value from a raw USAspending payload."""
    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value
    return default


def _naics_code(value: Any) -> str:
    """Normalize USAspending NAICS values to the code string."""

    if isinstance(value, dict):
        return str(value.get("code", "") or "")
    return str(value or "")
