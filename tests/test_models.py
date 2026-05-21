"""Tests for Pydantic data models."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from govintel.models import (
    AnalysisQuery,
    ContractAward,
    ContractorSummary,
    IntelligenceBrief,
    SearchResult,
)


class TestContractAward:
    def test_create_with_required_fields(self, sample_contract: ContractAward) -> None:
        assert sample_contract.award_id == "CONT_AWD_001"
        assert sample_contract.recipient_name == "Booz Allen Hamilton"
        assert sample_contract.award_amount == 5_000_000.0

    def test_from_usaspending(self, sample_usaspending_raw: dict[str, Any]) -> None:
        award = ContractAward.from_usaspending(sample_usaspending_raw)

        assert award.award_id == "CONT_AWD_002"
        assert award.recipient_name == "Leidos Inc."
        assert award.awarding_agency == "Department of Defense"
        assert award.award_amount == 12_500_000.0
        assert award.start_date == date(2023, 6, 1)
        assert award.naics_code == "541330"
        assert award.place_of_performance_state == "MD"

    def test_from_usaspending_missing_nested_fields(self) -> None:
        raw: dict[str, Any] = {
            "generated_internal_id": "CONT_AWD_003",
            "total_obligation": 1000.0,
            "period_of_performance_start_date": "2024-01-01",
        }
        award = ContractAward.from_usaspending(raw)

        assert award.award_id == "CONT_AWD_003"
        assert award.recipient_name == "Unknown"
        assert award.awarding_agency == "Unknown"
        assert award.award_amount == 1000.0

    def test_from_usaspending_null_obligation(self) -> None:
        raw: dict[str, Any] = {
            "generated_internal_id": "CONT_AWD_004",
            "total_obligation": None,
            "period_of_performance_start_date": "2024-01-01",
        }
        award = ContractAward.from_usaspending(raw)
        assert award.award_amount == 0.0

    def test_from_usaspending_search_award_flat_fields(self) -> None:
        raw: dict[str, Any] = {
            "generated_internal_id": "CONT_AWD_005",
            "Award ID": "70FA3024C00000001",
            "Recipient Name": "Example Corp",
            "Awarding Agency": "Department of Homeland Security",
            "Award Amount": 1_250_000.0,
            "Start Date": "2024-01-01",
            "End Date": "2025-01-01",
            "NAICS": "541512",
            "Description": "Cybersecurity services",
            "Place of Performance State Code": "VA",
            "Contract Award Type": "Definitive Contract",
        }

        award = ContractAward.from_usaspending(raw)

        assert award.award_id == "CONT_AWD_005"
        assert award.recipient_name == "Example Corp"
        assert award.awarding_agency == "Department of Homeland Security"
        assert award.award_amount == 1_250_000.0
        assert award.start_date == date(2024, 1, 1)
        assert award.end_date == date(2025, 1, 1)
        assert award.naics_code == "541512"
        assert award.description == "Cybersecurity services"
        assert award.place_of_performance_state == "VA"
        assert award.award_type == "Definitive Contract"

    def test_from_usaspending_search_award_extracts_naics_code_object(self) -> None:
        raw: dict[str, Any] = {
            "generated_internal_id": "CONT_AWD_006",
            "Recipient Name": "Example Corp",
            "Awarding Agency": "Department of State",
            "Award Amount": 1_250_000.0,
            "Start Date": "2024-01-01",
            "NAICS": {"code": "541512", "description": "Computer systems design services"},
            "Description": "IT consolidation support",
        }

        award = ContractAward.from_usaspending(raw)

        assert award.naics_code == "541512"

    def test_serialization_roundtrip(self, sample_contract: ContractAward) -> None:
        data = sample_contract.model_dump()
        restored = ContractAward(**data)
        assert restored == sample_contract


class TestAnalysisQuery:
    def test_valid_query(self) -> None:
        query = AnalysisQuery(question="Who are the top DHS cybersecurity contractors?")
        assert query.question == "Who are the top DHS cybersecurity contractors?"
        assert query.agency_filter is None
        assert query.date_range_years == 3

    def test_query_with_filters(self) -> None:
        query = AnalysisQuery(
            question="Cybersecurity spending trends",
            agency_filter="DHS",
            date_range_years=5,
            naics_filter="541512",
        )
        assert query.agency_filter == "DHS"
        assert query.date_range_years == 5
        assert query.naics_filter == "541512"

    def test_question_too_short(self) -> None:
        with pytest.raises(Exception):
            AnalysisQuery(question="Hi")

    def test_question_too_long(self) -> None:
        with pytest.raises(Exception):
            AnalysisQuery(question="x" * 501)

    def test_date_range_bounds(self) -> None:
        with pytest.raises(Exception):
            AnalysisQuery(question="Valid question here", date_range_years=0)
        with pytest.raises(Exception):
            AnalysisQuery(question="Valid question here", date_range_years=11)

    def test_naics_filter_accepts_only_codes(self) -> None:
        AnalysisQuery(question="Valid question here", naics_filter="541512")

        with pytest.raises(Exception):
            AnalysisQuery(question="Valid question here", naics_filter="54%")


class TestContractorSummary:
    def test_create(self, sample_contractor_summary: ContractorSummary) -> None:
        assert sample_contractor_summary.name == "Booz Allen Hamilton"
        assert sample_contractor_summary.contract_count == 42
        assert sample_contractor_summary.win_rate == 0.35

    def test_win_rate_bounds(self) -> None:
        with pytest.raises(Exception):
            ContractorSummary(
                name="Bad",
                total_award_value=100.0,
                contract_count=1,
                win_rate=1.5,
            )


class TestIntelligenceBrief:
    def test_create_minimal(self, sample_contract: ContractAward) -> None:
        brief = IntelligenceBrief(
            query="Test query for analysis",
            executive_summary="Summary text",
            competitive_landscape="Landscape text",
            top_contractors=[],
            spend_trends="Trends text",
            key_contracts=[sample_contract],
            strategic_implications="Implications text",
        )
        assert brief.query == "Test query for analysis"
        assert len(brief.key_contracts) == 1
        assert brief.citations == []
        assert brief.metadata == {}


class TestSearchResult:
    def test_create(self) -> None:
        result = SearchResult(text="contract text", score=0.95, doc_id="doc_1")
        assert result.score == 0.95
        assert result.metadata == {}
