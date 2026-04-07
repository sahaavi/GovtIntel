"""Shared test fixtures and sample data."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient

from govintel.api.app import create_app
from govintel.models import ContractAward, ContractorSummary


@pytest.fixture
def app_client() -> TestClient:
    """Test client for the FastAPI application."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_contract() -> ContractAward:
    """A single sample contract award."""
    return ContractAward(
        award_id="CONT_AWD_001",
        recipient_name="Booz Allen Hamilton",
        awarding_agency="Department of Homeland Security",
        award_amount=5_000_000.0,
        start_date=date(2024, 1, 15),
        end_date=date(2026, 1, 14),
        naics_code="541512",
        description="Cybersecurity monitoring and incident response services",
        place_of_performance_state="VA",
        award_type="Definitive Contract",
    )


@pytest.fixture
def sample_usaspending_raw() -> dict[str, Any]:
    """Raw USAspending API response for a single award."""
    return {
        "generated_internal_id": "CONT_AWD_002",
        "recipient": {"recipient_name": "Leidos Inc."},
        "awarding_agency": {"toptier_agency": {"name": "Department of Defense"}},
        "total_obligation": 12_500_000.0,
        "period_of_performance_start_date": "2023-06-01",
        "period_of_performance_current_end_date": "2025-05-31",
        "naics_code": "541330",
        "description": "Engineering and technical support services",
        "place_of_performance": {"state_code": "MD"},
        "type_description": "Delivery Order",
    }


@pytest.fixture
def sample_contractor_summary() -> ContractorSummary:
    """A sample contractor summary."""
    return ContractorSummary(
        name="Booz Allen Hamilton",
        total_award_value=50_000_000.0,
        contract_count=42,
        win_rate=0.35,
        primary_agencies=["DHS", "DoD"],
        avg_contract_value=1_190_476.19,
    )
