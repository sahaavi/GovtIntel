"""Tests for API routes."""

from __future__ import annotations

from fastapi.testclient import TestClient

from govintel.api.dependencies import get_report_generator
from govintel.generation.gemini import LLMGenerationError
from govintel.generation.report import CitationValidationError
from govintel.models import AnalysisQuery, ContractAward, ContractorSummary, IntelligenceBrief


class TestHealthEndpoint:
    def test_health_returns_ok(self, app_client: TestClient) -> None:
        response = app_client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAnalyzeEndpoint:
    def test_analyze_returns_intelligence_brief_from_report_generator(
        self,
        app_client: TestClient,
        sample_contract: ContractAward,
        sample_contractor_summary: ContractorSummary,
    ) -> None:
        generator = FakeReportGenerator(
            IntelligenceBrief(
                query="Who are the top DHS cybersecurity contractors?",
                executive_summary="DHS cybersecurity awards are led by Booz Allen Hamilton.",
                competitive_landscape="Booz Allen Hamilton leads the retrieved market.",
                top_contractors=[sample_contractor_summary],
                spend_trends="Spend is concentrated in recent cybersecurity services awards.",
                key_contracts=[sample_contract],
                strategic_implications="The evidence supports continued demand.",
                citations=["CONT_AWD_001"],
                metadata={"strategy": "zero_shot"},
            )
        )
        app_client.app.dependency_overrides[get_report_generator] = lambda: generator

        response = app_client.post(
            "/api/v1/analyze",
            json={"question": "Who are the top DHS cybersecurity contractors?"},
        )

        app_client.app.dependency_overrides.clear()

        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "Who are the top DHS cybersecurity contractors?"
        assert data["executive_summary"].startswith("DHS cybersecurity awards")
        assert data["citations"] == ["CONT_AWD_001"]
        assert data["key_contracts"][0]["award_id"] == "CONT_AWD_001"

    def test_analyze_passes_request_body_to_report_generator(
        self,
        app_client: TestClient,
        sample_contract: ContractAward,
    ) -> None:
        generator = FakeReportGenerator(
            IntelligenceBrief(
                query="Cybersecurity spending trends analysis",
                executive_summary="Summary",
                competitive_landscape="Landscape",
                top_contractors=[],
                spend_trends="Trends",
                key_contracts=[sample_contract],
                strategic_implications="Implications",
                citations=["CONT_AWD_001"],
            )
        )
        app_client.app.dependency_overrides[get_report_generator] = lambda: generator

        response = app_client.post(
            "/api/v1/analyze",
            json={
                "question": "Cybersecurity spending trends analysis",
                "agency_filter": "DHS",
                "date_range_years": 5,
                "naics_filter": "541512",
            },
        )

        app_client.app.dependency_overrides.clear()

        assert response.status_code == 200
        assert generator.calls == [
            (
                AnalysisQuery(
                    question="Cybersecurity spending trends analysis",
                    agency_filter="DHS",
                    date_range_years=5,
                    naics_filter="541512",
                ),
                "zero_shot",
            )
        ]

    def test_analyze_rejects_short_question(self, app_client: TestClient) -> None:
        response = app_client.post(
            "/api/v1/analyze",
            json={"question": "Hi"},
        )
        assert response.status_code == 422

    def test_analyze_rejects_invalid_date_range(self, app_client: TestClient) -> None:
        response = app_client.post(
            "/api/v1/analyze",
            json={"question": "Valid question here", "date_range_years": 15},
        )
        assert response.status_code == 422

    def test_analyze_returns_502_when_generation_provider_fails(
        self,
        app_client: TestClient,
    ) -> None:
        app_client.app.dependency_overrides[get_report_generator] = lambda: FailingReportGenerator(
            LLMGenerationError("provider unavailable")
        )

        response = app_client.post(
            "/api/v1/analyze",
            json={"question": "Who are the top DHS cybersecurity contractors?"},
        )

        app_client.app.dependency_overrides.clear()

        assert response.status_code == 502
        assert response.json()["detail"] == "Generation provider failed"

    def test_analyze_returns_422_when_citation_validation_fails(
        self,
        app_client: TestClient,
    ) -> None:
        app_client.app.dependency_overrides[get_report_generator] = lambda: FailingReportGenerator(
            CitationValidationError("unsupported citation CONT-999")
        )

        response = app_client.post(
            "/api/v1/analyze",
            json={"question": "Who are the top DHS cybersecurity contractors?"},
        )

        app_client.app.dependency_overrides.clear()

        assert response.status_code == 422
        assert response.json()["detail"] == "unsupported citation CONT-999"


class FakeReportGenerator:
    def __init__(self, brief: IntelligenceBrief) -> None:
        self.brief = brief
        self.calls: list[tuple[AnalysisQuery, str]] = []

    async def generate(
        self,
        query: AnalysisQuery,
        strategy: str = "zero_shot",
    ) -> IntelligenceBrief:
        self.calls.append((query, strategy))
        return self.brief.model_copy(update={"query": query.question})


class FailingReportGenerator:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def generate(
        self,
        query: AnalysisQuery,
        strategy: str = "zero_shot",
    ) -> IntelligenceBrief:
        raise self.error
