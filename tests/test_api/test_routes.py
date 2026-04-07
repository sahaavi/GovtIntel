"""Tests for API routes."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    def test_health_returns_ok(self, app_client: TestClient) -> None:
        response = app_client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAnalyzeEndpoint:
    def test_analyze_stub_returns_200(self, app_client: TestClient) -> None:
        response = app_client.post(
            "/api/v1/analyze",
            json={"question": "Who are the top DHS cybersecurity contractors?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "stub"
        assert "query" in data

    def test_analyze_with_filters(self, app_client: TestClient) -> None:
        response = app_client.post(
            "/api/v1/analyze",
            json={
                "question": "Cybersecurity spending trends analysis",
                "agency_filter": "DHS",
                "date_range_years": 5,
            },
        )
        assert response.status_code == 200

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
