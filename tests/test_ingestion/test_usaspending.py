"""Tests for the USAspending award search client."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from govintel.config import Settings
from govintel.ingestion import usaspending
from govintel.ingestion.usaspending import USAspendingClient


def _client_with_transport(
    handler: httpx.MockTransport | httpx.AsyncBaseTransport,
) -> USAspendingClient:
    settings = Settings(usaspending_base_url="https://example.test/api/v2")
    client = USAspendingClient(settings)
    client._client = httpx.AsyncClient(  # noqa: SLF001
        base_url=settings.usaspending_base_url,
        transport=handler,
        headers={"Content-Type": "application/json"},
    )
    return client


@pytest.mark.asyncio
async def test_search_awards_posts_required_contract_fields() -> None:
    captured_payload: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "generated_internal_id": "CONT_AWD_001",
                        "Award ID": "70FA3024C00000001",
                        "Recipient Name": "Example Corp",
                        "Awarding Agency": "Department of Homeland Security",
                        "Award Amount": 1_250_000.0,
                        "Start Date": "2024-01-01",
                        "End Date": "2025-01-01",
                        "NAICS": "541512",
                        "Description": "Cybersecurity services",
                        "Contract Award Type": "Definitive Contract",
                    }
                ],
                "page_metadata": {"page": 1, "hasNext": False},
            },
        )

    client = _client_with_transport(httpx.MockTransport(handler))

    results = await client.search_awards(
        filters=usaspending.build_contract_award_filters(naics_code="541512"),
        page=1,
        limit=10,
    )

    assert len(results) == 1
    assert captured_payload["subawards"] is False
    assert captured_payload["filters"]["award_type_codes"] == ["A", "B", "C", "D"]
    assert captured_payload["filters"]["naics_codes"] == {"require": ["541512"]}
    assert "Description" in captured_payload["fields"]
    assert "generated_internal_id" in captured_payload["fields"]
    assert captured_payload["sort"] == "Award Amount"
    assert captured_payload["order"] == "desc"

    await client.close()


@pytest.mark.asyncio
async def test_fetch_all_awards_maps_flat_search_results_and_stops_on_page_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    sleeps: list[float] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode())
        requests.append(payload)
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "generated_internal_id": "CONT_AWD_001",
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
                ],
                "page_metadata": {"page": payload["page"], "hasNext": False},
            },
        )

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    client = _client_with_transport(httpx.MockTransport(handler))

    awards = await client.fetch_all_awards(
        filters=usaspending.build_contract_award_filters(naics_code="541512"),
        max_pages=5,
        page_size=1,
    )

    assert len(awards) == 1
    assert awards[0].award_id == "CONT_AWD_001"
    assert awards[0].recipient_name == "Example Corp"
    assert awards[0].awarding_agency == "Department of Homeland Security"
    assert awards[0].naics_code == "541512"
    assert awards[0].description == "Cybersecurity services"
    assert len(requests) == 1
    assert sleeps == []

    await client.close()


@pytest.mark.asyncio
async def test_fetch_all_awards_returns_empty_list_when_no_results() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "page_metadata": {"hasNext": False}})

    client = _client_with_transport(httpx.MockTransport(handler))

    awards = await client.fetch_all_awards(
        filters=usaspending.build_contract_award_filters(),
        max_pages=3,
        page_size=10,
    )

    assert awards == []

    await client.close()
