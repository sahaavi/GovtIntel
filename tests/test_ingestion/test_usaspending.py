import pytest
from govintel.ingestion.usaspending import USASpendingClient

@pytest.fixture
def client():
    return USASpendingClient()

def test_fetch_awards_returns_contracts(client):
    """Integration test — hits real API. Mark slow."""
    awards = client.fetch_awards(naics_code="541512", limit=10)
    assert len(awards) > 0
    assert all(a.naics_code == "541512" for a in awards)

def test_fetch_awards_filters_by_agency(client):
    awards = client.fetch_awards(
        naics_code="541512",
        agency="Department of Homeland Security",
        limit=5,
    )
    assert all("Homeland" in a.awarding_agency for a in awards)