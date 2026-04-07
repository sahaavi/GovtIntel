import pytest
from govintel.ingestion.loader import ContractLoader
from govintel.models import ContractAward
from datetime import date

@pytest.fixture
def sample_contracts():
    return [
        ContractAward(
            award_id="TEST_001",
            recipient_name="Test Corp",
            awarding_agency="DHS",
            award_amount=1_000_000.0,
            start_date=date(2024, 1, 1),
            naics_code="541512",
            description="Test cybersecurity contract",
        )
    ]

def test_loader_inserts_and_queries(sample_contracts, test_db_session):
    loader = ContractLoader(session=test_db_session)
    loader.load(sample_contracts)
    results = loader.query(agency="DHS")
    assert len(results) == 1
    assert results[0].recipient_name == "Test Corp"