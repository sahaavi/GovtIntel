"""Tests for report generation orchestration."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from govintel.analysis.engine import SpendTrendPoint
from govintel.generation.report import (
    CitationValidationError,
    DatabaseContractRetriever,
    InsufficientEvidenceError,
    ReportGenerator,
    parse_intelligence_brief_text,
)
from govintel.ingestion.loader import contracts_table, metadata
from govintel.models import (
    AnalysisQuery,
    ContractorSummary,
    IntelligenceBrief,
    SearchResult,
)

VALID_BRIEF_TEXT = """
Executive Summary:
DHS cybersecurity awards are led by Booz Allen Hamilton based on CONT-001.

Competitive Landscape:
Booz Allen Hamilton is the leading contractor in the retrieved evidence.

Spend Trends:
Quarterly spend is concentrated in 2024-Q1.

Strategic Implications:
The cited award suggests continued demand for cyber monitoring.

Citations:
- CONT-001
"""

NUMBERED_BRIEF_TEXT = """
1. Executive Summary:
DHS cybersecurity awards are led by Booz Allen Hamilton based on CONT-001.

2. Competitive Landscape:
Booz Allen Hamilton is the leading contractor in the retrieved evidence.

3. Spend Trends:
Quarterly spend is concentrated in 2024-Q1.

4. Strategic Implications:
The cited award suggests continued demand for cyber monitoring.

5. Citations:
- CONT-001
"""


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[AnalysisQuery, int]] = []

    def retrieve(self, query: AnalysisQuery, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return [
            SearchResult(
                text="DHS-541512 Alpha Analytics Department of Homeland Security 541512",
                score=0.9,
                doc_id="DHS-541512:chunk:0",
                metadata={
                    "award_id": "DHS-541512",
                    "recipient_name": "Alpha Analytics",
                    "awarding_agency": "Department of Homeland Security",
                    "award_amount": 5_000_000.0,
                    "start_date": date(2024, 1, 15).isoformat(),
                    "naics_code": "541512",
                    "description": "Zero trust cybersecurity monitoring",
                },
            ),
            SearchResult(
                text="DOD-541512 Wrong Agency Department of Defense 541512",
                score=0.8,
                doc_id="DOD-541512:chunk:0",
                metadata={
                    "award_id": "DOD-541512",
                    "recipient_name": "Wrong Agency",
                    "awarding_agency": "Department of Defense",
                    "award_amount": 7_000_000.0,
                    "start_date": date(2024, 1, 17).isoformat(),
                    "naics_code": "541512",
                    "description": "Zero trust cybersecurity monitoring",
                },
            ),
        ][:top_k]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, candidates: list[str], top_k: int) -> list[SearchResult]:
        self.calls.append((query, candidates, top_k))
        return [
            SearchResult(
                text=candidate,
                score=float(len(candidates) - index),
                metadata={"candidate_index": index, "reranker": "fake-cross-encoder"},
            )
            for index, candidate in enumerate(candidates[:top_k])
        ]


class FakeAnalytics:
    def __init__(self) -> None:
        self.top_calls: list[tuple[str | None, int, int, str | None]] = []
        self.trend_calls: list[tuple[str | None, int, str | None]] = []
        self.hhi_calls: list[tuple[str | None, int, str | None]] = []

    async def top_contractors(
        self,
        *,
        agency: str | None,
        years: int,
        limit: int,
        naics_code: str | None = None,
    ) -> list[ContractorSummary]:
        self.top_calls.append((agency, years, limit, naics_code))
        return [
            ContractorSummary(
                name="Booz Allen Hamilton",
                total_award_value=5_000_000.0,
                contract_count=1,
                win_rate=1.0,
                primary_agencies=["Department of Homeland Security"],
                avg_contract_value=5_000_000.0,
            )
        ]

    async def spend_trend(
        self,
        *,
        agency: str | None,
        quarters: int,
        naics_code: str | None = None,
    ) -> list[SpendTrendPoint]:
        self.trend_calls.append((agency, quarters, naics_code))
        return [SpendTrendPoint(quarter="2024-Q1", total_spend=5_000_000.0, award_count=1)]

    async def market_hhi(
        self,
        *,
        agency: str | None,
        years: int,
        naics_code: str | None = None,
    ) -> float:
        self.hhi_calls.append((agency, years, naics_code))
        return 10_000.0


class FakePrompt:
    def __init__(self) -> None:
        self.render_calls: list[dict[str, Any]] = []

    def render(self, **variables: Any) -> dict[str, str]:
        self.render_calls.append(variables)
        return {
            "system": "system prompt",
            "user": f"context={variables['context']}\nanalytics={variables['analytics']}",
        }


class FakePromptLoader:
    def __init__(self, prompt: FakePrompt) -> None:
        self.prompt = prompt
        self.calls: list[tuple[str, str]] = []

    def __call__(self, name: str, version: str = "v1") -> FakePrompt:
        self.calls.append((name, version))
        return self.prompt


class FakeLLM:
    def __init__(self, response: str = VALID_BRIEF_TEXT) -> None:
        self.response = response
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system: str, user_message: str) -> str:
        self.calls.append((system, user_message))
        return self.response


def _retrieved_contract() -> SearchResult:
    return SearchResult(
        text="CONT-001 Booz Allen Hamilton $5M DHS cybersecurity monitoring",
        score=0.95,
        doc_id="CONT-001:chunk:0",
        metadata={
            "award_id": "CONT-001",
            "recipient_name": "Booz Allen Hamilton",
            "awarding_agency": "Department of Homeland Security",
            "award_amount": 5_000_000.0,
            "start_date": date(2024, 1, 15).isoformat(),
            "end_date": date(2026, 1, 14).isoformat(),
            "naics_code": "541512",
            "description": "Cybersecurity monitoring services",
            "place_of_performance_state": "VA",
            "award_type": "Definitive Contract",
        },
    )


def test_parse_intelligence_brief_text_accepts_numbered_section_headers() -> None:
    parsed = parse_intelligence_brief_text(NUMBERED_BRIEF_TEXT)

    assert parsed["executive_summary"].startswith("DHS cybersecurity awards")
    assert parsed["citations"] == ["CONT-001"]


@pytest.mark.asyncio
async def test_database_contract_retriever_uses_hybrid_stack_and_filters_contracts() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
        await conn.execute(
            contracts_table.insert(),
            [
                {
                    "award_id": "DHS-541512",
                    "recipient_name": "Alpha Analytics",
                    "awarding_agency": "Department of Homeland Security",
                    "award_amount": 5_000_000.0,
                    "start_date": date(2024, 1, 15),
                    "end_date": None,
                    "naics_code": "541512",
                    "description": "Zero trust cybersecurity monitoring",
                    "place_of_performance_state": "VA",
                    "award_type": "Definitive Contract",
                },
                {
                    "award_id": "DHS-541330",
                    "recipient_name": "Wrong NAICS",
                    "awarding_agency": "Department of Homeland Security",
                    "award_amount": 4_000_000.0,
                    "start_date": date(2024, 1, 16),
                    "end_date": None,
                    "naics_code": "541330",
                    "description": "Zero trust cybersecurity monitoring",
                    "place_of_performance_state": "VA",
                    "award_type": "Definitive Contract",
                },
                {
                    "award_id": "DOD-541512",
                    "recipient_name": "Wrong Agency",
                    "awarding_agency": "Department of Defense",
                    "award_amount": 7_000_000.0,
                    "start_date": date(2024, 1, 17),
                    "end_date": None,
                    "naics_code": "541512",
                    "description": "Zero trust cybersecurity monitoring",
                    "place_of_performance_state": "VA",
                    "award_type": "Definitive Contract",
                },
            ],
        )

    try:
        retriever = DatabaseContractRetriever(
            engine=engine,
            vector_store=FakeVectorStore(),
            reranker=FakeReranker(),
        )

        results = await retriever.retrieve(
            AnalysisQuery(
                question="zero trust cybersecurity",
                agency_filter="DHS",
                naics_filter="541512",
            ),
            top_k=5,
        )

        assert [result.doc_id for result in results] == ["DHS-541512"]
        assert results[0].metadata["retrieval_sources"] == ["bm25", "vector"]
        assert results[0].metadata["reranker"] == "fake-cross-encoder"
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_generate_returns_intelligence_brief_from_retrieval_and_generation() -> None:
    prompt = FakePrompt()
    generator = ReportGenerator(
        llm=FakeLLM(),
        retriever=FakeRetriever([_retrieved_contract()]),
        analytics=FakeAnalytics(),
        prompt_loader=FakePromptLoader(prompt),
    )

    brief = await generator.generate(
        query=AnalysisQuery(question="Top DHS cyber contractors", agency_filter="DHS"),
        strategy="zero_shot",
    )

    assert isinstance(brief, IntelligenceBrief)
    assert brief.executive_summary.startswith("DHS cybersecurity awards")
    assert brief.competitive_landscape.startswith("Booz Allen Hamilton")
    assert brief.spend_trends.startswith("Quarterly spend")
    assert brief.strategic_implications.startswith("The cited award")
    assert brief.citations == ["CONT-001"]
    assert [contract.award_id for contract in brief.key_contracts] == ["CONT-001"]
    assert brief.top_contractors[0].name == "Booz Allen Hamilton"


@pytest.mark.asyncio
async def test_generate_passes_query_filters_to_retriever_and_analytics() -> None:
    retriever = FakeRetriever([_retrieved_contract()])
    analytics = FakeAnalytics()
    generator = ReportGenerator(
        llm=FakeLLM(),
        retriever=retriever,
        analytics=analytics,
        prompt_loader=FakePromptLoader(FakePrompt()),
        retrieval_top_k=7,
    )

    await generator.generate(
        query=AnalysisQuery(
            question="Top DHS cyber contractors",
            agency_filter="DHS",
            date_range_years=5,
            naics_filter="541512",
        ),
        strategy="few_shot",
    )

    assert retriever.calls == [
        (
            AnalysisQuery(
                question="Top DHS cyber contractors",
                agency_filter="DHS",
                date_range_years=5,
                naics_filter="541512",
            ),
            7,
        )
    ]
    assert analytics.top_calls == [("DHS", 5, 5, "541512")]
    assert analytics.trend_calls == [("DHS", 20, "541512")]
    assert analytics.hhi_calls == [("DHS", 5, "541512")]


@pytest.mark.asyncio
async def test_generate_renders_prompt_with_retrieved_context_and_analytics() -> None:
    prompt = FakePrompt()
    generator = ReportGenerator(
        llm=FakeLLM(),
        retriever=FakeRetriever([_retrieved_contract()]),
        analytics=FakeAnalytics(),
        prompt_loader=FakePromptLoader(prompt),
    )

    await generator.generate(
        query=AnalysisQuery(question="Top DHS cyber contractors", agency_filter="DHS"),
        strategy="zero_shot",
    )

    variables = prompt.render_calls[0]
    assert variables["question"] == "Top DHS cyber contractors"
    assert variables["agency_filter"] == "DHS"
    assert variables["date_range_years"] == 3
    assert "CONT-001" in variables["context"]
    assert "Booz Allen Hamilton" in variables["context"]
    assert "$5,000,000.00" in variables["context"]
    assert "Top Contractors" in variables["analytics"]
    assert "HHI: 10000.00" in variables["analytics"]


@pytest.mark.asyncio
async def test_generate_calls_selected_llm_once_with_rendered_prompt() -> None:
    llm = FakeLLM()
    generator = ReportGenerator(
        llm=llm,
        retriever=FakeRetriever([_retrieved_contract()]),
        analytics=FakeAnalytics(),
        prompt_loader=FakePromptLoader(FakePrompt()),
    )

    await generator.generate(
        query=AnalysisQuery(question="Top DHS cyber contractors"),
        strategy="chain_of_thought",
    )

    assert llm.calls == [
        (
            "system prompt",
            "context=1. CONT-001 | Booz Allen Hamilton | Department of Homeland Security | "
            "$5,000,000.00 | 2024-01-15 | Cybersecurity monitoring services\n"
            "Evidence text: CONT-001 Booz Allen Hamilton $5M DHS cybersecurity monitoring\n"
            "analytics=Top Contractors:\n"
            "- Booz Allen Hamilton: $5,000,000.00 across 1 awards; avg $5,000,000.00; "
            "agencies: Department of Homeland Security\n"
            "\n"
            "Spend Trend:\n"
            "- 2024-Q1: $5,000,000.00 across 1 awards\n"
            "\n"
            "Market Concentration HHI: 10000.00",
        )
    ]


@pytest.mark.asyncio
async def test_generate_rejects_unretrieved_citations() -> None:
    bad_response = VALID_BRIEF_TEXT.replace("CONT-001", "CONT-999")
    generator = ReportGenerator(
        llm=FakeLLM(response=bad_response),
        retriever=FakeRetriever([_retrieved_contract()]),
        analytics=FakeAnalytics(),
        prompt_loader=FakePromptLoader(FakePrompt()),
    )

    with pytest.raises(CitationValidationError, match="CONT-999"):
        await generator.generate(query=AnalysisQuery(question="Top DHS cyber contractors"))


@pytest.mark.asyncio
async def test_generate_does_not_call_llm_when_retrieval_returns_no_evidence() -> None:
    llm = FakeLLM()
    generator = ReportGenerator(
        llm=llm,
        retriever=FakeRetriever([]),
        analytics=FakeAnalytics(),
        prompt_loader=FakePromptLoader(FakePrompt()),
    )

    with pytest.raises(InsufficientEvidenceError):
        await generator.generate(query=AnalysisQuery(question="Top DHS cyber contractors"))

    assert llm.calls == []
