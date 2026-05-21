"""Report generation orchestration and citation guardrails."""

from __future__ import annotations

import inspect
import re
from collections.abc import Awaitable, Callable, Sequence
from datetime import date
from typing import Any, Protocol, TypeVar, cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from govintel.analysis.engine import SpendTrendPoint, apply_agency_filter
from govintel.generation.prompts import PromptTemplate, load_prompt
from govintel.ingestion.loader import contracts_table
from govintel.models import (
    AnalysisQuery,
    ContractAward,
    ContractorSummary,
    IntelligenceBrief,
    SearchResult,
)
from govintel.retrieval.bm25 import BM25Index
from govintel.retrieval.hybrid import HybridRetriever

DEFAULT_RETRIEVAL_TOP_K = 5
DEFAULT_TOP_CONTRACTORS_LIMIT = 5
CONTRACT_ID_PATTERN = re.compile(r"\b[A-Z]{2,}[A-Z0-9_-]*\d[A-Z0-9_-]*\b")


class CitationValidationError(ValueError):
    """Raised when generated citations are not grounded in retrieved evidence."""


class InsufficientEvidenceError(ValueError):
    """Raised when retrieval returns no evidence for generation."""


class ReportParsingError(ValueError):
    """Raised when LLM output cannot be parsed into an IntelligenceBrief."""


class LLMClient(Protocol):
    """Generation client interface consumed by the report generator."""

    def generate(self, system: str, user_message: str) -> Awaitable[str]:
        """Generate text from rendered prompt messages."""


class Retriever(Protocol):
    """Retrieval interface consumed by the report generator."""

    def retrieve(
        self,
        query: AnalysisQuery,
        top_k: int,
    ) -> list[SearchResult] | Awaitable[list[SearchResult]]:
        """Return relevant evidence for the query."""


class Analytics(Protocol):
    """Analytics interface consumed by the report generator."""

    def top_contractors(
        self,
        *,
        agency: str | None,
        years: int,
        limit: int,
        naics_code: str | None = None,
    ) -> Awaitable[list[ContractorSummary]]:
        """Return ranked contractor summaries."""

    def spend_trend(
        self,
        *,
        agency: str | None,
        quarters: int,
        naics_code: str | None = None,
    ) -> Awaitable[list[SpendTrendPoint]]:
        """Return quarterly spend aggregates."""

    def market_hhi(
        self,
        *,
        agency: str | None,
        years: int,
        naics_code: str | None = None,
    ) -> Awaitable[float]:
        """Return market concentration HHI."""


PromptLoader = Callable[[str, str], PromptTemplate]
T = TypeVar("T")


class ReportGenerator:
    """Compose retrieval, analytics, prompts, and LLM output into a brief."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        retriever: Retriever,
        analytics: Analytics,
        prompt_loader: PromptLoader = load_prompt,
        prompt_version: str = "v1",
        retrieval_top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        top_contractors_limit: int = DEFAULT_TOP_CONTRACTORS_LIMIT,
    ) -> None:
        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")
        if top_contractors_limit <= 0:
            raise ValueError("top_contractors_limit must be positive")

        self._llm = llm
        self._retriever = retriever
        self._analytics = analytics
        self._prompt_loader = prompt_loader
        self._prompt_version = prompt_version
        self._retrieval_top_k = retrieval_top_k
        self._top_contractors_limit = top_contractors_limit

    async def generate(
        self,
        query: AnalysisQuery,
        strategy: str = "zero_shot",
    ) -> IntelligenceBrief:
        """Generate a citation-grounded procurement intelligence brief."""

        retrieved = await _maybe_await(self._retriever.retrieve(query, self._retrieval_top_k))
        if not retrieved:
            raise InsufficientEvidenceError("No retrieved contract evidence supports this query")

        top_contractors = await self._analytics.top_contractors(
            agency=query.agency_filter,
            years=query.date_range_years,
            limit=self._top_contractors_limit,
            naics_code=query.naics_filter,
        )
        trend = await self._analytics.spend_trend(
            agency=query.agency_filter,
            quarters=query.date_range_years * 4,
            naics_code=query.naics_filter,
        )
        hhi = await self._analytics.market_hhi(
            agency=query.agency_filter,
            years=query.date_range_years,
            naics_code=query.naics_filter,
        )

        prompt = self._prompt_loader(strategy, self._prompt_version)
        rendered = prompt.render(
            question=query.question,
            agency_filter=query.agency_filter,
            date_range_years=query.date_range_years,
            context=format_retrieved_context(retrieved),
            analytics=format_analytics_summary(top_contractors, trend, hhi),
        )

        response_text = await self._llm.generate(rendered["system"], rendered["user"])
        parsed = parse_intelligence_brief_text(response_text)
        key_contracts = validate_citations(parsed["citations"], retrieved)

        return IntelligenceBrief(
            query=query.question,
            executive_summary=parsed["executive_summary"],
            competitive_landscape=parsed["competitive_landscape"],
            top_contractors=top_contractors,
            spend_trends=parsed["spend_trends"],
            key_contracts=key_contracts,
            strategic_implications=parsed["strategic_implications"],
            citations=parsed["citations"],
            metadata={
                "strategy": strategy,
                "prompt_version": self._prompt_version,
                "retrieved_contract_count": len(retrieved),
                "market_hhi": hhi,
            },
        )


class DatabaseContractRetriever:
    """Hybrid retriever over contract rows loaded from the configured database."""

    def __init__(
        self,
        *,
        engine: AsyncEngine,
        vector_store: SearchBackend,
        reranker: RerankBackend,
        corpus_limit: int = 5_000,
    ) -> None:
        self._engine = engine
        self._corpus_limit = corpus_limit
        self._vector_store = vector_store
        self._reranker = reranker

    async def retrieve(self, query: AnalysisQuery, top_k: int) -> list[SearchResult]:
        """Load filtered contract rows, build hybrid indexes, and return matches."""

        if not query.question.strip() or top_k <= 0:
            return []

        statement = select(contracts_table).order_by(contracts_table.c.start_date.desc()).limit(
            self._corpus_limit
        )
        statement = apply_agency_filter(statement, query.agency_filter)
        if query.naics_filter:
            statement = statement.where(contracts_table.c.naics_code == query.naics_filter)
        statement = statement.where(
            contracts_table.c.start_date >= _years_ago(date.today(), query.date_range_years)
        )

        async with self._engine.connect() as conn:
            rows = (await conn.execute(statement)).mappings().all()

        base_results = [
            SearchResult(
                text=_format_contract_row(row),
                score=0.0,
                doc_id=str(row["award_id"]),
                metadata=_row_metadata(row),
            )
            for row in rows
        ]
        documents = [result.text for result in base_results]
        if not documents:
            return []

        try:
            bm25 = BM25Index(documents)
        except ValueError:
            return []

        bm25_backend = MetadataRestoringSearchBackend(base_results, bm25)
        allowed_award_ids = {result.doc_id for result in base_results}
        vector_backend = FilteredSearchBackend(self._vector_store, allowed_award_ids)
        retriever = HybridRetriever(
            bm25_index=bm25_backend,
            vector_store=vector_backend,
            reranker=self._reranker,
        )
        return retriever.retrieve(query.question, top_k=top_k)


class SearchBackend(Protocol):
    """Synchronous search backend used by the hybrid retriever."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return ranked search results."""


class RerankBackend(Protocol):
    """Synchronous reranker used by the hybrid retriever."""

    def rerank(self, query: str, candidates: list[str], top_k: int) -> list[SearchResult]:
        """Return reranked candidate texts."""


class MetadataRestoringSearchBackend:
    """Wrap BM25 results with authoritative contract metadata."""

    def __init__(self, base_results: Sequence[SearchResult], bm25_index: BM25Index) -> None:
        self._base_results = list(base_results)
        self._bm25_index = bm25_index

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Run BM25 and restore award IDs/metadata to returned hits."""

        results: list[SearchResult] = []
        for result in self._bm25_index.search(query, top_k=top_k):
            base_result = self._base_results[int(result.metadata["doc_index"])]
            metadata = dict(base_result.metadata)
            metadata["bm25_score"] = result.score
            results.append(
                SearchResult(
                    text=base_result.text,
                    score=result.score,
                    doc_id=base_result.doc_id,
                    metadata=metadata,
                )
            )
        return results


class FilteredSearchBackend:
    """Restrict an existing search backend to contract IDs allowed by query filters."""

    def __init__(self, backend: SearchBackend, allowed_award_ids: set[str]) -> None:
        self._backend = backend
        self._allowed_award_ids = allowed_award_ids

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return backend results whose authoritative award ID is in scope."""

        if top_k <= 0:
            return []

        oversampled = self._backend.search(query, top_k=max(top_k * 10, top_k))
        filtered: list[SearchResult] = []
        for result in oversampled:
            award_ids = _candidate_award_ids(result)
            if award_ids & self._allowed_award_ids:
                filtered.append(_normalize_search_result_id(result))
            if len(filtered) == top_k:
                break
        return filtered


def format_retrieved_context(results: Sequence[SearchResult]) -> str:
    """Render retrieved evidence into prompt context."""

    lines: list[str] = []
    for index, result in enumerate(results, start=1):
        award = _contract_from_result(result)
        lines.append(
            f"{index}. {award.award_id} | {award.recipient_name} | {award.awarding_agency} | "
            f"{_format_money(award.award_amount)} | {award.start_date.isoformat()} | "
            f"{award.description}"
        )
        lines.append(f"Evidence text: {result.text}")
    return "\n".join(lines)


def format_analytics_summary(
    top_contractors: Sequence[ContractorSummary],
    trend: Sequence[SpendTrendPoint],
    hhi: float,
) -> str:
    """Render SQL analytics into prompt context."""

    sections: list[str] = ["Top Contractors:"]
    if top_contractors:
        for contractor in top_contractors:
            agencies = ", ".join(contractor.primary_agencies) or "multiple agencies"
            sections.append(
                f"- {contractor.name}: {_format_money(contractor.total_award_value)} "
                f"across {contractor.contract_count} awards; "
                f"avg {_format_money(contractor.avg_contract_value)}; agencies: {agencies}"
            )
    else:
        sections.append("- No matching contractor awards found")

    sections.append("")
    sections.append("Spend Trend:")
    if trend:
        for point in trend:
            sections.append(
                f"- {point.quarter}: {_format_money(point.total_spend)} "
                f"across {point.award_count} awards"
            )
    else:
        sections.append("- No matching spend trend found")

    sections.append("")
    sections.append(f"Market Concentration HHI: {hhi:.2f}")
    return "\n".join(sections)


def parse_intelligence_brief_text(text: str) -> dict[str, Any]:
    """Parse headed LLM prose into IntelligenceBrief fields."""

    cleaned = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL | re.IGNORECASE)
    sections: dict[str, list[str]] = {
        "executive_summary": [],
        "competitive_landscape": [],
        "spend_trends": [],
        "strategic_implications": [],
        "citations": [],
    }
    current_section: str | None = None

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        section_name, remainder = _match_section_header(line)
        if section_name is not None:
            current_section = section_name
            if remainder:
                sections[current_section].append(remainder)
            continue

        if current_section is not None:
            sections[current_section].append(line)

    parsed: dict[str, Any] = {}
    for field in (
        "executive_summary",
        "competitive_landscape",
        "spend_trends",
        "strategic_implications",
    ):
        value = _clean_section_text(sections[field])
        if not value:
            raise ReportParsingError(f"Generated brief is missing {field.replace('_', ' ')}")
        parsed[field] = value

    citation_source = "\n".join(sections["citations"]) or cleaned
    parsed["citations"] = _extract_citation_ids(citation_source)
    return parsed


def validate_citations(
    citations: Sequence[str],
    retrieved_results: Sequence[SearchResult],
) -> list[ContractAward]:
    """Map cited IDs to retrieved contracts and reject unsupported citations."""

    if not citations:
        raise CitationValidationError("Generated brief must include at least one citation")

    result_by_award_id: dict[str, SearchResult] = {}
    for retrieved_result in retrieved_results:
        for award_id in _candidate_award_ids(retrieved_result):
            result_by_award_id[award_id] = retrieved_result

    key_contracts: list[ContractAward] = []
    seen: set[str] = set()
    unsupported: list[str] = []
    for citation in citations:
        if citation in seen:
            continue
        seen.add(citation)
        matched_result = result_by_award_id.get(citation)
        if matched_result is None:
            unsupported.append(citation)
            continue
        key_contracts.append(_contract_from_result(matched_result))

    if unsupported:
        raise CitationValidationError(
            "Generated brief cited unsupported contract IDs: " + ", ".join(unsupported)
        )

    return key_contracts


def _match_section_header(line: str) -> tuple[str | None, str]:
    """Return a normalized section name and inline content if a line is a header."""

    normalized = re.sub(r"^\s*(?:#+|-|\d+[.)])\s*", "", line).strip().strip("*").strip()
    labels = {
        "Executive Summary": "executive_summary",
        "Competitive Landscape": "competitive_landscape",
        "Spend Trends": "spend_trends",
        "Strategic Implications": "strategic_implications",
        "Citations": "citations",
    }

    for label, field in labels.items():
        match = re.match(rf"^{re.escape(label)}\s*:?\s*(.*)$", normalized, flags=re.IGNORECASE)
        if match:
            return field, match.group(1).strip().strip("*").strip()
    return None, ""


def _clean_section_text(lines: Sequence[str]) -> str:
    """Normalize section text without changing its meaning."""

    return " ".join(line.strip().lstrip("- ").strip() for line in lines).strip()


def _extract_citation_ids(text: str) -> list[str]:
    """Extract unique contract-like IDs in first-seen order."""

    citations: list[str] = []
    seen: set[str] = set()
    for match in CONTRACT_ID_PATTERN.finditer(text):
        citation = match.group(0)
        if citation not in seen:
            seen.add(citation)
            citations.append(citation)
    return citations


def _candidate_award_ids(result: SearchResult) -> set[str]:
    """Return award IDs implied by a retrieved result."""

    ids: set[str] = set()
    metadata_award_id = result.metadata.get("award_id")
    if metadata_award_id:
        ids.add(str(metadata_award_id))

    if result.doc_id:
        ids.add(result.doc_id.split(":chunk:", maxsplit=1)[0])

    return ids


def _normalize_search_result_id(result: SearchResult) -> SearchResult:
    """Return a search result keyed by authoritative award ID for deduplication."""

    metadata_award_id = result.metadata.get("award_id")
    if metadata_award_id:
        return result.model_copy(update={"doc_id": str(metadata_award_id)})
    if ":chunk:" in result.doc_id:
        return result.model_copy(update={"doc_id": result.doc_id.split(":chunk:", maxsplit=1)[0]})
    return result


def _contract_from_result(result: SearchResult) -> ContractAward:
    """Construct a ContractAward from retrieval metadata."""

    metadata = result.metadata
    award_id = str(metadata.get("award_id") or result.doc_id.split(":chunk:", maxsplit=1)[0])
    return ContractAward(
        award_id=award_id,
        recipient_name=str(metadata.get("recipient_name") or ""),
        awarding_agency=str(metadata.get("awarding_agency") or ""),
        award_amount=_float_value(metadata.get("award_amount")),
        start_date=_date_value(metadata.get("start_date")),
        end_date=_optional_date_value(metadata.get("end_date")),
        naics_code=str(metadata.get("naics_code") or ""),
        description=str(metadata.get("description") or result.text),
        place_of_performance_state=str(metadata.get("place_of_performance_state") or ""),
        award_type=str(metadata.get("award_type") or ""),
    )


def _format_contract_row(row: Any) -> str:
    """Render a database row as retrieval text."""

    metadata = _row_metadata(row)
    return (
        f"{metadata['award_id']} {metadata['recipient_name']} {metadata['awarding_agency']} "
        f"{metadata['award_amount']} {metadata['naics_code']} {metadata['description']} "
        f"{metadata['place_of_performance_state']} {metadata['award_type']}"
    )


def _row_metadata(row: Any) -> dict[str, Any]:
    """Convert a SQLAlchemy row mapping into SearchResult metadata."""

    end_date = _optional_date_value(row["end_date"])
    return {
        "award_id": str(row["award_id"]),
        "recipient_name": str(row["recipient_name"]),
        "awarding_agency": str(row["awarding_agency"]),
        "award_amount": float(row["award_amount"] or 0.0),
        "start_date": _date_value(row["start_date"]).isoformat(),
        "end_date": end_date.isoformat() if end_date else None,
        "naics_code": str(row["naics_code"] or ""),
        "description": str(row["description"] or ""),
        "place_of_performance_state": str(row["place_of_performance_state"] or ""),
        "award_type": str(row["award_type"] or ""),
    }


def _format_money(value: float) -> str:
    """Format money consistently for prompt context."""

    return f"${value:,.2f}"


def _float_value(value: object) -> float:
    """Coerce metadata values into floats."""

    if value in (None, ""):
        return 0.0
    if isinstance(value, (int, float, str)):
        return float(value)
    return float(str(value))


def _date_value(value: object) -> date:
    """Coerce metadata values into dates."""

    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        return date.fromisoformat(value)
    return date(2000, 1, 1)


def _optional_date_value(value: object) -> date | None:
    """Coerce optional metadata values into dates."""

    if value in (None, ""):
        return None
    return _date_value(value)


def _years_ago(value: date, years: int) -> date:
    """Subtract whole years while handling leap-day edge cases."""

    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)


async def _maybe_await(value: T | Awaitable[T]) -> T:
    """Await protocol results only when the implementation is async."""

    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return value


__all__ = [
    "CitationValidationError",
    "DatabaseContractRetriever",
    "InsufficientEvidenceError",
    "ReportGenerator",
    "ReportParsingError",
    "format_analytics_summary",
    "format_retrieved_context",
    "parse_intelligence_brief_text",
    "validate_citations",
]
