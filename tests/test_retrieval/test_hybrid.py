"""Tests for hybrid retrieval orchestration."""

from __future__ import annotations

from govintel.models import SearchResult
from govintel.retrieval.hybrid import HybridRetriever


class FakeRetriever:
    """Simple retriever stub that records calls."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeReranker:
    """Reranker stub that returns scores from a fixed lookup."""

    def __init__(self, score_by_text: dict[str, float]) -> None:
        self.score_by_text = score_by_text
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, candidates: list[str], top_k: int) -> list[SearchResult]:
        self.calls.append((query, candidates, top_k))
        ranked = sorted(
            candidates,
            key=lambda text: self.score_by_text[text],
            reverse=True,
        )
        return [
            SearchResult(text=text, score=self.score_by_text[text])
            for text in ranked[:top_k]
        ]


def test_hybrid_retriever_merges_deduplicates_and_reranks() -> None:
    bm25 = FakeRetriever(
        [
            SearchResult(text="keyword DHS cyber", score=10.0, doc_id="doc-1"),
            SearchResult(text="shared cloud contract", score=5.0, doc_id="doc-2"),
        ]
    )
    vector = FakeRetriever(
        [
            SearchResult(text="shared cloud contract", score=0.8, doc_id="doc-2"),
            SearchResult(text="semantic zero trust", score=0.7, doc_id="doc-3"),
        ]
    )
    rerank = FakeReranker(
        {
            "keyword DHS cyber": 0.3,
            "shared cloud contract": 0.7,
            "semantic zero trust": 0.95,
        }
    )

    retriever = HybridRetriever(bm25_index=bm25, vector_store=vector, reranker=rerank)

    results = retriever.retrieve("DHS zero trust", top_k=2)

    assert bm25.calls == [("DHS zero trust", 20)]
    assert vector.calls == [("DHS zero trust", 20)]
    assert rerank.calls == [
        (
            "DHS zero trust",
            ["keyword DHS cyber", "shared cloud contract", "semantic zero trust"],
            3,
        )
    ]

    assert [result.doc_id for result in results] == ["doc-3", "doc-2"]
    assert [result.text for result in results] == ["semantic zero trust", "shared cloud contract"]
    assert [result.score for result in results] == [0.95, 0.7]
    assert results[1].metadata["retrieval_sources"] == ["bm25", "vector"]
    assert results[1].metadata["bm25_score"] == 5.0
    assert results[1].metadata["vector_score"] == 0.8


def test_hybrid_retriever_respects_top_k() -> None:
    bm25 = FakeRetriever(
        [
            SearchResult(text="first", score=1.0, doc_id="doc-1"),
            SearchResult(text="second", score=1.0, doc_id="doc-2"),
        ]
    )
    vector = FakeRetriever([SearchResult(text="third", score=1.0, doc_id="doc-3")])
    rerank = FakeReranker({"first": 0.1, "second": 0.8, "third": 0.3})

    retriever = HybridRetriever(bm25_index=bm25, vector_store=vector, reranker=rerank)

    results = retriever.retrieve("query", top_k=1)

    assert len(results) == 1
    assert results[0].text == "second"


def test_hybrid_retriever_returns_empty_for_blank_query_without_calling_backends() -> None:
    bm25 = FakeRetriever([SearchResult(text="first", score=1.0, doc_id="doc-1")])
    vector = FakeRetriever([SearchResult(text="second", score=1.0, doc_id="doc-2")])
    rerank = FakeReranker({"first": 1.0, "second": 0.5})

    retriever = HybridRetriever(bm25_index=bm25, vector_store=vector, reranker=rerank)

    assert retriever.retrieve("   ", top_k=5) == []
    assert bm25.calls == []
    assert vector.calls == []
    assert rerank.calls == []
