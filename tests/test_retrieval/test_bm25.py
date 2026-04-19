"""Tests for BM25 keyword retrieval."""

from __future__ import annotations

import pytest

from govintel.retrieval.bm25 import BM25Index


def test_bm25_index_finds_keyword_matches() -> None:
    docs = [
        "Cybersecurity monitoring for DHS SOC operations",
        "Cloud migration services for Department of Defense",
        "IT helpdesk support for GSA facilities",
    ]

    index = BM25Index(docs)
    results = index.search("DHS cybersecurity", top_k=2)

    assert len(results) == 1
    assert results[0].doc_id == "0"
    assert results[0].metadata["doc_index"] == 0
    assert "bm25_score" in results[0].metadata
    assert results[0].text == docs[0]


def test_bm25_returns_scores_in_descending_order() -> None:
    docs = [
        "cybersecurity contract services",
        "cybersecurity",
        "agriculture grant program",
    ]

    index = BM25Index(docs)
    results = index.search("cybersecurity", top_k=3)

    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_bm25_prefers_better_multi_term_match() -> None:
    docs = [
        "apple apple apple",
        "apple banana",
        "apple",
    ]

    index = BM25Index(docs)
    results = index.search("apple banana", top_k=3)

    assert len(results) == 3
    assert results[0].text == "apple banana"


def test_bm25_returns_empty_for_blank_query() -> None:
    index = BM25Index(["Cybersecurity monitoring for DHS SOC operations"])

    assert index.search("   ", top_k=3) == []


def test_bm25_returns_empty_for_non_positive_top_k() -> None:
    index = BM25Index(["Cybersecurity monitoring for DHS SOC operations"])

    assert index.search("cybersecurity", top_k=0) == []


def test_bm25_returns_exact_match_from_small_corpus() -> None:
    index = BM25Index(["banana", "apple"])

    results = index.search("apple", top_k=1)

    assert len(results) == 1
    assert results[0].text == "apple"
    assert results[0].doc_id == "1"


def test_bm25_prefers_exact_single_term_match() -> None:
    index = BM25Index(["apple", "apple banana", "banana"])

    results = index.search("apple", top_k=2)

    assert len(results) == 2
    assert results[0].text == "apple"


def test_bm25_prefers_exact_repeated_token_match() -> None:
    index = BM25Index(["apple banana", "apple apple banana", "banana"])

    results = index.search("apple apple banana", top_k=2)

    assert len(results) == 2
    assert results[0].text == "apple apple banana"


def test_bm25_scores_follow_returned_rank() -> None:
    index = BM25Index(
        [
            "apple banana",
            "apple banana cherry",
            "apple apple banana",
        ]
    )

    results = index.search("apple banana", top_k=3)

    assert len(results) == 3
    assert results[0].score >= results[1].score >= results[2].score


def test_bm25_rejects_empty_or_tokenless_corpus() -> None:
    with pytest.raises(ValueError, match="documents"):
        BM25Index([])

    with pytest.raises(ValueError, match="documents"):
        BM25Index(["", "   "])


def test_bm25_defensively_copies_input_documents() -> None:
    docs = [
        "apple banana",
        "banana only",
    ]
    index = BM25Index(docs)

    docs.append("apple")

    results = index.search("apple", top_k=1)

    assert len(results) == 1
    assert results[0].text == "apple banana"
