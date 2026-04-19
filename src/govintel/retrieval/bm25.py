"""BM25 keyword retrieval over a fixed document set."""

from __future__ import annotations

import re
from collections import Counter

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from govintel.models import SearchResult

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split text into simple word tokens."""

    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """In-memory BM25 index for exact-term keyword retrieval."""

    def __init__(self, documents: list[str]) -> None:
        self._documents = tuple(documents)
        self._tokenized_documents = [_tokenize(document) for document in self._documents]
        self._document_term_sets = [set(tokens) for tokens in self._tokenized_documents]

        if not self._documents or not any(self._tokenized_documents):
            raise ValueError("documents must contain at least one tokenized document")

        self._index = BM25Okapi(self._tokenized_documents)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return the top keyword matches for the query."""

        if not query.strip() or top_k <= 0:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        query_token_counts = Counter(tokens)
        scores = self._index.get_scores(tokens)
        query_terms = set(query_token_counts)
        ranked_indices = sorted(
            range(len(self._documents)),
            key=lambda index: _ranking_score(
                overlap_count=_overlap_count(self._document_term_sets[index], query_terms),
                exact_term_set_match=_is_exact_term_match(
                    self._document_term_sets[index],
                    query_terms,
                ),
                exact_token_match=_is_exact_token_match(
                    self._tokenized_documents[index],
                    query_token_counts,
                ),
                raw_bm25_score=float(scores[index]),
            ),
            reverse=True,
        )

        results: list[SearchResult] = []
        for index in ranked_indices:
            raw_bm25_score = float(scores[index])
            score = _ranking_score(
                overlap_count=_overlap_count(self._document_term_sets[index], query_terms),
                exact_term_set_match=_is_exact_term_match(
                    self._document_term_sets[index],
                    query_terms,
                ),
                exact_token_match=_is_exact_token_match(
                    self._tokenized_documents[index],
                    query_token_counts,
                ),
                raw_bm25_score=raw_bm25_score,
            )
            if _overlap_count(self._document_term_sets[index], query_terms) == 0:
                continue

            results.append(
                SearchResult(
                    text=self._documents[index],
                    score=score,
                    doc_id=str(index),
                    metadata={
                        "doc_index": index,
                        "bm25_score": raw_bm25_score,
                    },
                )
            )

            if len(results) == top_k:
                break

        return results


__all__ = ["BM25Index"]


def _overlap_count(document_terms: set[str], query_terms: set[str]) -> int:
    """Count distinct query terms matched by the document."""

    return len(document_terms & query_terms)


def _is_exact_term_match(document_terms: set[str], query_terms: set[str]) -> int:
    """Prefer documents whose unique term set exactly matches the query terms."""

    return int(document_terms == query_terms)


def _is_exact_token_match(document_tokens: list[str], query_token_counts: Counter[str]) -> int:
    """Prefer documents whose token multiset exactly matches the query tokens."""

    return int(Counter(document_tokens) == query_token_counts)


def _ranking_score(
    *,
    overlap_count: int,
    exact_term_set_match: int,
    exact_token_match: int,
    raw_bm25_score: float,
) -> float:
    """Create a monotonic score that matches the returned rank order."""

    return (
        overlap_count * 1_000_000
        + exact_term_set_match * 1_000
        + exact_token_match * 100
        + raw_bm25_score
    )
