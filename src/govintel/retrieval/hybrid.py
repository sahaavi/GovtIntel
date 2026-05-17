"""Hybrid BM25 + vector retrieval with cross-encoder reranking."""

from __future__ import annotations

from collections import deque
from typing import Protocol

from govintel.models import SearchResult

DEFAULT_CANDIDATE_POOL_SIZE = 20


class SearchBackend(Protocol):
    """Shared interface for first-stage retrieval backends."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return candidate search results."""


class Reranker(Protocol):
    """Shared interface for second-stage rerankers."""

    def rerank(self, query: str, candidates: list[str], top_k: int) -> list[SearchResult]:
        """Return reranked candidate texts."""


class HybridRetriever:
    """Combine lexical and semantic recall, then rerank for precision."""

    def __init__(
        self,
        *,
        bm25_index: SearchBackend,
        vector_store: SearchBackend,
        reranker: Reranker,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    ) -> None:
        if candidate_pool_size <= 0:
            raise ValueError("candidate_pool_size must be positive")

        self._bm25_index = bm25_index
        self._vector_store = vector_store
        self._reranker = reranker
        self._candidate_pool_size = candidate_pool_size

    def retrieve(self, query: str, top_k: int) -> list[SearchResult]:
        """Run BM25 and vector search, dedupe, rerank, and return top-k results."""

        if not query.strip() or top_k <= 0:
            return []

        bm25_results = self._bm25_index.search(query, top_k=self._candidate_pool_size)
        vector_results = self._vector_store.search(query, top_k=self._candidate_pool_size)
        candidates = _deduplicate_results(
            [
                *[_with_source(result, "bm25") for result in bm25_results],
                *[_with_source(result, "vector") for result in vector_results],
            ]
        )

        if not candidates:
            return []

        reranked = self._reranker.rerank(
            query,
            [candidate.text for candidate in candidates],
            top_k=len(candidates),
        )

        return _restore_candidate_context(candidates, reranked)[:top_k]


def _with_source(result: SearchResult, source: str) -> SearchResult:
    """Attach retrieval-source metadata without mutating backend results."""

    metadata = dict(result.metadata)
    sources = _metadata_sources(metadata)
    if source not in sources:
        sources.append(source)
    metadata["retrieval_sources"] = sources
    metadata[f"{source}_score"] = result.score
    return result.model_copy(update={"metadata": metadata})


def _metadata_sources(metadata: dict[str, object]) -> list[str]:
    """Read retrieval source metadata defensively."""

    raw_sources = metadata.get("retrieval_sources")
    if isinstance(raw_sources, list):
        return [str(source) for source in raw_sources]
    if isinstance(raw_sources, str):
        return [raw_sources]
    return []


def _deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate by document ID, falling back to text for ID-less results."""

    merged: list[SearchResult] = []
    index_by_key: dict[str, int] = {}

    for result in results:
        key = result.doc_id or result.text
        if key not in index_by_key:
            index_by_key[key] = len(merged)
            merged.append(result)
            continue

        existing_index = index_by_key[key]
        existing = merged[existing_index]
        merged[existing_index] = _merge_duplicate(existing, result)

    return merged


def _merge_duplicate(existing: SearchResult, duplicate: SearchResult) -> SearchResult:
    """Merge duplicate metadata and retain the strongest first-stage score."""

    metadata = dict(existing.metadata)
    duplicate_metadata = dict(duplicate.metadata)

    sources = _metadata_sources(metadata)
    for source in _metadata_sources(duplicate_metadata):
        if source not in sources:
            sources.append(source)
    metadata["retrieval_sources"] = sources

    for key, value in duplicate_metadata.items():
        if key == "retrieval_sources":
            continue
        if key not in metadata:
            metadata[key] = value

    return existing.model_copy(
        update={
            "score": max(existing.score, duplicate.score),
            "metadata": metadata,
        }
    )


def _restore_candidate_context(
    candidates: list[SearchResult],
    reranked: list[SearchResult],
) -> list[SearchResult]:
    """Map reranked texts back to original result IDs and metadata."""

    candidates_by_text: dict[str, deque[SearchResult]] = {}
    for candidate in candidates:
        candidates_by_text.setdefault(candidate.text, deque()).append(candidate)

    restored: list[SearchResult] = []
    for ranked in reranked:
        queue = candidates_by_text.get(ranked.text)
        if not queue:
            restored.append(ranked)
            continue

        candidate = queue.popleft()
        metadata = dict(candidate.metadata)
        metadata["first_stage_score"] = candidate.score
        metadata["reranker_score"] = ranked.score
        metadata.update(ranked.metadata)
        restored.append(
            candidate.model_copy(
                update={
                    "score": ranked.score,
                    "metadata": metadata,
                }
            )
        )

    return restored


__all__ = ["HybridRetriever", "DEFAULT_CANDIDATE_POOL_SIZE"]
