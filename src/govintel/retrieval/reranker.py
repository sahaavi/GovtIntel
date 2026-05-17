"""Cross-encoder reranking for second-stage retrieval precision."""

from __future__ import annotations

from typing import Any

from govintel.models import SearchResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CrossEncoder: Any | None = None


def _load_cross_encoder_class() -> Any:
    """Load the CrossEncoder class lazily to keep imports and tests lightweight."""

    global CrossEncoder
    if CrossEncoder is None:
        from sentence_transformers import CrossEncoder as LoadedCrossEncoder

        CrossEncoder = LoadedCrossEncoder
    return CrossEncoder


class CrossEncoderReranker:
    """Rank query/document pairs with a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def _get_model(self) -> Any:
        """Load and cache the underlying cross-encoder on first real use."""

        if self._model is None:
            self._model = _load_cross_encoder_class()(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[str], top_k: int) -> list[SearchResult]:
        """Return candidates sorted by query/document relevance score."""

        if not query.strip() or not candidates or top_k <= 0:
            return []

        pairs = [(query, candidate) for candidate in candidates]
        raw_scores = self._get_model().predict(pairs)
        scored_candidates = [
            (index, candidate, float(score))
            for index, (candidate, score) in enumerate(zip(candidates, raw_scores))
        ]

        ranked = sorted(scored_candidates, key=lambda item: item[2], reverse=True)

        return [
            SearchResult(
                text=candidate,
                score=score,
                doc_id=str(index),
                metadata={
                    "candidate_index": index,
                    "reranker_model": self.model_name,
                },
            )
            for index, candidate, score in ranked[:top_k]
        ]


__all__ = ["CrossEncoderReranker", "DEFAULT_RERANKER_MODEL"]
