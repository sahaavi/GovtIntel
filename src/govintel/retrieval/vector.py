"""Vector search wrappers for ChromaDB and Pinecone."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import chromadb

from govintel.ingestion.embedder import EmbeddingModel
from govintel.models import SearchResult


def _distance_to_score(distance: float) -> float:
    """Convert a Chroma distance into a higher-is-better score."""

    return 1.0 / (1.0 + max(distance, 0.0))


def _match_value(match: Any, key: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a plain dict."""

    if isinstance(match, dict):
        return match.get(key, default)
    return getattr(match, key, default)


class ChromaVectorStore:
    """Query a ChromaDB collection with the same embedding model used at ingest time."""

    def __init__(
        self,
        *,
        chromadb_path: str,
        collection_name: str,
        model_name: str,
        create_if_missing: bool = False,
    ) -> None:
        self._embedder = EmbeddingModel(model_name)
        self._client = chromadb.PersistentClient(path=chromadb_path)
        if create_if_missing:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "embedding_model": model_name,
                    "hnsw:space": "cosine",
                },
            )
        else:
            self._collection = self._client.get_collection(name=collection_name)

        metadata = self._collection.metadata or {}
        existing_model = metadata.get("embedding_model")
        if existing_model and existing_model != model_name:
            raise ValueError(
                f"Collection '{collection_name}' already uses embedding model '{existing_model}', "
                f"not '{model_name}'"
            )

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return the most semantically similar chunks from ChromaDB."""

        if not query.strip() or top_k <= 0:
            return []

        query_vector = self._embedder.encode([query])[0]
        response = self._collection.query(
            query_embeddings=cast(Any, [query_vector]),
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = cast(list[list[str]], response.get("ids") or [[]])
        documents = cast(list[list[str]], response.get("documents") or [[]])
        metadatas = cast(list[list[Mapping[str, Any]]], response.get("metadatas") or [[]])
        distances = cast(list[list[float]], response.get("distances") or [[]])

        results: list[SearchResult] = []
        for doc_id, text, metadata, distance in zip(
            ids[0],
            documents[0],
            metadatas[0],
            distances[0],
        ):
            results.append(
                SearchResult(
                    text=text,
                    score=_distance_to_score(float(distance)),
                    doc_id=str(doc_id),
                    metadata=dict(metadata or {}),
                )
            )

        return results


class PineconeVectorStore:
    """Query a Pinecone index using the shared embedding wrapper."""

    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        model_name: str,
    ) -> None:
        from pinecone import Pinecone  # type: ignore[import-untyped]

        self._embedder = EmbeddingModel(model_name)
        self._index = Pinecone(api_key=api_key).Index(index_name)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return the most similar matches from Pinecone."""

        if not query.strip() or top_k <= 0:
            return []

        query_vector = self._embedder.encode([query])[0]
        response = self._index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
        )

        matches = _match_value(response, "matches", []) or []
        results: list[SearchResult] = []

        for match in matches:
            raw_metadata = _match_value(match, "metadata", {}) or {}
            metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
            text = str(metadata.pop("text", ""))
            results.append(
                SearchResult(
                    text=text,
                    score=float(_match_value(match, "score", 0.0) or 0.0),
                    doc_id=str(_match_value(match, "id", "")),
                    metadata=metadata,
                )
            )

        return results


__all__ = ["ChromaVectorStore", "PineconeVectorStore"]
