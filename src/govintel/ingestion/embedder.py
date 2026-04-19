"""Embedding model wrapper and vector-store loading helpers.

This module owns document-side embedding and indexing. Query-time vector search
belongs in ``retrieval/vector.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import chromadb
from sentence_transformers import SentenceTransformer

MetadataScalar: TypeAlias = str | int | float | bool | None
MetadataListValue: TypeAlias = list[str | int | float | bool]
MetadataValue: TypeAlias = MetadataScalar | MetadataListValue


@dataclass(frozen=True)
class ModelSpec:
    """Static configuration for a supported embedding model."""

    model_id: str
    dimension: int


SUPPORTED_MODELS: dict[str, ModelSpec] = {
    "all-MiniLM-L6-v2": ModelSpec(
        model_id="all-MiniLM-L6-v2",
        dimension=384,
    ),
    "BAAI/bge-small-en-v1.5": ModelSpec(
        model_id="BAAI/bge-small-en-v1.5",
        dimension=384,
    ),
}


class EmbeddingModel:
    """Thin wrapper around sentence-transformers models."""

    def __init__(self, name: str) -> None:
        if name not in SUPPORTED_MODELS:
            supported = ", ".join(SUPPORTED_MODELS)
            raise ValueError(f"Unsupported embedding model: {name}. Supported models: {supported}")

        self.name = name
        self._spec = SUPPORTED_MODELS[name]
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        """Load and cache the underlying model on first use."""

        if self._model is None:
            self._model = SentenceTransformer(self._spec.model_id)
        return self._model

    @property
    def dimension(self) -> int:
        """Return the known embedding dimension for the configured model."""

        return self._spec.dimension

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Convert a batch of input strings into plain Python float vectors."""

        if not texts:
            return []

        vectors = self._get_model().encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return [list(map(float, vector)) for vector in vectors]


def _build_ids(metadata: list[dict[str, Any]]) -> list[str]:
    """Create deterministic chunk IDs for repeatable indexing."""

    ids: list[str] = []
    for index, item in enumerate(metadata):
        award_id = str(item.get("award_id", f"doc-{index}"))
        raw_chunk_index = item.get("chunk_index")
        chunk_index = index if raw_chunk_index is None else int(raw_chunk_index)
        ids.append(f"{award_id}:chunk:{chunk_index}")
    return ids


def _enrich_metadata(
    metadata: list[dict[str, Any]],
    *,
    model_name: str,
) -> list[dict[str, MetadataValue]]:
    """Copy metadata and attach indexing context used later for debugging."""

    enriched: list[dict[str, MetadataValue]] = []
    for item in metadata:
        record = {key: _normalize_metadata_value(value) for key, value in item.items()}
        record["embedding_model"] = model_name
        enriched.append(record)
    return enriched


def _normalize_metadata_value(value: Any) -> MetadataValue:
    """Convert metadata into Chroma-safe scalar or scalar-list values."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, list):
        normalized_items: list[str | int | float | bool] = []
        for item in value:
            if isinstance(item, (str, int, float, bool)):
                normalized_items.append(item)
            else:
                normalized_items.append(str(item))
        return normalized_items

    return str(value)


def _validate_pinecone_config(
    *,
    pinecone_api_key: str | None,
    pinecone_index_name: str | None,
) -> None:
    """Require that Pinecone configuration is either fully present or absent."""

    if bool(pinecone_api_key) != bool(pinecone_index_name):
        raise ValueError(
            "pinecone_api_key and pinecone_index_name must either both be "
            "provided or both be omitted"
        )


def embed_and_load(
    *,
    chunks: list[str],
    metadata: list[dict[str, Any]],
    model_name: str,
    chromadb_path: str,
    collection_name: str,
    pinecone_api_key: str | None = None,
    pinecone_index_name: str | None = None,
) -> Any:
    """Embed text chunks and upsert them into ChromaDB.

    If Pinecone configuration is supplied, the same vectors are mirrored there.
    """

    if not chunks:
        raise ValueError("chunks must not be empty")
    if len(chunks) != len(metadata):
        raise ValueError("chunks and metadata must have the same length")

    _validate_pinecone_config(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name=pinecone_index_name,
    )

    embedder = EmbeddingModel(model_name)
    ids = _build_ids(metadata)
    metadatas = _enrich_metadata(metadata, model_name=model_name)
    vectors = embedder.encode(chunks)

    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "embedding_model": model_name,
            "hnsw:space": "cosine",
        },
    )

    existing_model = collection.metadata.get("embedding_model") if collection.metadata else None
    if existing_model and existing_model != model_name:
        raise ValueError(
            f"Collection '{collection_name}' already uses embedding model '{existing_model}', "
            f"not '{model_name}'"
        )

    chroma_embeddings: list[Sequence[float] | Sequence[int]] = [vector for vector in vectors]
    chroma_metadatas: list[Mapping[str, MetadataValue]] = [record for record in metadatas]

    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=chroma_embeddings,
        metadatas=cast(Any, chroma_metadatas),
    )

    if pinecone_api_key and pinecone_index_name:
        _upsert_to_pinecone(
            ids=ids,
            chunks=chunks,
            vectors=vectors,
            metadata=metadatas,
            api_key=pinecone_api_key,
            index_name=pinecone_index_name,
        )

    return collection


def _upsert_to_pinecone(
    *,
    ids: list[str],
    chunks: list[str],
    vectors: list[list[float]],
    metadata: list[dict[str, Any]],
    api_key: str,
    index_name: str,
) -> None:
    """Mirror chunk vectors into Pinecone using the same IDs and metadata."""

    from pinecone import Pinecone  # type: ignore[import-untyped]

    pinecone = Pinecone(api_key=api_key)
    index = pinecone.Index(index_name)

    records: list[dict[str, Any]] = []
    for chunk_id, chunk_text, vector, item_metadata in zip(ids, chunks, vectors, metadata):
        record_metadata = dict(item_metadata)
        record_metadata["text"] = chunk_text
        records.append(
            {
                "id": chunk_id,
                "values": vector,
                "metadata": record_metadata,
            }
        )

    index.upsert(vectors=records)


__all__ = ["EmbeddingModel", "SUPPORTED_MODELS", "embed_and_load"]
