"""Tests for embedding model wrapping and vector-store loading."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from govintel.ingestion import embedder
from govintel.ingestion.embedder import EmbeddingModel, embed_and_load


class FakeSentenceTransformer:
    """Tiny stand-in for sentence-transformers in unit tests."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id

    def encode(
        self,
        texts: list[str],
        *,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> list[list[float]]:
        assert show_progress_bar is False
        assert normalize_embeddings is True
        return [[float(index), float(index) + 0.5] for index, _ in enumerate(texts, start=1)]


def test_embedding_model_rejects_unsupported_name() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        EmbeddingModel("not-a-real-model")


def test_embedding_model_encodes_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embedder, "SentenceTransformer", FakeSentenceTransformer)

    model = EmbeddingModel("all-MiniLM-L6-v2")

    assert model.dimension == 384
    assert model.encode(["alpha", "beta"]) == [[1.0, 1.5], [2.0, 2.5]]


def test_embedding_model_returns_empty_list_for_empty_input() -> None:
    model = EmbeddingModel("all-MiniLM-L6-v2")
    assert model.encode([]) == []


def test_embed_and_load_rejects_length_mismatch(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="same length"):
        embed_and_load(
            chunks=["chunk one"],
            metadata=[{"award_id": "001"}, {"award_id": "002"}],
            model_name="all-MiniLM-L6-v2",
            chromadb_path=str(tmp_path / "chroma"),
            collection_name="contracts",
        )


def test_embed_and_load_requires_complete_pinecone_config(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="pinecone"):
        embed_and_load(
            chunks=["chunk one"],
            metadata=[{"award_id": "001"}],
            model_name="all-MiniLM-L6-v2",
            chromadb_path=str(tmp_path / "chroma"),
            collection_name="contracts",
            pinecone_api_key="test-key",
        )


def test_embed_and_load_stores_chunks_in_chromadb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    def fake_encode(self: EmbeddingModel, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(EmbeddingModel, "encode", fake_encode)

    collection = embed_and_load(
        chunks=["chunk one", "chunk two"],
        metadata=[
            {"award_id": "001", "chunk_index": 0},
            {"award_id": "002", "chunk_index": 1},
        ],
        model_name="all-MiniLM-L6-v2",
        chromadb_path=str(tmp_path / "chroma"),
        collection_name="contracts",
    )

    assert collection.count() == 2

    loaded = collection.get(ids=["001:chunk:0", "002:chunk:1"])

    assert loaded["documents"] == ["chunk one", "chunk two"]
    assert loaded["metadatas"] == [
        {"award_id": "001", "chunk_index": 0, "embedding_model": "all-MiniLM-L6-v2"},
        {"award_id": "002", "chunk_index": 1, "embedding_model": "all-MiniLM-L6-v2"},
    ]


def test_embed_and_load_falls_back_when_chunk_index_is_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    def fake_encode(self: EmbeddingModel, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(EmbeddingModel, "encode", fake_encode)

    collection = embed_and_load(
        chunks=["chunk one"],
        metadata=[{"award_id": "001", "chunk_index": None}],
        model_name="all-MiniLM-L6-v2",
        chromadb_path=str(tmp_path / "chroma"),
        collection_name="contracts",
    )

    loaded = collection.get(ids=["001:chunk:0"])
    assert loaded["documents"] == ["chunk one"]


def test_embed_and_load_rejects_mixed_model_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    def fake_encode(self: EmbeddingModel, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(EmbeddingModel, "encode", fake_encode)

    embed_and_load(
        chunks=["chunk one"],
        metadata=[{"award_id": "001", "chunk_index": 0}],
        model_name="all-MiniLM-L6-v2",
        chromadb_path=str(tmp_path / "chroma"),
        collection_name="contracts",
    )

    with pytest.raises(ValueError, match="embedding model"):
        embed_and_load(
            chunks=["chunk two"],
            metadata=[{"award_id": "002", "chunk_index": 0}],
            model_name="BAAI/bge-small-en-v1.5",
            chromadb_path=str(tmp_path / "chroma"),
            collection_name="contracts",
        )


def test_embed_and_load_mirrors_vectors_to_pinecone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    captured: dict[str, Any] = {}

    class FakeIndex:
        def upsert(self, *, vectors: list[dict[str, Any]]) -> None:
            captured["vectors"] = vectors

    class FakePineconeClient:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key

        def Index(self, index_name: str) -> FakeIndex:  # noqa: N802
            captured["index_name"] = index_name
            return FakeIndex()

    def fake_encode(self: EmbeddingModel, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(EmbeddingModel, "encode", fake_encode)
    monkeypatch.setitem(
        sys.modules,
        "pinecone",
        SimpleNamespace(Pinecone=FakePineconeClient),
    )

    embed_and_load(
        chunks=["chunk one"],
        metadata=[{"award_id": "001", "chunk_index": 0}],
        model_name="all-MiniLM-L6-v2",
        chromadb_path=str(tmp_path / "chroma"),
        collection_name="contracts",
        pinecone_api_key="test-key",
        pinecone_index_name="govintel",
    )

    assert captured["api_key"] == "test-key"
    assert captured["index_name"] == "govintel"
    assert captured["vectors"] == [
        {
            "id": "001:chunk:0",
            "values": [0.1, 0.2],
            "metadata": {
                "award_id": "001",
                "chunk_index": 0,
                "embedding_model": "all-MiniLM-L6-v2",
                "text": "chunk one",
            },
        }
    ]
