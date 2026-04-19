"""Tests for vector search wrappers."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest
from govintel.retrieval.vector import ChromaVectorStore, PineconeVectorStore

from govintel.ingestion.embedder import EmbeddingModel, embed_and_load


def test_chroma_vector_store_returns_ranked_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    vectors = {
        "Cybersecurity monitoring for DHS SOC operations": [1.0, 0.0],
        "Cloud migration services for Department of Defense": [0.0, 1.0],
        "information security services": [1.0, 0.0],
    }

    def fake_encode(self: EmbeddingModel, texts: list[str]) -> list[list[float]]:
        return [vectors[text] for text in texts]

    monkeypatch.setattr(EmbeddingModel, "encode", fake_encode)

    embed_and_load(
        chunks=[
            "Cybersecurity monitoring for DHS SOC operations",
            "Cloud migration services for Department of Defense",
        ],
        metadata=[
            {"award_id": "001", "chunk_index": 0},
            {"award_id": "002", "chunk_index": 0},
        ],
        model_name="all-MiniLM-L6-v2",
        chromadb_path=str(tmp_path / "chroma"),
        collection_name="contracts",
    )

    store = ChromaVectorStore(
        chromadb_path=str(tmp_path / "chroma"),
        collection_name="contracts",
        model_name="all-MiniLM-L6-v2",
    )

    results = store.search("information security services", top_k=2)

    assert len(results) == 2
    assert results[0].doc_id == "001:chunk:0"
    assert results[0].text == "Cybersecurity monitoring for DHS SOC operations"
    assert results[0].score > results[1].score


def test_chroma_vector_store_rejects_mixed_model_collection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    def fake_encode(self: EmbeddingModel, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(EmbeddingModel, "encode", fake_encode)

    embed_and_load(
        chunks=["Cybersecurity monitoring for DHS SOC operations"],
        metadata=[{"award_id": "001", "chunk_index": 0}],
        model_name="all-MiniLM-L6-v2",
        chromadb_path=str(tmp_path / "chroma"),
        collection_name="contracts",
    )

    with pytest.raises(ValueError, match="embedding model"):
        ChromaVectorStore(
            chromadb_path=str(tmp_path / "chroma"),
            collection_name="contracts",
            model_name="BAAI/bge-small-en-v1.5",
        )


def test_chroma_vector_store_returns_empty_for_blank_query(tmp_path: Any) -> None:
    store = ChromaVectorStore(
        chromadb_path=str(tmp_path / "chroma"),
        collection_name="missing-contracts",
        model_name="all-MiniLM-L6-v2",
        create_if_missing=True,
    )

    assert store.search("   ", top_k=3) == []


def test_pinecone_vector_store_returns_search_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeIndex:
        def query(
            self,
            *,
            vector: list[float],
            top_k: int,
            include_metadata: bool,
        ) -> Any:
            captured["vector"] = vector
            captured["top_k"] = top_k
            captured["include_metadata"] = include_metadata
            return SimpleNamespace(
                matches=[
                    SimpleNamespace(
                        id="001:chunk:0",
                        score=0.91,
                        metadata={
                            "text": "Cybersecurity monitoring for DHS SOC operations",
                            "award_id": "001",
                        },
                    )
                ]
            )

    class FakePineconeClient:
        def __init__(self, *, api_key: str) -> None:
            captured["api_key"] = api_key

        def Index(self, index_name: str) -> FakeIndex:  # noqa: N802
            captured["index_name"] = index_name
            return FakeIndex()

    def fake_encode(self: EmbeddingModel, texts: list[str]) -> list[list[float]]:
        assert texts == ["information security services"]
        return [[0.9, 0.1]]

    monkeypatch.setattr(EmbeddingModel, "encode", fake_encode)
    monkeypatch.setitem(
        sys.modules,
        "pinecone",
        SimpleNamespace(Pinecone=FakePineconeClient),
    )

    store = PineconeVectorStore(
        api_key="test-key",
        index_name="govintel",
        model_name="all-MiniLM-L6-v2",
    )

    results = store.search("information security services", top_k=3)

    assert captured == {
        "api_key": "test-key",
        "index_name": "govintel",
        "vector": [0.9, 0.1],
        "top_k": 3,
        "include_metadata": True,
    }
    assert len(results) == 1
    assert results[0].doc_id == "001:chunk:0"
    assert results[0].text == "Cybersecurity monitoring for DHS SOC operations"
    assert results[0].score == 0.91
    assert results[0].metadata == {"award_id": "001"}
