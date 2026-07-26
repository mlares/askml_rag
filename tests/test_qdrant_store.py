import numpy as np
import pytest
from qdrant_client import QdrantClient

from askml_rag.models import Chunk, DocumentType, RetrievalFilters
from askml_rag.retrieval.qdrant_store import QdrantSemanticRetriever


class FakeEmbedder:
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        vectors = {
            "recommendation query": [1.0, 0.0],
            "Built a recommendation ranking model.": [0.9, 0.1],
            "Taught statistics at university.": [0.0, 1.0],
        }

        return np.asarray([vectors[sentence] for sentence in sentences])


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="example_source",
        chunk_index=0,
        document_content_hash="a" * 64,
        title="Example source",
        document_type="website",
        source_url="https://www.mlares.space/",
        topics=["example"],
        text=text,
    )


def test_qdrant_retriever_returns_most_similar_chunk() -> None:
    chunks = [
        make_chunk("ranking_chunk", "Built a recommendation ranking model."),
        make_chunk("teaching_chunk", "Taught statistics at university."),
    ]

    retriever = QdrantSemanticRetriever(
        QdrantClient(":memory:"),
        "test_chunks",
        FakeEmbedder(),
        model_name="fake-model",
    )
    retriever.index(chunks)

    results = retriever.search("recommendation query", limit=1)

    assert [chunk.chunk_id for chunk in results] == ["ranking_chunk"]
    assert results[0].topics == ["example"]


def test_qdrant_retriever_rejects_empty_index() -> None:
    retriever = QdrantSemanticRetriever(
        QdrantClient(":memory:"),
        "test_chunks",
        FakeEmbedder(),
        model_name="fake-model",
    )

    with pytest.raises(ValueError, match="empty"):
        retriever.index([])


def test_qdrant_retriever_returns_no_results_for_blank_query() -> None:
    retriever = QdrantSemanticRetriever(
        QdrantClient(":memory:"),
        "test_chunks",
        FakeEmbedder(),
        model_name="fake-model",
    )

    assert retriever.search("   ", limit=3) == []


def test_qdrant_retriever_filters_by_document_type_and_topic() -> None:
    chunks = [
        Chunk(
            chunk_id="project_chunk",
            document_id="project_source",
            chunk_index=0,
            document_content_hash="a" * 64,
            title="Recommendation project",
            document_type=DocumentType.project,
            year=2025,
            visibility="public",
            source_url="https://www.mlares.space/",
            topics=["recommendation-systems"],
            text="Built a recommendation ranking model.",
        ),
        Chunk(
            chunk_id="teaching_chunk",
            document_id="teaching_source",
            chunk_index=0,
            document_content_hash="b" * 64,
            title="Statistics course",
            document_type=DocumentType.teaching,
            year=2020,
            visibility="public",
            source_url="https://www.mlares.space/",
            topics=["statistics"],
            text="Taught statistics at university.",
        ),
    ]

    retriever = QdrantSemanticRetriever(
        QdrantClient(":memory:"),
        "test_chunks",
        FakeEmbedder(),
        model_name="fake-model",
    )
    retriever.index(chunks)

    results = retriever.search(
        "recommendation query",
        limit=3,
        filters=RetrievalFilters(
            document_types=[DocumentType.teaching],
            topics=["statistics"],
        ),
    )

    assert [chunk.chunk_id for chunk in results] == ["teaching_chunk"]
