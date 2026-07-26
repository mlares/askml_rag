import numpy as np

from askml_rag.models import Chunk
from askml_rag.retrieval.semantic import SemanticRetriever


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
        topics=[],
        text=text,
    )


def test_semantic_retriever_returns_most_similar_chunk() -> None:
    chunks = [
        make_chunk("ranking_chunk", "Built a recommendation ranking model."),
        make_chunk("teaching_chunk", "Taught statistics at university."),
    ]

    retriever = SemanticRetriever(chunks, FakeEmbedder())

    results = retriever.search("recommendation query", limit=1)

    assert [chunk.chunk_id for chunk in results] == ["ranking_chunk"]


def test_semantic_retriever_returns_no_results_for_blank_query() -> None:
    retriever = SemanticRetriever([], FakeEmbedder())

    assert retriever.search("   ", limit=3) == []
