import pytest

from askml_rag.models import Chunk, DocumentType, RetrievalFilters
from askml_rag.retrieval.hybrid import HybridRetriever


class StaticRetriever:
    def __init__(self, results: list[Chunk]) -> None:
        self.results = results
        self.received_filters: list[RetrievalFilters | None] = []

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilters | None = None,
    ) -> list[Chunk]:
        self.received_filters.append(filters)
        return self.results[:limit]


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="example_source",
        chunk_index=0,
        document_content_hash="a" * 64,
        title="Example source",
        document_type="website",
        source_url="https://www.mlares.space/",
        topics=[],
        text="Example text.",
    )


def test_hybrid_retriever_promotes_chunks_found_by_both_retrievers() -> None:
    lexical = StaticRetriever([make_chunk("lexical_only"), make_chunk("shared")])
    semantic = StaticRetriever([make_chunk("shared"), make_chunk("semantic_only")])

    retriever = HybridRetriever(
        lexical,
        semantic,
        rank_constant=1,
        candidate_limit=2,
    )

    results = retriever.search("example query", limit=3)

    assert [chunk.chunk_id for chunk in results] == [
        "shared",
        "lexical_only",
        "semantic_only",
    ]


def test_hybrid_retriever_rejects_invalid_limit() -> None:
    retriever = HybridRetriever(
        StaticRetriever([]),
        StaticRetriever([]),
    )

    with pytest.raises(ValueError, match="greater than zero"):
        retriever.search("example query", limit=0)


def test_hybrid_retriever_forwards_filters_to_both_retrievers() -> None:
    lexical = StaticRetriever([make_chunk("lexical_chunk")])
    semantic = StaticRetriever([make_chunk("semantic_chunk")])
    retriever = HybridRetriever(lexical, semantic)
    filters = RetrievalFilters(document_types=[DocumentType.project])

    retriever.search("example query", limit=2, filters=filters)

    assert lexical.received_filters == [filters]
    assert semantic.received_filters == [filters]
