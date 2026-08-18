from askml_rag.models import Chunk
from askml_rag.retrieval.bm25 import BM25Retriever


def make_chunk(
    *,
    chunk_id: str,
    text: str,
) -> Chunk:
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


def test_bm25_returns_the_most_relevant_chunk() -> None:
    chunks = [
        make_chunk(
            chunk_id="ranking_chunk",
            text="Built and evaluated a production ranking model for recommendations.",
        ),
        make_chunk(
            chunk_id="teaching_chunk",
            text="Taught university courses in statistics and data science.",
        ),
        make_chunk(
            chunk_id="research_chunk",
            text="Developed astronomical data-analysis pipelines.",
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("ranking model", limit=1)

    assert [chunk.chunk_id for chunk in results] == ["ranking_chunk"]


def test_bm25_returns_no_results_when_no_terms_match() -> None:
    chunks = [
        make_chunk(
            chunk_id="teaching_chunk",
            text="Taught university courses in statistics and data science.",
        ),
    ]

    retriever = BM25Retriever(chunks)

    results = retriever.search("underwater basket weaving", limit=3)

    assert results == []


def test_bm25_can_index_titles_and_topics_for_planned_retrieval() -> None:
    relevant = make_chunk(chunk_id="cloud_chunk", text="Compute Engine details.")
    relevant.title = "Google Cloud Platform services"
    decoys = [
        make_chunk(chunk_id=f"decoy_{index}", text=f"Unrelated material {index}")
        for index in range(4)
    ]
    retriever = BM25Retriever([relevant, *decoys], include_metadata=True)

    results = retriever.search("Google Cloud Platform", limit=1)

    assert [chunk.chunk_id for chunk in results] == ["cloud_chunk"]
