from askml_rag.models import Chunk, RetrievalFilters
from askml_rag.retrieval.planned_bm25 import (
    PlannedBM25Retriever,
    decompose_query,
    rewrite_query,
)


def make_chunk(
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    text: str,
    *,
    language: str = "en",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        document_content_hash="a" * 64,
        title=document_id,
        document_type="website",
        language=language,
        topics=[],
        text=text,
    )


def test_rewrite_query_removes_boilerplate_and_expands_gcp() -> None:
    rewritten = rewrite_query(
        "Which GCP services are documented in Marcelo's indexed public corpus?"
    )

    assert "marcelo" not in rewritten
    assert "documented" not in rewritten
    assert "google cloud platform" in rewritten
    assert "compute engine" in rewritten
    assert "bigquery" in rewritten


def test_decompose_query_separates_multi_part_evidence_needs() -> None:
    assert decompose_query(
        "What evidence documents both mentoring and formal evaluation work?"
    ) == [
        "mentoring students advisor supervision",
        "formal evaluation reviewer committee conicet",
    ]


def test_professional_routing_excludes_famaf_candidates() -> None:
    chunks = [
        make_chunk(
            "website_teaching_chunk_000",
            "website_teaching",
            0,
            "university teaching statistics data science",
        ),
        make_chunk(
            "famaf_teaching_activity_chunk_000",
            "famaf_teaching_activity",
            0,
            "university teaching statistics data science teaching",
        ),
        make_chunk("noise_one_chunk_000", "noise_one", 0, "astronomy"),
        make_chunk("noise_two_chunk_000", "noise_two", 0, "software"),
        make_chunk("noise_three_chunk_000", "noise_three", 0, "galaxies"),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "What teaching experience does Marcelo have in statistics?",
        limit=3,
        filters=RetrievalFilters(languages=["en"]),
    )

    assert [chunk.chunk_id for chunk in results] == [
        "website_teaching_chunk_000"
    ]


def test_comparison_queries_expand_adjacent_chunks() -> None:
    chunks = [
        make_chunk(
            "website_projects_chunk_000",
            "website_projects",
            0,
            "cosmic void detection methodology",
        ),
        make_chunk(
            "website_projects_chunk_001",
            "website_projects",
            1,
            "galaxy spin alignment inference",
        ),
        make_chunk("noise_one_chunk_000", "noise_one", 0, "astronomy"),
        make_chunk("noise_two_chunk_000", "noise_two", 0, "software"),
        make_chunk("noise_three_chunk_000", "noise_three", 0, "statistics"),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "What is the difference between cosmic void detection and galaxy spin?",
        limit=2,
    )

    assert {chunk.chunk_id for chunk in results} == {
        "website_projects_chunk_000",
        "website_projects_chunk_001",
    }


def test_professional_results_are_diversified_by_document() -> None:
    chunks = [
        *[
            make_chunk(
                f"website_home_chunk_00{index}",
                "website_home",
                index,
                f"python experience software {index}",
            )
            for index in range(3)
        ],
        make_chunk(
            "skills_chunk_000",
            "skills",
            0,
            "python scientific computing",
        ),
        make_chunk("noise_one_chunk_000", "noise_one", 0, "astronomy"),
        make_chunk("noise_two_chunk_000", "noise_two", 0, "galaxies"),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search("What Python experience does Marcelo have?", limit=4)

    assert sum(chunk.document_id == "website_home" for chunk in results) <= 2
    assert any(chunk.document_id == "skills" for chunk in results)


def test_selected_language_uses_a_separate_lexical_index() -> None:
    chunks = [
        make_chunk(
            "website_home_chunk_000",
            "website_home",
            0,
            "python scientific computing",
        ),
        make_chunk(
            "website_home_es_chunk_000",
            "website_home_es",
            0,
            "python computación científica python",
            language="es",
        ),
        make_chunk("noise_one_chunk_000", "noise_one", 0, "astronomy"),
        make_chunk("noise_two_chunk_000", "noise_two", 0, "galaxies"),
        make_chunk("noise_three_chunk_000", "noise_three", 0, "statistics"),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "Python experience",
        limit=2,
        filters=RetrievalFilters(languages=["en"]),
    )

    assert [chunk.language for chunk in results] == ["en"]
