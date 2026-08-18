import pytest

from askml_rag.models import (
    Chunk,
    DocumentType,
    Language,
    RetrievalFilters,
    Visibility,
)
from askml_rag.retrieval.filtering import matches_filters


def make_chunk(
    *,
    document_type: DocumentType = DocumentType.project,
    topics: list[str] | None = None,
    year: int | None = 2024,
    visibility: Visibility = Visibility.public,
    language: Language | None = None,
) -> Chunk:
    return Chunk(
        chunk_id="example_chunk",
        document_id="example_source",
        chunk_index=0,
        document_content_hash="a" * 64,
        title="Example source",
        document_type=document_type,
        year=year,
        visibility=visibility,
        language=language,
        source_url="https://www.mlares.space/",
        topics=topics or [],
        text="Example text.",
    )


def test_matches_filters_for_public_chunk_with_all_constraints() -> None:
    chunk = make_chunk(topics=["machine-learning", "astronomy"])
    filters = RetrievalFilters(
        document_types=[DocumentType.project, DocumentType.teaching],
        topics=["machine-learning"],
        year_from=2020,
        year_to=2025,
    )

    assert matches_filters(chunk, filters)


def test_rejects_chunk_with_a_different_document_type() -> None:
    chunk = make_chunk(document_type=DocumentType.teaching)
    filters = RetrievalFilters(document_types=[DocumentType.project])

    assert not matches_filters(chunk, filters)


def test_filters_chunks_by_document_id() -> None:
    chunk = make_chunk()

    assert matches_filters(
        chunk,
        RetrievalFilters(document_ids=["example_source"]),
    )
    assert not matches_filters(
        chunk,
        RetrievalFilters(document_ids=["different_source"]),
    )


def test_rejects_chunk_without_a_requested_topic() -> None:
    chunk = make_chunk(topics=["astronomy"])
    filters = RetrievalFilters(topics=["machine-learning"])

    assert not matches_filters(chunk, filters)


def test_filters_chunks_by_language_but_can_retain_unknown_language() -> None:
    spanish_chunk = make_chunk(language=Language.spanish)
    unknown_language_chunk = make_chunk(language=None)
    spanish_only = RetrievalFilters(languages=[Language.spanish])
    spanish_with_papers = RetrievalFilters(
        languages=[Language.spanish],
        include_unknown_language=True,
    )

    assert matches_filters(spanish_chunk, spanish_only)
    assert not matches_filters(unknown_language_chunk, spanish_only)
    assert matches_filters(unknown_language_chunk, spanish_with_papers)


def test_rejects_chunk_without_year_when_a_year_filter_is_requested() -> None:
    chunk = make_chunk(year=None)
    filters = RetrievalFilters(year_from=2020)

    assert not matches_filters(chunk, filters)


def test_retrieval_filters_reject_invalid_year_range() -> None:
    with pytest.raises(ValueError, match="year_from cannot be greater"):
        RetrievalFilters(year_from=2025, year_to=2024)
