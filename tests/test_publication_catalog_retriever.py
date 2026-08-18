import pytest

from askml_rag.models import PublicationMetadata
from askml_rag.retrieval.publication_catalog import (
    PublicationCatalogRetriever,
    author_name_matches,
    normalize_author_name,
)


def publication(
    document_id: str,
    *,
    title: str,
    year: int,
    authors: list[str],
) -> PublicationMetadata:
    return PublicationMetadata(
        document_id=document_id,
        bibtex_key=document_id,
        entry_type="article",
        title=title,
        authors=authors,
        year=year,
        arxiv_id=document_id.removeprefix("arxiv_"),
    )


def test_normalize_author_name_removes_accents_and_separates_hyphens() -> None:
    assert normalize_author_name("I. V. Daza-Perilla") == (
        "i",
        "v",
        "daza",
        "perilla",
    )
    assert normalize_author_name("García Lambas") == ("garcia", "lambas")


def test_author_name_matches_initials_and_surname_queries() -> None:
    assert author_name_matches("M. Lares", "Marcelo Lares")
    assert author_name_matches("I. V. Daza-Perilla", "Daza Perilla")
    assert author_name_matches("I. V. Daza-Perilla", "I. Daza-Perilla")
    assert not author_name_matches("D. G. Lambas", "Daza-Perilla")
    assert not author_name_matches("G. Smith", "Gramajo")


def test_catalog_retriever_returns_all_papers_shared_by_two_authors() -> None:
    retriever = PublicationCatalogRetriever(
        [
            publication(
                "arxiv_2023_a",
                title="Eclipsing systems",
                year=2023,
                authors=["I. V. Daza-Perilla", "M. Lares"],
            ),
            publication(
                "arxiv_2025_a",
                title="Galaxy catalogue",
                year=2025,
                authors=["M. Lares", "I. V. Daza Perilla", "D. Minniti"],
            ),
            publication(
                "arxiv_2024_a",
                title="Other collaboration",
                year=2024,
                authors=["M. Lares", "D. G. Lambas"],
            ),
        ]
    )

    publications = retriever.publications_by_authors(
        ["Marcelo Lares", "Daza-Perilla"]
    )
    chunks = retriever.chunks_by_authors(["Marcelo Lares", "Daza-Perilla"])

    assert [publication.document_id for publication in publications] == [
        "arxiv_2025_a",
        "arxiv_2023_a",
    ]
    assert [chunk.chunk_id for chunk in chunks] == [
        "arxiv_2025_a_summary",
        "arxiv_2023_a_summary",
    ]


def test_catalog_retriever_validates_author_query_and_limit() -> None:
    retriever = PublicationCatalogRetriever([])

    with pytest.raises(ValueError, match="At least one"):
        retriever.publications_by_authors([])
    with pytest.raises(ValueError, match="letters or numbers"):
        retriever.publications_by_authors(["---"])
    with pytest.raises(ValueError, match="greater than zero"):
        retriever.publications_by_authors(["Marcelo Lares"], limit=0)
