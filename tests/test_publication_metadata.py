import json
from pathlib import Path
from textwrap import dedent

import pytest

from askml_rag.ingestion.publications import (
    arxiv_id_from_document_id,
    extract_abstract_from_text,
    generate_publication_catalog,
    load_publication_catalog,
    normalize_journal,
    publication_summary_chunk,
    publication_summary_text,
    write_publication_catalog,
    write_publication_summary_chunks,
)
from askml_rag.models import AbstractSource


def write_manifest(path: Path) -> None:
    path.write_text(
        dedent("""
        document_id: arxiv_1101_1961
        title: Future virialized structures
        document_type: publication
        year: 2011
        authors:
          - H. Luparello
          - M. Lares
        source_url: https://ui.adsabs.harvard.edu/abs/2011MNRAS.415..964L
        source_path: data/raw/publications/arXiv-1101.1961.pdf
        visibility: public
        topics:
          - large-scale structure
        """).lstrip(),
        encoding="utf-8",
    )


def test_arxiv_id_from_document_id_supports_new_and_old_ids() -> None:
    assert arxiv_id_from_document_id("arxiv_1101_1961") == "1101.1961"
    assert (
        arxiv_id_from_document_id("arxiv_astro_ph_0507144")
        == "astro-ph/0507144"
    )


def test_normalize_journal_expands_ads_macro() -> None:
    assert normalize_journal(r"\apjl") == (
        "The Astrophysical Journal Letters",
        "ApJL",
    )


def test_extract_abstract_from_text_uses_explicit_section_boundaries() -> None:
    text = dedent("""
    A paper title
    ABSTRACT
    We identify future virialized structures in a galaxy survey.
    The catalogue provides a reproducible sample.[astro-ph.CO]
    Key words: large-scale structure
    1 INTRODUCTION
    Other text.
    """)

    assert extract_abstract_from_text(text) == (
        "We identify future virialized structures in a galaxy survey. "
        "The catalogue provides a reproducible sample."
    )


@pytest.mark.parametrize(
    ("heading", "expected_start"),
    [
        ("Abstract. Context. We study cosmic voids.", "Context."),
        ("ABSTRACTApr\nWe study cosmic voids.", "We study"),
        ("AcceptedABSTRACTXXX. Received YYY; in original form ZZZ\nWe study cosmic voids.", "We study"),
        ("ABSTRACTWe study cosmic voids.", "We study"),
    ],
)
def test_extract_abstract_from_text_handles_common_pdf_headings(
    heading: str,
    expected_start: str,
) -> None:
    text = f"Paper title\n{heading}\nKeywords: voids\n1. Introduction\nBody"

    abstract = extract_abstract_from_text(text)

    assert abstract is not None
    assert abstract.startswith(expected_start)


def test_generate_publication_catalog_matches_manifest_by_arxiv_id(
    tmp_path: Path,
) -> None:
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    write_manifest(manifests / "arXiv-1101.1961.yaml")

    bibtex_path = tmp_path / "publications.bib"
    bibtex_path.write_text(
        dedent(r"""
        @ARTICLE{2011MNRAS.415..964L,
          author = {{Luparello}, H. and {Lares}, M.},
          title = "{Future virialized structures: an analysis of superstructures}",
          journal = {\mnras},
          year = 2011,
          doi = {10.1000/example},
          eprint = {1101.1961},
          primaryClass = {astro-ph.CO},
          adsurl = {https://ui.adsabs.harvard.edu/abs/2011MNRAS.415..964L},
          abstract = {We identify future virialized structures.}
        }
        """).lstrip(),
        encoding="utf-8",
    )

    publications = generate_publication_catalog(
        bibtex_path,
        manifests,
        project_root=tmp_path,
    )

    assert len(publications) == 1
    publication = publications[0]
    assert publication.document_id == "arxiv_1101_1961"
    assert publication.authors == ["H. Luparello", "M. Lares"]
    assert publication.journal_abbreviation == "MNRAS"
    assert publication.abstract_source == AbstractSource.bibtex
    assert publication.abstract == "We identify future virialized structures."

    output_path = write_publication_catalog(
        publications,
        output_path=tmp_path / "data" / "processed" / "publications.jsonl",
    )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["arxiv_id"] == "1101.1961"

    loaded_publications = load_publication_catalog(output_path)
    assert loaded_publications == publications


def test_publication_summary_chunk_is_linked_to_full_paper(tmp_path: Path) -> None:
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    write_manifest(manifests / "arXiv-1101.1961.yaml")
    bibtex_path = tmp_path / "publications.bib"
    bibtex_path.write_text(
        dedent(r"""
        @ARTICLE{2011MNRAS.415..964L,
          author = {{Luparello}, H. and {Lares}, M.},
          title = {Future virialized structures},
          journal = {\mnras},
          year = 2011,
          eprint = {1101.1961},
          abstract = {We identify superstructures that will virialize.}
        }
        """).lstrip(),
        encoding="utf-8",
    )
    publication = generate_publication_catalog(
        bibtex_path,
        manifests,
        project_root=tmp_path,
    )[0]

    summary_text = publication_summary_text(publication)
    chunk = publication_summary_chunk(publication)

    assert "Title: Future virialized structures" in summary_text
    assert "Authors: H. Luparello; M. Lares" in summary_text
    assert "Abstract: We identify superstructures" in summary_text
    assert chunk.chunk_id == "arxiv_1101_1961_summary"
    assert chunk.document_id == "arxiv_1101_1961"
    assert chunk.text == summary_text
    assert len(chunk.document_content_hash) == 64

    chunks_path = write_publication_summary_chunks(
        [publication],
        output_path=tmp_path / "publication_summary_chunks.jsonl",
    )
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    assert payload["chunk_id"] == "arxiv_1101_1961_summary"


def test_generate_publication_catalog_rejects_missing_bibtex_entry(
    tmp_path: Path,
) -> None:
    manifests = tmp_path / "data" / "manifests"
    manifests.mkdir(parents=True)
    write_manifest(manifests / "arXiv-1101.1961.yaml")
    bibtex_path = tmp_path / "publications.bib"
    bibtex_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="No BibTeX entry found"):
        generate_publication_catalog(
            bibtex_path,
            manifests,
            project_root=tmp_path,
        )
