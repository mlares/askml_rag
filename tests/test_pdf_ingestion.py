from pathlib import Path
from textwrap import dedent

import pymupdf
import pytest

from askml_rag.ingestion.pdf import (
    extract_pdf_pages,
    extract_pdf_text,
    ingest_pdf,
    normalize_pdf_text,
)


def create_pdf(path: Path, pages: list[str]) -> None:
    """Create a small text-based PDF fixture without external files."""
    document = pymupdf.open()

    for text in pages:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)

    document.save(path)
    document.close()


def test_normalize_pdf_text_collapses_whitespace() -> None:
    text = " First   line  \n\n second\tline \n"

    assert normalize_pdf_text(text) == "First line\nsecond line\n"


def test_extract_pdf_text_preserves_page_order(tmp_path: Path) -> None:
    source_path = tmp_path / "example.pdf"
    create_pdf(
        source_path,
        ["First page evidence.", "Second page evidence."],
    )

    extracted = extract_pdf_text(source_path)

    assert extracted == "First page evidence.\n\nSecond page evidence.\n"


def test_extract_pdf_pages_preserves_page_numbers(tmp_path: Path) -> None:
    source_path = tmp_path / "example.pdf"
    create_pdf(source_path, ["First page.", "Second page."])

    assert extract_pdf_pages(source_path) == [
        (1, "First page.\n"),
        (2, "Second page.\n"),
    ]


def test_extract_pdf_text_can_limit_front_matter_pages(tmp_path: Path) -> None:
    source_path = tmp_path / "example.pdf"
    create_pdf(source_path, ["Abstract page.", "Body page."])

    assert extract_pdf_text(source_path, page_limit=1) == "Abstract page.\n"


def test_extract_pdf_text_rejects_a_textless_pdf(tmp_path: Path) -> None:
    source_path = tmp_path / "textless.pdf"
    create_pdf(source_path, [""])

    with pytest.raises(ValueError, match="No extractable text"):
        extract_pdf_text(source_path)


def test_ingest_pdf_creates_a_canonical_document(tmp_path: Path) -> None:
    raw_path = tmp_path / "data" / "raw" / "example.pdf"
    raw_path.parent.mkdir(parents=True)
    create_pdf(raw_path, ["A public research result."])

    manifest_path = tmp_path / "data" / "manifests" / "example.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        dedent("""
        document_id: example_paper
        title: Example paper
        document_type: publication
        year: 2024
        authors:
          - Marcelo Lares
        source_url: https://example.org/paper
        source_path: data/raw/example.pdf
        visibility: public
        topics:
          - example
        """).lstrip(),
        encoding="utf-8",
    )

    document = ingest_pdf(manifest_path, project_root=tmp_path)

    assert document.document_id == "example_paper"
    assert document.content_hash
    assert len(document.content_hash) == 64
    assert document.body_markdown == "A public research result.\n"
