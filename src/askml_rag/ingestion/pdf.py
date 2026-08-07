"""Extraction and ingestion of text-based PDF source documents."""

import hashlib
from pathlib import Path

import pymupdf
import yaml

from askml_rag.models import CanonicalDocument, SourceManifest


def normalize_pdf_text(text: str) -> str:
    """Normalize whitespace from text extracted from one PDF page."""
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]

    if not lines:
        return ""

    return "\n".join(lines) + "\n"


def extract_pdf_pages(
    source_path: Path,
    *,
    page_limit: int | None = None,
) -> list[tuple[int, str]]:
    """Extract normalized text from every page, preserving page numbers."""
    if page_limit is not None and page_limit <= 0:
        raise ValueError("page_limit must be greater than zero.")

    with pymupdf.open(source_path) as document:
        return [
            (
                page.number + 1,
                normalize_pdf_text(page.get_text("text", sort=True)),
            )
            for page in document
            if page_limit is None or page.number < page_limit
        ]


def extract_pdf_text(
    source_path: Path,
    *,
    page_limit: int | None = None,
) -> str:
    """Extract normalized text from a text-based PDF in page order."""
    non_empty_pages = [
        text.rstrip()
        for _, text in extract_pdf_pages(source_path, page_limit=page_limit)
        if text
    ]

    if not non_empty_pages:
        raise ValueError("No extractable text found in PDF.")

    return "\n\n".join(non_empty_pages) + "\n"


def ingest_pdf(
    manifest_path: Path,
    *,
    project_root: Path,
) -> CanonicalDocument:
    """Create one canonical document from a PDF source and YAML manifest."""
    manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = SourceManifest.model_validate(manifest_data)

    root = project_root.resolve()
    source_path = (root / manifest.source_path).resolve()

    try:
        source_path.relative_to(root)
    except ValueError as error:
        message = "The manifest source_path must stay inside the project root."
        raise ValueError(message) from error

    body_markdown = extract_pdf_text(source_path)
    content_hash = hashlib.sha256(body_markdown.encode("utf-8")).hexdigest()

    return CanonicalDocument(
        **manifest.model_dump(),
        content_hash=content_hash,
        body_markdown=body_markdown,
    )
