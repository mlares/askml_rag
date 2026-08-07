"""Generate normalized publication metadata from ADS BibTeX and manifests."""

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

import yaml
from pybtex.database import Entry, Person, parse_file
from pybtex.richtext import Text

from askml_rag.ingestion.pdf import extract_pdf_text
from askml_rag.models import (
    AbstractSource,
    Chunk,
    DocumentType,
    PublicationMetadata,
    SourceManifest,
    Visibility,
)


JOURNAL_NAMES = {
    r"\aap": ("Astronomy & Astrophysics", "A&A"),
    r"\aj": ("The Astronomical Journal", "AJ"),
    r"\apj": ("The Astrophysical Journal", "ApJ"),
    r"\apjl": ("The Astrophysical Journal Letters", "ApJL"),
    r"\apjs": ("The Astrophysical Journal Supplement Series", "ApJS"),
    r"\mnras": ("Monthly Notices of the Royal Astronomical Society", "MNRAS"),
}

ABSTRACT_HEADING_PATTERN = re.compile(r"abstract", flags=re.IGNORECASE)
ABSTRACT_END_PATTERN = re.compile(
    r"\n\s*(?:key\s*words?|keywords?)\s*[:.]"
    r"|\n\s*1(?:\.|\s+)\s*(?:introduction|intro)\b",
    flags=re.IGNORECASE,
)
LEADING_PDF_NOISE_PATTERN = re.compile(
    r"^(?:[.:—-]+\s*|\d+\s*|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b\s*)+",
    flags=re.IGNORECASE,
)
ARXIV_CLASS_MARKER_PATTERN = re.compile(
    r"\[(?:astro-ph|cs|econ|eess|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|"
    r"math|math-ph|nlin|nucl-ex|nucl-th|physics|q-bio|q-fin|quant-ph|stat)"
    r"(?:\.[a-z-]+)?\]",
    flags=re.IGNORECASE,
)


def render_latex(value: str) -> str:
    """Render a BibTeX/LaTeX value as normalized Unicode text."""
    return " ".join(Text.from_latex(value).render_as("text").split())


def render_person(person: Person) -> str:
    """Render one Pybtex person in natural given-name-first order."""
    parts = (
        *person.first_names,
        *person.middle_names,
        *person.prelast_names,
        *person.last_names,
        *person.lineage_names,
    )
    return render_latex(" ".join(parts))


def arxiv_id_from_document_id(document_id: str) -> str:
    """Convert a corpus document ID to the arXiv identifier used by ADS."""
    prefix = "arxiv_"
    if not document_id.startswith(prefix):
        raise ValueError(f"Not an arXiv document ID: {document_id}")

    identifier = document_id.removeprefix(prefix)
    old_style_prefix = "astro_ph_"

    if identifier.startswith(old_style_prefix):
        number = identifier.removeprefix(old_style_prefix)
        return f"astro-ph/{number}"

    return identifier.replace("_", ".")


def normalize_arxiv_id(value: str) -> str:
    """Normalize optional arXiv prefixes while preserving old-style IDs."""
    normalized = value.strip()
    normalized = re.sub(r"^arxiv\s*:\s*", "", normalized, flags=re.IGNORECASE)
    return normalized


def normalize_journal(value: str | None) -> tuple[str | None, str | None]:
    """Expand known ADS journal macros and return name plus abbreviation."""
    if not value:
        return None, None

    raw_value = value.strip().strip("{}")
    if raw_value in JOURNAL_NAMES:
        return JOURNAL_NAMES[raw_value]

    journal = render_latex(value)
    return journal, None


def extract_abstract_from_text(text: str) -> str | None:
    """Extract and normalize an explicitly labelled abstract section."""
    # Abstracts occur in the front matter. Limiting the search also avoids
    # matching references or appendices in long full-text extractions.
    front_matter = text[:30_000]
    heading = ABSTRACT_HEADING_PATTERN.search(front_matter)
    if not heading:
        return None

    candidate = front_matter[heading.end() :]
    candidate = LEADING_PDF_NOISE_PATTERN.sub("", candidate.lstrip())

    first_line, separator, remaining = candidate.partition("\n")
    first_line_lower = first_line.lower()
    if separator and any(
        marker in first_line_lower
        for marker in ("received", "accepted", "original form")
    ):
        candidate = remaining

    end = ABSTRACT_END_PATTERN.search(candidate)
    if not end:
        return None

    abstract_text = ARXIV_CLASS_MARKER_PATTERN.sub(" ", candidate[: end.start()])
    abstract = " ".join(abstract_text.split())
    return abstract or None


def load_publication_manifests(directory: Path) -> list[SourceManifest]:
    """Load public arXiv publication manifests in deterministic order."""
    manifests = []

    for path in sorted(directory.glob("arXiv*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = SourceManifest.model_validate(payload)

        if (
            manifest.document_type == DocumentType.publication
            and manifest.visibility == Visibility.public
        ):
            manifests.append(manifest)

    document_ids = [manifest.document_id for manifest in manifests]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Publication manifests contain duplicate document IDs.")

    return manifests


def index_bibtex_by_arxiv_id(entries: Iterable[tuple[str, Entry]]) -> dict[str, tuple[str, Entry]]:
    """Index BibTeX entries by eprint and reject ambiguous identifiers."""
    indexed: dict[str, tuple[str, Entry]] = {}

    for bibtex_key, entry in entries:
        eprint = entry.fields.get("eprint")
        if not eprint:
            continue

        arxiv_id = normalize_arxiv_id(eprint)
        if arxiv_id in indexed:
            raise ValueError(f"Duplicate BibTeX eprint: {arxiv_id}")
        indexed[arxiv_id] = (bibtex_key, entry)

    return indexed


def publication_from_entry(
    manifest: SourceManifest,
    *,
    bibtex_key: str,
    entry: Entry,
    project_root: Path,
) -> PublicationMetadata:
    """Combine a selected manifest and its authoritative BibTeX entry."""
    arxiv_id = arxiv_id_from_document_id(manifest.document_id)
    fields = entry.fields
    abstract = fields.get("abstract")
    abstract_source = AbstractSource.bibtex if abstract else None

    if abstract:
        abstract = render_latex(abstract)
    else:
        source_path = (project_root / manifest.source_path).resolve()
        root = project_root.resolve()
        try:
            source_path.relative_to(root)
        except ValueError as error:
            message = "The manifest source_path must stay inside the project root."
            raise ValueError(message) from error

        abstract = extract_abstract_from_text(
            extract_pdf_text(source_path, page_limit=3)
        )
        if abstract:
            abstract_source = AbstractSource.pdf

    journal, journal_abbreviation = normalize_journal(fields.get("journal"))
    keywords = [
        render_latex(keyword)
        for keyword in fields.get("keywords", "").split(",")
        if keyword.strip()
    ]

    return PublicationMetadata(
        document_id=manifest.document_id,
        bibtex_key=bibtex_key,
        entry_type=entry.type,
        title=render_latex(fields["title"]),
        authors=[render_person(person) for person in entry.persons.get("author", [])],
        year=int(fields["year"]),
        journal=journal,
        journal_abbreviation=journal_abbreviation,
        abstract=abstract,
        abstract_source=abstract_source,
        keywords=keywords,
        doi=fields.get("doi"),
        arxiv_id=arxiv_id,
        primary_class=fields.get("primaryClass"),
        volume=fields.get("volume"),
        number=fields.get("number"),
        pages=fields.get("pages"),
        source_url=manifest.source_url,
        ads_url=fields.get("adsurl"),
    )


def generate_publication_catalog(
    bibtex_path: Path,
    manifests_directory: Path,
    *,
    project_root: Path,
) -> list[PublicationMetadata]:
    """Generate metadata for every selected public publication manifest."""
    bibliography = parse_file(str(bibtex_path), bib_format="bibtex")
    entries_by_arxiv_id = index_bibtex_by_arxiv_id(bibliography.entries.items())
    publications = []

    for manifest in load_publication_manifests(manifests_directory):
        arxiv_id = arxiv_id_from_document_id(manifest.document_id)
        match = entries_by_arxiv_id.get(arxiv_id)
        if match is None:
            raise ValueError(
                f"No BibTeX entry found for {manifest.document_id} ({arxiv_id})."
            )

        bibtex_key, entry = match
        publications.append(
            publication_from_entry(
                manifest,
                bibtex_key=bibtex_key,
                entry=entry,
                project_root=project_root,
            )
        )

    return publications


def write_publication_catalog(
    publications: Iterable[PublicationMetadata],
    *,
    output_path: Path,
) -> Path:
    """Write deterministic publication metadata as JSON Lines."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        publication.model_dump_json(exclude_none=True)
        for publication in publications
    )
    output_path.write_text(f"{content}\n", encoding="utf-8")
    return output_path


def load_publication_catalog(path: Path) -> list[PublicationMetadata]:
    """Load a generated JSONL publication catalog."""
    with path.open(encoding="utf-8") as file:
        return [
            PublicationMetadata.model_validate_json(line)
            for line in file
            if line.strip()
        ]


def publication_summary_text(publication: PublicationMetadata) -> str:
    """Render metadata and abstract as one retrieval-oriented summary."""
    fields = [
        f"Title: {publication.title}",
        f"Authors: {'; '.join(publication.authors)}",
        f"Year: {publication.year}",
    ]

    if publication.journal:
        journal = publication.journal
        if publication.journal_abbreviation:
            journal = f"{journal} ({publication.journal_abbreviation})"
        fields.append(f"Journal: {journal}")
    if publication.keywords:
        fields.append(f"Keywords: {'; '.join(publication.keywords)}")
    if publication.abstract:
        fields.append(f"Abstract: {publication.abstract}")
    fields.append(f"arXiv: {publication.arxiv_id}")
    if publication.doi:
        fields.append(f"DOI: {publication.doi}")

    return "\n".join(fields)


def publication_summary_chunk(publication: PublicationMetadata) -> Chunk:
    """Create one retrievable summary chunk linked to its full paper."""
    text = publication_summary_text(publication)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return Chunk(
        chunk_id=f"{publication.document_id}_summary",
        document_id=publication.document_id,
        chunk_index=0,
        document_content_hash=content_hash,
        title=publication.title,
        document_type=DocumentType.publication,
        year=publication.year,
        visibility=Visibility.public,
        source_url=publication.source_url,
        topics=publication.keywords,
        text=text,
    )


def write_publication_summary_chunks(
    publications: Iterable[PublicationMetadata],
    *,
    output_path: Path,
) -> Path:
    """Write one separate retrieval-summary chunk per publication."""
    chunks = [publication_summary_chunk(publication) for publication in publications]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False)
        for chunk in chunks
    )
    output_path.write_text(f"{content}\n", encoding="utf-8")
    return output_path
