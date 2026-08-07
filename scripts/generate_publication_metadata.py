import argparse
from pathlib import Path

from askml_rag.ingestion.publications import (
    generate_publication_catalog,
    write_publication_catalog,
    write_publication_summary_chunks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate normalized metadata for selected publications."
    )
    parser.add_argument(
        "--bibtex",
        type=Path,
        default=Path("data/raw/publications/export-bibtex.bib"),
        help="BibTeX input path, relative to the project root.",
    )
    parser.add_argument(
        "--manifests",
        type=Path,
        default=Path("data/manifests"),
        help="Publication manifest directory, relative to the project root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/publications.jsonl"),
        help="Generated JSONL path, relative to the project root.",
    )
    parser.add_argument(
        "--summary-chunks-output",
        type=Path,
        default=Path("data/processed/publication_summary_chunks.jsonl"),
        help="Generated summary-chunk JSONL path, relative to the project root.",
    )
    arguments = parser.parse_args()

    publications = generate_publication_catalog(
        PROJECT_ROOT / arguments.bibtex,
        PROJECT_ROOT / arguments.manifests,
        project_root=PROJECT_ROOT,
    )
    output_path = write_publication_catalog(
        publications,
        output_path=PROJECT_ROOT / arguments.output,
    )
    summary_chunks_path = write_publication_summary_chunks(
        publications,
        output_path=PROJECT_ROOT / arguments.summary_chunks_output,
    )

    missing_abstracts = [
        publication.document_id
        for publication in publications
        if publication.abstract is None
    ]

    print(f"Publications: {len(publications)}")
    print(f"Missing abstracts: {len(missing_abstracts)}")
    for document_id in missing_abstracts:
        print(f"- {document_id}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Summary chunks: {summary_chunks_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
