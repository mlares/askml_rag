import argparse
from pathlib import Path

from askml_rag.ingestion.markdown import write_canonical_document
from askml_rag.ingestion.pdf import ingest_pdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest one text-based PDF source document."
    )
    parser.add_argument(
        "manifest",
        help="Path to a source manifest, relative to the project root.",
    )
    arguments = parser.parse_args()

    manifest_path = PROJECT_ROOT / arguments.manifest
    document = ingest_pdf(manifest_path, project_root=PROJECT_ROOT)

    output_path = write_canonical_document(
        document,
        output_directory=PROJECT_ROOT / "data" / "processed",
    )

    print(f"Ingested: {document.document_id}")
    print(f"Characters: {len(document.body_markdown)}")
    print(f"Content hash: {document.content_hash}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
