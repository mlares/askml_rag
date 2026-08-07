import argparse
from pathlib import Path

from askml_rag.ingestion.publications import load_publication_catalog
from askml_rag.models import Chunk
from askml_rag.retrieval.bm25 import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "publications.jsonl"
SUMMARY_CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "publication_summary_chunks.jsonl"
)


def load_summary_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as file:
        return [Chunk.model_validate_json(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search publication titles, authors, keywords, and abstracts."
    )
    parser.add_argument("query", help="Natural-language publication query.")
    parser.add_argument("--limit", type=int, default=5)
    arguments = parser.parse_args()

    publications = load_publication_catalog(CATALOG_PATH)
    publications_by_id = {
        publication.document_id: publication for publication in publications
    }
    retriever = BM25Retriever(load_summary_chunks(SUMMARY_CHUNKS_PATH))
    results = retriever.search(arguments.query, limit=arguments.limit)

    if not results:
        print("No matching publications found.")
        return

    for rank, chunk in enumerate(results, start=1):
        publication = publications_by_id[chunk.document_id]
        journal = publication.journal_abbreviation or publication.journal or "Unknown"
        print(f"{rank}. {publication.title} ({publication.year})")
        print(f"   Document ID: {publication.document_id}")
        print(f"   Authors: {'; '.join(publication.authors)}")
        print(f"   Journal: {journal}")
        print(f"   Source: {publication.source_url or publication.ads_url}")


if __name__ == "__main__":
    main()
