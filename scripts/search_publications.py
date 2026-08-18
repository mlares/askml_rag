import argparse
from pathlib import Path

from askml_rag.ingestion.publications import load_publication_catalog
from askml_rag.retrieval.publication_catalog import PublicationCatalogRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "processed" / "publications.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search publication titles, authors, keywords, and abstracts."
    )
    parser.add_argument("query", nargs="?", help="Natural-language publication query.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--author",
        action="append",
        default=[],
        help="Require this author; repeat to require coauthors."
    )
    arguments = parser.parse_args()
    if not arguments.query and not arguments.author:
        parser.error("provide a query or at least one --author")

    publications = load_publication_catalog(CATALOG_PATH)
    publications_by_id = {
        publication.document_id: publication for publication in publications
    }
    retriever = PublicationCatalogRetriever(publications)
    results = (
        retriever.chunks_by_authors(arguments.author, limit=arguments.limit)
        if arguments.author
        else retriever.search(arguments.query, limit=arguments.limit)
    )

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
