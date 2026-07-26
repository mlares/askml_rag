import argparse
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from askml_rag.models import DocumentType, RetrievalFilters
from askml_rag.retrieval.qdrant_store import QdrantSemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QDRANT_PATH = PROJECT_ROOT / "data" / "processed" / "qdrant"

COLLECTION_NAME = "askml_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the local AskML Qdrant collection."
    )

    parser.add_argument("query", help="Natural-language search query.")
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of chunks to return (default: 3).",
    )
    parser.add_argument(
        "--document-type",
        action="append",
        choices=[document_type.value for document_type in DocumentType],
        help=(
            "Restrict results to a document type. "
            "Repeat the option to allow multiple types."
        ),
    )
    parser.add_argument(
        "--topic",
        action="append",
        help=(
            "Restrict results to a topic. Repeat the option to allow multiple topics."
        ),
    )
    parser.add_argument(
        "--year-from",
        type=int,
        help="Only return chunks from this year onwards.",
    )
    parser.add_argument(
        "--year-to",
        type=int,
        help="Only return chunks up to and including this year.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    client = QdrantClient(path=QDRANT_PATH)

    if not client.collection_exists(COLLECTION_NAME):
        raise SystemExit(
            "Qdrant collection does not exist. "
            "Run `uv run python scripts/index_qdrant.py` first."
        )

    print(f"Loading embedding model: {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)

    retriever = QdrantSemanticRetriever(
        client,
        COLLECTION_NAME,
        embedder,
        model_name=MODEL_NAME,
    )

    filters = RetrievalFilters(
        document_types=[DocumentType(value) for value in args.document_type or []],
        topics=args.topic or [],
        year_from=args.year_from,
        year_to=args.year_to,
    )

    results = retriever.search(
        args.query,
        limit=args.limit,
        filters=filters,
    )

    if not results:
        print("No public evidence matched this query and filter combination.")
        return

    for rank, chunk in enumerate(results, start=1):
        print(f"\n--- Result {rank}: {chunk.chunk_id} ---")
        print(f"Document: {chunk.title} ({chunk.document_id})")
        print(f"Type: {chunk.document_type.value}")
        print(f"Year: {chunk.year or 'not recorded'}")
        print(f"Topics: {', '.join(chunk.topics) or 'not recorded'}")
        print(f"Source: {chunk.source_url or 'not recorded'}")
        print()
        print(chunk.text)


if __name__ == "__main__":
    main()
