import argparse
from pathlib import Path
from askml_rag.models import Chunk
from askml_rag.retrieval.bm25 import BM25Retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"


def load_chunks(path: Path) -> list[Chunk]:
    """Load retrieval chunks from a JSON Lines file."""
    chunks = []

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunks.append(Chunk.model_validate_json(line))

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the chunk corpus with BM25 lexical retrieval."
    )
    parser.add_argument("query", help="Question or search query.")
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Maximum number of chunks to return.",
    )
    arguments = parser.parse_args()

    chunks = load_chunks(CHUNKS_PATH)
    retriever = BM25Retriever(chunks)
    results = retriever.search(arguments.query, limit=arguments.limit)

    if not results:
        print("No lexical evidence found in the indexed corpus.")
        return

    for rank, chunk in enumerate(results, start=1):
        print(f"\n[{rank}] {chunk.title}")
        print(f"Chunk: {chunk.chunk_id}")
        print(f"Source: {chunk.source_url}")
        print(f"Topics: {', '.join(chunk.topics)}")
        print()
        print(chunk.text)


if __name__ == "__main__":
    main()
