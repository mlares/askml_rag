from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from askml_rag.models import Chunk
from askml_rag.retrieval.qdrant_store import QdrantSemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
QDRANT_PATH = PROJECT_ROOT / "data" / "processed" / "qdrant"

COLLECTION_NAME = "askml_chunks"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as file:
        return [Chunk.model_validate_json(line) for line in file if line.strip()]


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)

    print(f"Loading embedding model: {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)

    print(f"Opening local Qdrant database: {QDRANT_PATH}")
    client = QdrantClient(path=QDRANT_PATH)

    retriever = QdrantSemanticRetriever(
        client,
        COLLECTION_NAME,
        embedder,
        model_name=MODEL_NAME,
    )

    print(f"Indexing {len(chunks)} chunks into '{COLLECTION_NAME}'...")
    retriever.index(chunks)

    point_count = client.count(COLLECTION_NAME).count
    print(f"Indexed points: {point_count}")
    print("Indexing complete.")


if __name__ == "__main__":
    main()
