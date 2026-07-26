import argparse
import json
from pathlib import Path

from askml_rag.models import CanonicalDocument
from askml_rag.retrieval.chunking import chunk_document


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_canonical_documents(directory: Path) -> list[CanonicalDocument]:
    """Load all top-level canonical JSON documents from a directory."""
    documents = []

    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents.append(CanonicalDocument.model_validate(payload))

    return documents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk canonical documents into retrieval-ready JSONL."
    )
    parser.add_argument(
        "--chunk-size-words",
        type=int,
        default=250,
        help="Maximum number of words per chunk.",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=40,
        help="Number of words shared by adjacent chunks.",
    )
    arguments = parser.parse_args()

    input_directory = PROJECT_ROOT / "data" / "processed"
    output_directory = input_directory / "chunks"
    output_directory.mkdir(parents=True, exist_ok=True)

    documents = load_canonical_documents(input_directory)

    chunks = [
        chunk
        for document in documents
        for chunk in chunk_document(
            document,
            chunk_size_words=arguments.chunk_size_words,
            overlap_words=arguments.overlap_words,
        )
    ]

    chunks_path = output_directory / "chunks.jsonl"
    chunks_content = "\n".join(
        json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False)
        for chunk in chunks
    )
    chunks_path.write_text(f"{chunks_content}\n", encoding="utf-8")

    chunking_manifest = {
        "strategy": "word_overlap",
        "chunk_size_words": arguments.chunk_size_words,
        "overlap_words": arguments.overlap_words,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "documents": [
            {
                "document_id": document.document_id,
                "content_hash": document.content_hash,
            }
            for document in documents
        ],
    }

    manifest_path = output_directory / "chunking_manifest.json"
    manifest_path.write_text(
        f"{json.dumps(chunking_manifest, indent=2, ensure_ascii=False)}\n",
        encoding="utf-8",
    )

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Chunks file: {chunks_path.relative_to(PROJECT_ROOT)}")
    print(f"Configuration: {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
