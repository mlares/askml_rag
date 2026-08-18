"""Generate the compact corpus identity index used by benchmark tests."""

import hashlib
import json
from pathlib import Path

from askml_rag.models import Chunk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_TEXT_CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
)
SUMMARY_CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "publication_summary_chunks.jsonl"
)
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "corpus_index.json"


def load_chunk_index(path: Path) -> dict[str, str]:
    """Map every chunk ID to its parent document ID."""
    with path.open(encoding="utf-8") as file:
        chunks = [Chunk.model_validate_json(line) for line in file if line.strip()]
    return dict(sorted((chunk.chunk_id, chunk.document_id) for chunk in chunks))


def sha256_file(path: Path) -> str:
    """Return a stable SHA-256 digest for one generated corpus artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a compact identity index for one retrieval corpus."
    )
    parser.add_argument(
        "--full-text-chunks",
        type=Path,
        default=Path("data/processed/chunks/chunks.jsonl"),
    )
    parser.add_argument(
        "--summary-chunks",
        type=Path,
        default=Path("data/processed/publication_summary_chunks.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/corpus_index.json"),
    )
    arguments = parser.parse_args()
    full_text_path = PROJECT_ROOT / arguments.full_text_chunks
    summary_path = PROJECT_ROOT / arguments.summary_chunks
    output_path = PROJECT_ROOT / arguments.output

    full_text_chunks = load_chunk_index(full_text_path)
    summary_chunks = load_chunk_index(summary_path)
    document_ids = sorted(
        set(full_text_chunks.values()) | set(summary_chunks.values())
    )
    payload = {
        "schema_version": "1",
        "sources": {
            "full_text_chunks_sha256": sha256_file(full_text_path),
            "summary_chunks_sha256": sha256_file(summary_path),
        },
        "document_ids": document_ids,
        "full_text_chunks": full_text_chunks,
        "publication_summary_chunks": summary_chunks,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    print(f"Documents: {len(document_ids)}")
    print(f"Full-text chunks: {len(full_text_chunks)}")
    print(f"Publication summary chunks: {len(summary_chunks)}")
    print(f"Output: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
