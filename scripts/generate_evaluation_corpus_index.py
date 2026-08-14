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
    full_text_chunks = load_chunk_index(FULL_TEXT_CHUNKS_PATH)
    summary_chunks = load_chunk_index(SUMMARY_CHUNKS_PATH)
    document_ids = sorted(
        set(full_text_chunks.values()) | set(summary_chunks.values())
    )
    payload = {
        "schema_version": "1",
        "sources": {
            "full_text_chunks_sha256": sha256_file(FULL_TEXT_CHUNKS_PATH),
            "summary_chunks_sha256": sha256_file(SUMMARY_CHUNKS_PATH),
        },
        "document_ids": document_ids,
        "full_text_chunks": full_text_chunks,
        "publication_summary_chunks": summary_chunks,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n",
        encoding="utf-8",
    )
    print(f"Documents: {len(document_ids)}")
    print(f"Full-text chunks: {len(full_text_chunks)}")
    print(f"Publication summary chunks: {len(summary_chunks)}")
    print(f"Output: {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
