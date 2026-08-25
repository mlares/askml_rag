"""Validation for one reproducible corpus and retrieval-benchmark release."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from askml_rag.models import Chunk, EvaluationQuestion


@dataclass(frozen=True)
class ReleaseDefinition:
    """The versioned inputs and fixed retrieval settings for one release."""

    release_id: str
    full_text_chunks: Path
    publication_summary_chunks: Path
    identity_index: Path
    chunk_size_words: int
    overlap_words: int
    question_sets: tuple[Path, ...]
    method: str
    corpus: str
    retrieval_limit: int


@dataclass(frozen=True)
class ReleaseValidation:
    """A compact, printable result for a validated corpus release."""

    release_id: str
    document_count: int
    full_text_chunk_count: int
    publication_summary_chunk_count: int
    question_count: int
    question_counts_by_topic_language: dict[str, int]


def _relative_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay within the repository.")
    return path


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def load_release_definition(path: Path) -> ReleaseDefinition:
    """Load a deliberately small, strict release-definition YAML file."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "release_id",
        "corpus",
        "benchmark",
    }:
        raise ValueError("Release definition must contain only its four required fields.")
    if payload["schema_version"] != "1":
        raise ValueError("Unsupported release-definition schema version.")
    if not isinstance(payload["release_id"], str) or not payload["release_id"]:
        raise ValueError("release_id must be a non-empty string.")

    corpus = payload["corpus"]
    if not isinstance(corpus, dict) or set(corpus) != {
        "full_text_chunks",
        "publication_summary_chunks",
        "identity_index",
        "chunk_size_words",
        "overlap_words",
    }:
        raise ValueError("corpus must contain the fixed corpus inputs and chunking settings.")
    benchmark = payload["benchmark"]
    if not isinstance(benchmark, dict) or set(benchmark) != {
        "question_sets",
        "method",
        "corpus",
        "retrieval_limit",
    }:
        raise ValueError("benchmark must contain its fixed question sets and settings.")
    question_sets = benchmark["question_sets"]
    if not isinstance(question_sets, list) or not question_sets:
        raise ValueError("benchmark.question_sets must be a non-empty list.")
    resolved_question_sets = tuple(
        _relative_path(value, field="benchmark.question_sets")
        for value in question_sets
    )
    if len(set(resolved_question_sets)) != len(resolved_question_sets):
        raise ValueError("benchmark.question_sets cannot contain duplicates.")
    if not all(isinstance(value, str) and value for value in (benchmark["method"], benchmark["corpus"])):
        raise ValueError("benchmark.method and benchmark.corpus must be non-empty strings.")

    return ReleaseDefinition(
        release_id=payload["release_id"],
        full_text_chunks=_relative_path(
            corpus["full_text_chunks"], field="corpus.full_text_chunks"
        ),
        publication_summary_chunks=_relative_path(
            corpus["publication_summary_chunks"],
            field="corpus.publication_summary_chunks",
        ),
        identity_index=_relative_path(corpus["identity_index"], field="corpus.identity_index"),
        chunk_size_words=_positive_int(corpus["chunk_size_words"], field="corpus.chunk_size_words"),
        overlap_words=_positive_int(corpus["overlap_words"], field="corpus.overlap_words"),
        question_sets=resolved_question_sets,
        method=benchmark["method"],
        corpus=benchmark["corpus"],
        retrieval_limit=_positive_int(
            benchmark["retrieval_limit"], field="benchmark.retrieval_limit"
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as file:
        return [Chunk.model_validate_json(line) for line in file if line.strip()]


def _load_questions(path: Path) -> list[EvaluationQuestion]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"questions"}:
        raise ValueError(f"{path}: expected one top-level 'questions' field.")
    questions = payload["questions"]
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"{path}: questions must be a non-empty list.")
    return [EvaluationQuestion.model_validate(question) for question in questions]


def validate_release(
    definition: ReleaseDefinition, *, project_root: Path
) -> ReleaseValidation:
    """Verify that versioned labels exactly match the local generated corpus."""
    full_text_path = project_root / definition.full_text_chunks
    summary_path = project_root / definition.publication_summary_chunks
    index_path = project_root / definition.identity_index
    required_paths = [full_text_path, summary_path, index_path]
    required_paths.extend(project_root / path for path in definition.question_sets)
    missing = [path.relative_to(project_root) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Release input is missing: {', '.join(map(str, missing))}")

    full_text_chunks = _load_chunks(full_text_path)
    summary_chunks = _load_chunks(summary_path)
    all_chunk_ids = [chunk.chunk_id for chunk in full_text_chunks]
    if len(all_chunk_ids) != len(set(all_chunk_ids)):
        raise ValueError("Canonical full-text corpus contains duplicate chunk IDs.")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("schema_version") != "1":
        raise ValueError("Corpus identity index has an unsupported schema version.")
    expected_full_text = {chunk.chunk_id: chunk.document_id for chunk in full_text_chunks}
    expected_summaries = {chunk.chunk_id: chunk.document_id for chunk in summary_chunks}
    if index.get("full_text_chunks") != expected_full_text:
        raise ValueError("Corpus identity index does not match canonical full-text chunks.")
    if index.get("publication_summary_chunks") != expected_summaries:
        raise ValueError("Corpus identity index does not match publication summary chunks.")
    if index.get("document_ids") != sorted(
        set(expected_full_text.values()) | set(expected_summaries.values())
    ):
        raise ValueError("Corpus identity index document IDs do not match canonical chunks.")
    sources = index.get("sources")
    if not isinstance(sources, dict) or sources.get("full_text_chunks_sha256") != _sha256_file(full_text_path) or sources.get("summary_chunks_sha256") != _sha256_file(summary_path):
        raise ValueError("Corpus identity index hashes do not match generated chunks.")

    questions = [
        question
        for path in definition.question_sets
        for question in _load_questions(project_root / path)
    ]
    question_ids = [question.question_id for question in questions]
    question_texts = [question.question for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Release benchmark contains duplicate question IDs.")
    if len(question_texts) != len(set(question_texts)):
        raise ValueError("Release benchmark contains duplicate question text.")

    document_ids = set(index["document_ids"])
    full_text_by_id = index["full_text_chunks"]
    publication_document_ids = set(index["publication_summary_chunks"].values())
    for question in questions:
        if not set(question.expected_document_ids) <= document_ids:
            raise ValueError(f"{question.question_id}: references an unknown document.")
        if question.answer_mode.value == "metadata" and not set(
            question.expected_document_ids
        ) <= publication_document_ids:
            raise ValueError(f"{question.question_id}: metadata evidence is not a publication.")
        for chunk_id in question.relevant_chunk_ids:
            if full_text_by_id.get(chunk_id) not in set(question.expected_document_ids):
                raise ValueError(
                    f"{question.question_id}: chunk {chunk_id} is not valid evidence."
                )

    counts = Counter(f"{question.topic.value}:{question.language.value}" for question in questions)
    return ReleaseValidation(
        release_id=definition.release_id,
        document_count=len(index["document_ids"]),
        full_text_chunk_count=len(full_text_chunks),
        publication_summary_chunk_count=len(summary_chunks),
        question_count=len(questions),
        question_counts_by_topic_language=dict(sorted(counts.items())),
    )
