import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from askml_rag.models import AnswerMode, Chunk, EvaluationQuestion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.yaml"
FULL_TEXT_CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
)
SUMMARY_CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "publication_summary_chunks.jsonl"
)
CORPUS_INDEX_PATH = PROJECT_ROOT / "data" / "evaluation" / "corpus_index.json"


def load_questions() -> list[EvaluationQuestion]:
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    assert set(payload) == {"questions"}
    assert payload["questions"], "The benchmark must contain at least one question."
    return [
        EvaluationQuestion.model_validate(question)
        for question in payload["questions"]
    ]


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as file:
        return [Chunk.model_validate_json(line) for line in file if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_evaluation_questions_validate() -> None:
    questions = load_questions()

    question_ids = [question.question_id for question in questions]
    assert len(question_ids) == len(set(question_ids))

    question_texts = [question.question for question in questions]
    assert len(question_texts) == len(set(question_texts))

    for question in questions:
        assert question.topic
        assert question.language
        if question.answerable:
            assert question.expected_document_ids
            assert question.expected_claims
            if question.answer_mode == AnswerMode.passage:
                assert question.relevant_chunk_ids
            else:
                assert question.answer_mode == AnswerMode.metadata
                assert not question.relevant_chunk_ids

            for claim in question.expected_claims:
                assert set(claim.source_document_ids) <= set(
                    question.expected_document_ids
                )
        else:
            assert question.answer_mode == AnswerMode.unanswerable
            assert not question.expected_document_ids
            assert not question.relevant_chunk_ids
            assert not question.expected_claims


def test_evaluation_labels_reference_the_generated_corpus() -> None:
    questions = load_questions()
    corpus_index = json.loads(CORPUS_INDEX_PATH.read_text(encoding="utf-8"))
    assert corpus_index["schema_version"] == "1"
    full_text_by_id = corpus_index["full_text_chunks"]
    all_document_ids = set(corpus_index["document_ids"])
    publication_document_ids = set(
        corpus_index["publication_summary_chunks"].values()
    )

    for question in questions:
        assert set(question.expected_document_ids) <= all_document_ids

        for claim in question.expected_claims:
            assert set(claim.source_document_ids) <= all_document_ids

        for chunk_id in question.relevant_chunk_ids:
            assert chunk_id in full_text_by_id
            assert full_text_by_id[chunk_id] in set(question.expected_document_ids)

        if question.answer_mode == AnswerMode.metadata:
            assert set(question.expected_document_ids) <= publication_document_ids


def test_evaluation_corpus_index_matches_local_generated_chunks() -> None:
    if not FULL_TEXT_CHUNKS_PATH.exists() or not SUMMARY_CHUNKS_PATH.exists():
        pytest.skip("Generated corpus is not present in this checkout.")

    corpus_index = json.loads(CORPUS_INDEX_PATH.read_text(encoding="utf-8"))
    full_text_chunks = load_chunks(FULL_TEXT_CHUNKS_PATH)
    summary_chunks = load_chunks(SUMMARY_CHUNKS_PATH)

    assert corpus_index["sources"]["full_text_chunks_sha256"] == sha256_file(
        FULL_TEXT_CHUNKS_PATH
    )
    assert corpus_index["sources"]["summary_chunks_sha256"] == sha256_file(
        SUMMARY_CHUNKS_PATH
    )
    assert corpus_index["full_text_chunks"] == {
        chunk.chunk_id: chunk.document_id for chunk in full_text_chunks
    }
    assert corpus_index["publication_summary_chunks"] == {
        chunk.chunk_id: chunk.document_id for chunk in summary_chunks
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"answerable": False, "answer_mode": "unanswerable"},
        {"answerable": True, "answer_mode": "passage", "relevant_chunk_ids": []},
        {
            "answerable": True,
            "answer_mode": "metadata",
            "relevant_chunk_ids": ["paper_chunk_000"],
        },
    ],
)
def test_evaluation_question_rejects_inconsistent_evidence_labels(
    overrides: dict[str, object],
) -> None:
    payload = {
        "question_id": "example_001",
        "question": "An example question?",
        "topic": "scientific_publications",
        "language": "en",
        "answerable": True,
        "answer_mode": "passage",
        "category": "direct_fact",
        "expected_document_ids": ["paper"],
        "relevant_chunk_ids": ["paper_chunk_000"],
        "expected_claims": [
            {"claim": "An example claim.", "source_document_ids": ["paper"]}
        ],
        "difficulty": "direct",
    }
    payload.update(overrides)

    with pytest.raises(ValidationError):
        EvaluationQuestion.model_validate(payload)


def test_evaluation_question_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="relevant_chunk_ds"):
        EvaluationQuestion.model_validate(
            {
                "question_id": "invalid_001",
                "question": "An invalid benchmark question?",
                "topic": "professional_profile",
                "language": "en",
                "answerable": False,
                "category": "unanswerable",
                "expected_document_ids": [],
                "relevant_chunk_ds": [],
                "expected_claims": [],
                "difficulty": "unanswerable",
            }
        )
