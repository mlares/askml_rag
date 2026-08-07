from pathlib import Path
import pytest
import yaml
from pydantic import ValidationError
from askml_rag.models import EvaluationQuestion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.yaml"


def test_evaluation_questions_validate() -> None:
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    assert set(payload) == {"questions"}
    assert payload["questions"], "The benchmark must contain at least one question."

    questions = [
        EvaluationQuestion.model_validate(question) for question in payload["questions"]
    ]

    question_ids = [question.question_id for question in questions]
    assert len(question_ids) == len(set(question_ids))

    question_texts = [question.question for question in questions]
    assert len(question_texts) == len(set(question_texts))

    for question in questions:
        if question.answerable:
            assert question.expected_document_ids
            assert question.relevant_chunk_ids
            assert question.expected_claims

            for claim in question.expected_claims:
                assert set(claim.source_document_ids) <= set(
                    question.expected_document_ids
                )


def test_evaluation_question_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="relevant_chunk_ds"):
        EvaluationQuestion.model_validate(
            {
                "question_id": "invalid_001",
                "question": "An invalid benchmark question?",
                "answerable": False,
                "category": "unanswerable",
                "expected_document_ids": [],
                "relevant_chunk_ds": [],
                "expected_claims": [],
                "difficulty": "unanswerable",
            }
        )
