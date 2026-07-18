from pathlib import Path
import yaml
from askml_rag.models import EvaluationQuestion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.yaml"


def test_evaluation_questions_validate() -> None:
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    assert set(payload) == {"questions"}
    assert payload["questions"], "The benchmark must contain at least one question."

    questions = [
            EvaluationQuestion.model_validate(question)
            for question in payload["questions"]
    ]

    question_ids = [question.question_id for question in questions]
    assert len(question_ids) == len(set(question_ids))
