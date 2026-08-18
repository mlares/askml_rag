import pytest

from askml_rag.evaluation.retrieval import (
    document_recall_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recall_fraction_at_k,
    reciprocal_rank_at_k,
)
from askml_rag.models import Chunk, EvaluationQuestion


def make_chunk(chunk_id: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="example_source",
        chunk_index=0,
        document_content_hash="a" * 64,
        title="Example source",
        document_type="website",
        source_url="https://www.mlares.space/",
        topics=[],
        text="Example text.",
    )


def make_question(relevant_chunk_ids: list[str]) -> EvaluationQuestion:
    return EvaluationQuestion(
        question_id="example_question_001",
        question="Example question?",
        topic="professional_profile",
        language="en",
        answerable=True,
        category="direct_fact",
        expected_document_ids=["example_source"],
        relevant_chunk_ids=relevant_chunk_ids,
        expected_claims=[
            {
                "claim": "The example source supports the answer.",
                "source_document_ids": ["example_source"],
            }
        ],
        difficulty="easy",
    )


def test_recall_at_k_is_one_when_relevant_chunk_is_retrieved() -> None:
    question = make_question(["relevant_chunk"])
    results = [make_chunk("other_chunk"), make_chunk("relevant_chunk")]

    assert recall_at_k(question, results, k=2) == 1.0


def test_recall_at_k_is_zero_when_relevant_chunk_is_outside_k() -> None:
    question = make_question(["relevant_chunk"])
    results = [make_chunk("other_chunk"), make_chunk("relevant_chunk")]

    assert recall_at_k(question, results, k=1) == 0.0


def test_recall_at_k_rejects_invalid_k() -> None:
    question = make_question(["relevant_chunk"])

    with pytest.raises(ValueError, match="greater than zero"):
        recall_at_k(question, [], k=0)


def test_precision_at_k_counts_all_relevant_results() -> None:
    question = make_question(["relevant_one", "relevant_two"])
    results = [
        make_chunk("relevant_one"),
        make_chunk("other_chunk"),
        make_chunk("relevant_two"),
    ]

    assert precision_at_k(question, results, k=3) == pytest.approx(2 / 3)


def test_precision_at_k_treats_missing_results_as_non_relevant() -> None:
    question = make_question(["relevant_chunk"])
    results = [make_chunk("relevant_chunk")]

    assert precision_at_k(question, results, k=3) == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("results", "expected_score"),
    [
        (["relevant_chunk", "other_chunk"], 1.0),
        (["other_chunk", "relevant_chunk"], 0.5),
        (["other_chunk", "another_chunk"], 0.0),
    ],
)
def test_reciprocal_rank_at_k_returns_first_relevant_rank(
    results: list[str],
    expected_score: float,
) -> None:
    question = make_question(["relevant_chunk"])
    retrieved_chunks = [make_chunk(chunk_id) for chunk_id in results]

    assert reciprocal_rank_at_k(question, retrieved_chunks, k=3) == expected_score


def test_reciprocal_rank_at_k_ignores_relevant_results_outside_k() -> None:
    question = make_question(["relevant_chunk"])
    results = [make_chunk("other_chunk"), make_chunk("relevant_chunk")]

    assert reciprocal_rank_at_k(question, results, k=1) == 0.0


def test_new_retrieval_metrics_reject_invalid_k() -> None:
    question = make_question(["relevant_chunk"])

    with pytest.raises(ValueError, match="greater than zero"):
        precision_at_k(question, [], k=0)

    with pytest.raises(ValueError, match="greater than zero"):
        reciprocal_rank_at_k(question, [], k=0)


def test_fractional_recall_and_ndcg_reward_complete_ranked_results() -> None:
    question = make_question(["relevant_one", "relevant_two"])
    results = [make_chunk("relevant_one"), make_chunk("relevant_two")]

    assert recall_fraction_at_k(question, results, k=1) == 0.5
    assert recall_fraction_at_k(question, results, k=2) == 1.0
    assert ndcg_at_k(question, results, k=2) == 1.0


def test_document_recall_measures_parent_document_coverage() -> None:
    question = make_question(["relevant_chunk"])
    question.expected_document_ids = ["paper_one", "paper_two"]

    assert document_recall_at_k(
        question,
        ["paper_one", "other_paper"],
        k=2,
    ) == 0.5
