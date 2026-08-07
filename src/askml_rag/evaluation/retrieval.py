import math
from collections.abc import Sequence

from askml_rag.models import Chunk, EvaluationQuestion


def recall_at_k(
    question: EvaluationQuestion,
    retrieved_chunks: Sequence[Chunk],
    *,
    k: int,
) -> float:
    """Return 1.0 when a labelled relevant chunk occurs in the top k."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant_ids = set(question.relevant_chunk_ids)

    if not relevant_ids:
        return 0.0

    retrieved_ids = {chunk.chunk_id for chunk in retrieved_chunks[:k]}

    return float(bool(relevant_ids & retrieved_ids))


def precision_at_k(
    question: EvaluationQuestion,
    retrieved_chunks: Sequence[Chunk],
    *,
    k: int,
) -> float:
    """Return the fraction of the top k results labelled relevant.

    The denominator remains k when fewer than k chunks are returned, treating
    missing results as non-relevant for a fixed-size retrieval request.
    """
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant_ids = set(question.relevant_chunk_ids)
    retrieved_ids = [chunk.chunk_id for chunk in retrieved_chunks[:k]]

    return sum(chunk_id in relevant_ids for chunk_id in retrieved_ids) / k


def reciprocal_rank_at_k(
    question: EvaluationQuestion,
    retrieved_chunks: Sequence[Chunk],
    *,
    k: int,
) -> float:
    """Return the reciprocal rank of the first relevant result in the top k."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant_ids = set(question.relevant_chunk_ids)

    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        if chunk.chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def recall_fraction_at_k(
    question: EvaluationQuestion,
    retrieved_chunks: Sequence[Chunk],
    *,
    k: int,
) -> float:
    """Return the fraction of all labelled relevant chunks found in the top k."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant_ids = set(question.relevant_chunk_ids)
    if not relevant_ids:
        return 0.0

    retrieved_ids = {chunk.chunk_id for chunk in retrieved_chunks[:k]}
    return len(relevant_ids & retrieved_ids) / len(relevant_ids)


def ndcg_at_k(
    question: EvaluationQuestion,
    retrieved_chunks: Sequence[Chunk],
    *,
    k: int,
) -> float:
    """Return binary-relevance normalized discounted cumulative gain at k."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant_ids = set(question.relevant_chunk_ids)
    if not relevant_ids:
        return 0.0

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk in enumerate(retrieved_chunks[:k], start=1)
        if chunk.chunk_id in relevant_ids
    )
    ideal_count = min(len(relevant_ids), k)
    ideal_dcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )
    return dcg / ideal_dcg


def document_recall_at_k(
    question: EvaluationQuestion,
    retrieved_document_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Return coverage of expected parent documents in the first k documents."""
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    expected_ids = set(question.expected_document_ids)
    if not expected_ids:
        return 0.0

    retrieved_ids = set(retrieved_document_ids[:k])
    return len(expected_ids & retrieved_ids) / len(expected_ids)
