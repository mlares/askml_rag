"""Configurable, machine-readable retrieval evaluation runner."""

import json
import math
import statistics
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from askml_rag.evaluation.retrieval import (
    document_recall_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    recall_fraction_at_k,
    reciprocal_rank_at_k,
)
from askml_rag.models import AnswerMode, Chunk, EvaluationQuestion, RetrievalFilters
from askml_rag.retrieval.hybrid import Retriever


class RetrievalMethod(StrEnum):
    bm25 = "bm25"
    semantic = "semantic"
    hybrid = "hybrid"


class CorpusMode(StrEnum):
    full_text = "full_text"
    summaries = "summaries"
    combined = "combined"
    two_stage = "two_stage"


class RetrievalEvaluationConfig(BaseModel):
    """Configuration recorded with every retrieval evaluation report."""

    model_config = ConfigDict(extra="forbid")

    method: RetrievalMethod = RetrievalMethod.bm25
    corpus_mode: CorpusMode = CorpusMode.full_text
    k_values: list[int] = Field(default_factory=lambda: [3, 5, 10], min_length=1)
    paper_limit: int = Field(default=10, gt=0)
    candidate_limit: int = Field(default=50, gt=0)
    embedding_model: str | None = None

    @model_validator(mode="after")
    def validate_limits(self) -> "RetrievalEvaluationConfig":
        if any(k <= 0 for k in self.k_values):
            raise ValueError("Every k value must be greater than zero.")

        self.k_values = sorted(set(self.k_values))
        maximum_k = max(self.k_values)

        if self.candidate_limit < maximum_k:
            raise ValueError("candidate_limit must be at least the maximum k value.")
        if self.corpus_mode == CorpusMode.two_stage and self.paper_limit < maximum_k:
            raise ValueError("paper_limit must be at least the maximum k value.")
        if self.method != RetrievalMethod.bm25 and not self.embedding_model:
            raise ValueError("An embedding_model is required for semantic retrieval.")

        return self


class MetricsAtK(BaseModel):
    chunk_hit_rate: float | None
    chunk_recall: float | None
    chunk_precision: float | None
    reciprocal_rank: float | None
    ndcg: float | None
    document_recall: float
    context_words: int


class QuestionEvaluationResult(BaseModel):
    question_id: str
    question: str
    answerable: bool
    category: str
    answer_mode: str
    expected_chunk_ids: list[str]
    expected_document_ids: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_document_ids: list[str]
    retrieved_summary_chunk_ids: list[str] = Field(default_factory=list)
    latency_ms: float
    metrics: dict[str, MetricsAtK]


class AggregateMetricsAtK(BaseModel):
    evaluated_questions: int
    passage_evaluated_questions: int
    mean_chunk_hit_rate: float | None
    mean_chunk_recall: float | None
    mean_chunk_precision: float | None
    mean_reciprocal_rank: float | None
    mean_ndcg: float | None
    mean_document_recall: float
    mean_context_words: float


class EvaluationReport(BaseModel):
    schema_version: str = "1"
    created_at: datetime
    config: RetrievalEvaluationConfig
    question_count: int
    answerable_question_count: int
    full_text_chunk_count: int
    summary_chunk_count: int
    latency_median_ms: float
    latency_p95_ms: float
    aggregate: dict[str, AggregateMetricsAtK]
    results: list[QuestionEvaluationResult]


RetrieverFactory = Callable[[Sequence[Chunk]], Retriever]


def load_chunks(path: Path) -> list[Chunk]:
    """Load retrieval chunks from JSON Lines."""
    with path.open(encoding="utf-8") as file:
        return [Chunk.model_validate_json(line) for line in file if line.strip()]


def load_questions(path: Path) -> list[EvaluationQuestion]:
    """Load and strictly validate benchmark questions from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(payload) != {"questions"}:
        raise ValueError("The benchmark root must contain only 'questions'.")
    return [
        EvaluationQuestion.model_validate(question)
        for question in payload["questions"]
    ]


def unique_document_ids(chunks: Sequence[Chunk]) -> list[str]:
    """Return parent document IDs in first-occurrence ranking order."""
    return list(dict.fromkeys(chunk.document_id for chunk in chunks))


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return a nearest-rank percentile without third-party dependencies."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


def evaluate_question(
    question: EvaluationQuestion,
    *,
    retrieved_chunks: Sequence[Chunk],
    retrieved_document_ids: Sequence[str],
    summary_chunks: Sequence[Chunk],
    latency_ms: float,
    k_values: Sequence[int],
    score_passages: bool,
) -> QuestionEvaluationResult:
    """Compute all configured metrics for one ranked retrieval result."""
    metrics = {}

    for k in k_values:
        metrics[str(k)] = MetricsAtK(
            chunk_hit_rate=(
                recall_at_k(question, retrieved_chunks, k=k)
                if score_passages
                else None
            ),
            chunk_recall=(
                recall_fraction_at_k(question, retrieved_chunks, k=k)
                if score_passages
                else None
            ),
            chunk_precision=(
                precision_at_k(question, retrieved_chunks, k=k)
                if score_passages
                else None
            ),
            reciprocal_rank=(
                reciprocal_rank_at_k(question, retrieved_chunks, k=k)
                if score_passages
                else None
            ),
            ndcg=(
                ndcg_at_k(question, retrieved_chunks, k=k)
                if score_passages
                else None
            ),
            document_recall=document_recall_at_k(
                question,
                retrieved_document_ids,
                k=k,
            ),
            context_words=sum(
                len(chunk.text.split()) for chunk in retrieved_chunks[:k]
            ),
        )

    return QuestionEvaluationResult(
        question_id=question.question_id,
        question=question.question,
        answerable=question.answerable,
        category=question.category.value,
        answer_mode=question.answer_mode.value,
        expected_chunk_ids=question.relevant_chunk_ids,
        expected_document_ids=question.expected_document_ids,
        retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
        retrieved_document_ids=list(retrieved_document_ids),
        retrieved_summary_chunk_ids=[chunk.chunk_id for chunk in summary_chunks],
        latency_ms=latency_ms,
        metrics=metrics,
    )


def aggregate_results(
    results: Sequence[QuestionEvaluationResult],
    *,
    k_values: Sequence[int],
) -> dict[str, AggregateMetricsAtK]:
    """Average metrics over answerable benchmark questions only."""
    scoreable = [result for result in results if result.answerable]
    aggregate = {}

    for k in k_values:
        key = str(k)
        values = [result.metrics[key] for result in scoreable]
        passage_values = [
            value for value in values if value.chunk_recall is not None
        ]
        count = len(values)

        def mean(attribute: str, metric_values: Sequence[MetricsAtK]) -> float:
            return statistics.fmean(
                getattr(value, attribute) for value in metric_values
            )

        def passage_mean(attribute: str) -> float | None:
            if not passage_values:
                return None
            return mean(attribute, passage_values)

        aggregate[key] = AggregateMetricsAtK(
            evaluated_questions=count,
            passage_evaluated_questions=len(passage_values),
            mean_chunk_hit_rate=passage_mean("chunk_hit_rate"),
            mean_chunk_recall=passage_mean("chunk_recall"),
            mean_chunk_precision=passage_mean("chunk_precision"),
            mean_reciprocal_rank=passage_mean("reciprocal_rank"),
            mean_ndcg=passage_mean("ndcg"),
            mean_document_recall=(
                mean("document_recall", values) if values else 0.0
            ),
            mean_context_words=mean("context_words", values) if values else 0.0,
        )

    return aggregate


def run_evaluation(
    questions: Sequence[EvaluationQuestion],
    full_text_chunks: Sequence[Chunk],
    summary_chunks: Sequence[Chunk],
    *,
    config: RetrievalEvaluationConfig,
    retriever_factory: RetrieverFactory,
) -> EvaluationReport:
    """Run one retrieval configuration over the complete benchmark."""
    maximum_k = max(config.k_values)
    retriever: Retriever | None = None
    summary_retriever: Retriever | None = None
    full_text_retriever: Retriever | None = None

    if config.corpus_mode == CorpusMode.full_text:
        retriever = retriever_factory(full_text_chunks)
    elif config.corpus_mode == CorpusMode.summaries:
        retriever = retriever_factory(summary_chunks)
    elif config.corpus_mode == CorpusMode.combined:
        retriever = retriever_factory([*full_text_chunks, *summary_chunks])
    else:
        summary_retriever = retriever_factory(summary_chunks)
        full_text_retriever = retriever_factory(full_text_chunks)

    results = []
    latencies = []

    for question in questions:
        started = time.perf_counter()
        selected_summaries: list[Chunk] = []

        if config.corpus_mode == CorpusMode.two_stage:
            assert summary_retriever is not None
            assert full_text_retriever is not None
            selected_summaries = summary_retriever.search(
                question.question,
                limit=config.paper_limit,
            )
            document_ids = unique_document_ids(selected_summaries)
            retrieved_chunks = full_text_retriever.search(
                question.question,
                limit=maximum_k,
                filters=RetrievalFilters(document_ids=document_ids),
            )
            retrieved_document_ids = document_ids
        else:
            assert retriever is not None
            retrieved_chunks = retriever.search(
                question.question,
                limit=maximum_k,
            )
            retrieved_document_ids = unique_document_ids(retrieved_chunks)
            if config.corpus_mode == CorpusMode.summaries:
                selected_summaries = list(retrieved_chunks)
            elif config.corpus_mode == CorpusMode.combined:
                selected_summaries = [
                    chunk
                    for chunk in retrieved_chunks
                    if chunk.chunk_id.endswith("_summary")
                ]

        latency_ms = (time.perf_counter() - started) * 1000
        latencies.append(latency_ms)
        results.append(
            evaluate_question(
                question,
                retrieved_chunks=retrieved_chunks,
                retrieved_document_ids=retrieved_document_ids,
                summary_chunks=selected_summaries,
                latency_ms=latency_ms,
                k_values=config.k_values,
                score_passages=(
                    question.answer_mode == AnswerMode.passage
                    and config.corpus_mode != CorpusMode.summaries
                ),
            )
        )

    return EvaluationReport(
        created_at=datetime.now(UTC),
        config=config,
        question_count=len(questions),
        answerable_question_count=sum(question.answerable for question in questions),
        full_text_chunk_count=len(full_text_chunks),
        summary_chunk_count=len(summary_chunks),
        latency_median_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p95_ms=percentile(latencies, 0.95),
        aggregate=aggregate_results(results, k_values=config.k_values),
        results=results,
    )


def write_evaluation_report(report: EvaluationReport, output_path: Path) -> Path:
    """Write a retrieval report as formatted, machine-readable JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        report.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
    )
    output_path.write_text(f"{serialized}\n", encoding="utf-8")
    return output_path
