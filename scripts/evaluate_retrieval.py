import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from askml_rag.evaluation.runner import (
    CorpusMode,
    RetrievalEvaluationConfig,
    RetrievalMethod,
    load_chunks,
    load_questions,
    run_evaluation,
    write_evaluation_report,
)
from askml_rag.models import Chunk
from askml_rag.retrieval.bm25 import BM25Retriever
from askml_rag.retrieval.hybrid import HybridRetriever, Retriever
from askml_rag.retrieval.semantic import SemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_TEXT_CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
)
SUMMARY_CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "publication_summary_chunks.jsonl"
)
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.yaml"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def format_metric(value: float | None) -> str:
    """Format an aggregate metric, preserving non-applicable values."""
    return "n/a" if value is None else f"{value:.3f}"


def build_retriever_factory(
    config: RetrievalEvaluationConfig,
) -> Callable[[Sequence[Chunk]], Retriever]:
    """Create a corpus-specific retriever factory from CLI configuration."""
    if config.method == RetrievalMethod.bm25:
        return lambda chunks: BM25Retriever(chunks)

    from sentence_transformers import SentenceTransformer

    assert config.embedding_model is not None
    print(f"Loading embedding model: {config.embedding_model}")
    embedder = SentenceTransformer(config.embedding_model)

    if config.method == RetrievalMethod.semantic:
        return lambda chunks: SemanticRetriever(chunks, embedder)

    def hybrid_factory(chunks: Sequence[Chunk]) -> Retriever:
        return HybridRetriever(
            BM25Retriever(chunks),
            SemanticRetriever(chunks, embedder),
            candidate_limit=config.candidate_limit,
        )

    return hybrid_factory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one retrieval configuration over the AskML benchmark."
    )
    parser.add_argument(
        "--method",
        choices=[method.value for method in RetrievalMethod],
        default=RetrievalMethod.bm25.value,
    )
    parser.add_argument(
        "--corpus",
        choices=[mode.value for mode in CorpusMode],
        default=CorpusMode.full_text.value,
    )
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--paper-limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path; defaults to reports/retrieval/<method>_<corpus>.json.",
    )
    arguments = parser.parse_args()

    method = RetrievalMethod(arguments.method)
    corpus_mode = CorpusMode(arguments.corpus)
    config = RetrievalEvaluationConfig(
        method=method,
        corpus_mode=corpus_mode,
        k_values=arguments.k,
        paper_limit=arguments.paper_limit,
        candidate_limit=arguments.candidate_limit,
        embedding_model=(
            arguments.embedding_model
            if method != RetrievalMethod.bm25
            else None
        ),
    )

    output_path = arguments.output or Path(
        f"reports/retrieval/{method.value}_{corpus_mode.value}.json"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    questions = load_questions(QUESTIONS_PATH)
    full_text_chunks = load_chunks(FULL_TEXT_CHUNKS_PATH)
    summary_chunks = load_chunks(SUMMARY_CHUNKS_PATH)
    report = run_evaluation(
        questions,
        full_text_chunks,
        summary_chunks,
        config=config,
        retriever_factory=build_retriever_factory(config),
    )
    write_evaluation_report(report, output_path)

    print(f"Questions: {report.question_count}")
    print(f"Answerable questions: {report.answerable_question_count}")
    print(f"Full-text chunks: {report.full_text_chunk_count}")
    print(f"Summary chunks: {report.summary_chunk_count}")
    for k in config.k_values:
        metrics = report.aggregate[str(k)]
        print(
            f"k={k}: chunk recall={format_metric(metrics.mean_chunk_recall)}, "
            f"MRR={format_metric(metrics.mean_reciprocal_rank)}, "
            f"nDCG={format_metric(metrics.mean_ndcg)}, "
            f"document recall={metrics.mean_document_recall:.3f}"
        )
    print(f"Median latency: {report.latency_median_ms:.2f} ms")
    print(f"P95 latency: {report.latency_p95_ms:.2f} ms")
    try:
        displayed_path = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        displayed_path = output_path
    print(f"Report: {displayed_path}")


if __name__ == "__main__":
    main()
