import json
from pathlib import Path

from askml_rag.evaluation.runner import (
    CorpusMode,
    RetrievalEvaluationConfig,
    RetrievalMethod,
    run_evaluation,
    write_evaluation_report,
)
from askml_rag.models import Chunk, EvaluationQuestion
from askml_rag.retrieval.bm25 import BM25Retriever


def make_chunk(
    chunk_id: str,
    document_id: str,
    text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        document_content_hash="a" * 64,
        title=document_id,
        document_type="publication",
        source_url="https://example.org/paper",
        topics=[],
        text=text,
    )


def make_question(
    relevant_chunk_id: str | None,
    *,
    answer_mode: str = "passage",
) -> EvaluationQuestion:
    return EvaluationQuestion(
        question_id="future_structures_001",
        question="future virialized structures",
        answerable=True,
        category="direct_fact",
        answer_mode=answer_mode,
        expected_document_ids=["paper_future"],
        relevant_chunk_ids=[relevant_chunk_id] if relevant_chunk_id else [],
        expected_claims=[
            {
                "claim": "The paper studies future virialized structures.",
                "source_document_ids": ["paper_future"],
            }
        ],
        difficulty="direct",
    )


def test_summary_mode_evaluates_publication_discovery() -> None:
    summaries = [
        make_chunk(
            "paper_future_summary",
            "paper_future",
            "Future virialized structures in the SDSS survey.",
        ),
        make_chunk(
            "paper_voids_summary",
            "paper_voids",
            "Cosmic void dynamics and galaxy velocities.",
        ),
        make_chunk(
            "paper_clusters_summary",
            "paper_clusters",
            "Galaxy clusters and the large-scale density field.",
        ),
    ]
    config = RetrievalEvaluationConfig(
        method=RetrievalMethod.bm25,
        corpus_mode=CorpusMode.summaries,
        k_values=[1],
    )

    report = run_evaluation(
        [make_question(None, answer_mode="metadata")],
        [],
        summaries,
        config=config,
        retriever_factory=BM25Retriever,
    )

    result = report.results[0]
    assert result.retrieved_chunk_ids == ["paper_future_summary"]
    assert result.answer_mode == "metadata"
    assert result.metrics["1"].chunk_recall is None
    assert result.metrics["1"].document_recall == 1.0
    assert report.aggregate["1"].passage_evaluated_questions == 0
    assert report.aggregate["1"].mean_ndcg is None


def test_two_stage_mode_records_paper_discovery_and_full_text_evidence() -> None:
    summaries = [
        make_chunk(
            "paper_future_summary",
            "paper_future",
            "Future virialized structures in the SDSS survey.",
        ),
        make_chunk(
            "paper_voids_summary",
            "paper_voids",
            "Cosmic void dynamics.",
        ),
        make_chunk(
            "paper_clusters_summary",
            "paper_clusters",
            "Galaxy clusters and the density field.",
        ),
    ]
    full_text = [
        make_chunk(
            "paper_future_chunk_000",
            "paper_future",
            "The future virialized structures use a luminosity-density map.",
        ),
        make_chunk(
            "paper_voids_chunk_000",
            "paper_voids",
            "Void velocity profiles.",
        ),
        make_chunk(
            "paper_clusters_chunk_000",
            "paper_clusters",
            "Cluster masses and density profiles.",
        ),
    ]
    config = RetrievalEvaluationConfig(
        method=RetrievalMethod.bm25,
        corpus_mode=CorpusMode.two_stage,
        k_values=[1],
        paper_limit=1,
    )

    report = run_evaluation(
        [make_question("paper_future_chunk_000")],
        full_text,
        summaries,
        config=config,
        retriever_factory=BM25Retriever,
    )

    result = report.results[0]
    assert result.retrieved_summary_chunk_ids == ["paper_future_summary"]
    assert result.retrieved_chunk_ids == ["paper_future_chunk_000"]
    assert result.metrics["1"].chunk_recall == 1.0
    assert result.metrics["1"].document_recall == 1.0


def test_evaluation_report_is_written_as_json(tmp_path: Path) -> None:
    summaries = [
        make_chunk(
            "paper_future_summary",
            "paper_future",
            "Future virialized structures.",
        ),
        make_chunk(
            "paper_voids_summary",
            "paper_voids",
            "Cosmic void dynamics.",
        ),
        make_chunk(
            "paper_clusters_summary",
            "paper_clusters",
            "Galaxy clusters and the density field.",
        ),
    ]
    config = RetrievalEvaluationConfig(
        corpus_mode=CorpusMode.summaries,
        k_values=[1],
    )
    report = run_evaluation(
        [make_question(None, answer_mode="metadata")],
        [],
        summaries,
        config=config,
        retriever_factory=BM25Retriever,
    )

    path = write_evaluation_report(report, tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["config"]["corpus_mode"] == "summaries"
    assert payload["aggregate"]["1"]["mean_chunk_recall"] is None
    assert payload["aggregate"]["1"]["mean_document_recall"] == 1.0
