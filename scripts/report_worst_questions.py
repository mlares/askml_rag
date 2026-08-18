#!/usr/bin/env python3
"""Rank the weakest answerable questions in a retrieval evaluation report."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def quality_score(result: dict[str, Any], *, k: str) -> float:
    """Average applicable evidence-coverage and ranking metrics."""
    metrics = result["metrics"][k]
    attributes = ["document_recall"]
    if metrics["chunk_recall"] is not None:
        attributes.extend(("chunk_recall", "reciprocal_rank", "ndcg"))
    return statistics.fmean(metrics[attribute] for attribute in attributes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--k", default="7")
    parser.add_argument("--limit", type=int, default=10)
    arguments = parser.parse_args()
    if arguments.limit <= 0:
        parser.error("--limit must be greater than zero.")
    payload = json.loads(arguments.report.read_text(encoding="utf-8"))
    ranked = sorted(
        (
            result
            for result in payload["results"]
            if result["answerable"]
        ),
        key=lambda result: (
            quality_score(result, k=arguments.k),
            result["question_id"],
        ),
    )
    print(
        "rank\tquality\ttopic\tlanguage\tquestion_id\tchunk_recall\tMRR\t"
        "nDCG\tdocument_recall\tquestion"
    )
    for rank, result in enumerate(ranked[: arguments.limit], start=1):
        metrics = result["metrics"][arguments.k]
        values = [
            rank,
            f"{quality_score(result, k=arguments.k):.3f}",
            result["topic"],
            result["language"],
            result["question_id"],
            "n/a"
            if metrics["chunk_recall"] is None
            else f"{metrics['chunk_recall']:.3f}",
            "n/a"
            if metrics["reciprocal_rank"] is None
            else f"{metrics['reciprocal_rank']:.3f}",
            "n/a" if metrics["ndcg"] is None else f"{metrics['ndcg']:.3f}",
            f"{metrics['document_recall']:.3f}",
            result["question"].replace("\t", " ").replace("\n", " "),
        ]
        print("\t".join(map(str, values)))


if __name__ == "__main__":
    main()
