"""Validate the versioned AskML corpus and retrieval-benchmark release."""

from pathlib import Path

from askml_rag.evaluation.release import load_release_definition, validate_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate one reproducible AskML corpus and benchmark release."
    )
    parser.add_argument(
        "--release",
        type=Path,
        default=Path("data/evaluation/release.yaml"),
        help="Versioned release-definition YAML relative to the project root.",
    )
    arguments = parser.parse_args()
    definition = load_release_definition(PROJECT_ROOT / arguments.release)
    result = validate_release(definition, project_root=PROJECT_ROOT)
    print(f"Release: {result.release_id}")
    print(f"Documents: {result.document_count}")
    print(f"Full-text chunks: {result.full_text_chunk_count}")
    print(f"Publication summary chunks: {result.publication_summary_chunk_count}")
    print(f"Benchmark questions: {result.question_count}")
    for slice_name, count in result.question_counts_by_topic_language.items():
        print(f"  {slice_name}: {count}")


if __name__ == "__main__":
    main()
