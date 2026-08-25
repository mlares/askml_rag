from pathlib import Path

from askml_rag.evaluation.release import load_release_definition, validate_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASE_PATH = PROJECT_ROOT / "data" / "evaluation" / "release.yaml"


def test_current_release_is_self_consistent() -> None:
    definition = load_release_definition(RELEASE_PATH)
    result = validate_release(definition, project_root=PROJECT_ROOT)

    assert result.release_id == "2026.08.2"
    assert result.document_count == 140
    assert result.full_text_chunk_count == 4260
    assert result.publication_summary_chunk_count == 38
    assert result.question_count == 110
    assert result.question_counts_by_topic_language == {
        "cross_source_reasoning:en": 20,
        "famaf_teaching:en": 12,
        "famaf_teaching:es": 12,
        "ithreex_projects:en": 9,
        "ithreex_projects:es": 9,
        "professional_profile:en": 35,
        "professional_profile:es": 5,
        "scientific_publications:en": 8,
    }
