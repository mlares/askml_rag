from collections import defaultdict
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIRECTORY = PROJECT_ROOT / "data" / "manifests"


def load_manifests() -> list[tuple[Path, dict[str, object]]]:
    return [
        (path, yaml.safe_load(path.read_text(encoding="utf-8")))
        for path in sorted(MANIFEST_DIRECTORY.glob("*.yaml"))
    ]


def test_every_source_has_exactly_one_opposite_language_counterpart() -> None:
    manifests = load_manifests()
    by_id: dict[str, tuple[Path, dict[str, object]]] = {}
    translations: defaultdict[str, list[dict[str, object]]] = defaultdict(list)

    for path, manifest in manifests:
        document_id = str(manifest["document_id"])
        assert document_id not in by_id, f"Duplicate document_id: {document_id}"
        by_id[document_id] = (path, manifest)
        assert manifest.get("language") in {"en", "es"}, path
        if original_id := manifest.get("translation_of"):
            translations[str(original_id)].append(manifest)

    originals = {
        document_id: manifest
        for document_id, (_, manifest) in by_id.items()
        if not manifest.get("translation_of")
    }
    assert set(translations) == set(originals)
    for original_id, original in originals.items():
        counterparts = translations[original_id]
        assert len(counterparts) == 1, original_id
        counterpart = counterparts[0]
        assert counterpart["language"] != original["language"], original_id
        assert {counterpart["language"], original["language"]} == {"en", "es"}


def test_every_manifest_source_path_exists_in_the_local_corpus() -> None:
    for manifest_path, manifest in load_manifests():
        source_path = PROJECT_ROOT / str(manifest["source_path"])
        assert source_path.is_file(), f"{manifest_path}: missing {source_path}"


def test_full_text_publication_translations_pass_completeness_screen() -> None:
    for translated_path in sorted(
        (PROJECT_ROOT / "data/processed").glob("arxiv_*_es.json")
    ):
        translated = json.loads(translated_path.read_text(encoding="utf-8"))
        original_id = translated.get("translation_of")
        if not original_id:
            continue
        original_path = PROJECT_ROOT / "data/processed" / f"{original_id}.json"
        original = json.loads(original_path.read_text(encoding="utf-8"))
        length_ratio = len(translated["body_markdown"]) / len(
            original["body_markdown"]
        )
        assert length_ratio >= 0.75, (
            f"Incomplete translation: {translated['document_id']} "
            f"({length_ratio:.3f})"
        )
