#!/usr/bin/env python3
"""Prepare, submit, inspect, and apply Spanish publication translations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from translate_bilingual_dataset import (
    DEFAULT_MAX_COST,
    INPUT_PRICE,
    MODEL,
    OUTPUT_PRICE,
    client,
    estimate_tokens,
    output_text,
    request,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "publication_translation"
INPUT_PATH = ARTIFACT_DIR / "batch_input.jsonl"
TASKS_PATH = ARTIFACT_DIR / "tasks.json"
STATE_PATH = ARTIFACT_DIR / "batch_state.json"
OUTPUT_PATH = ARTIFACT_DIR / "batch_output.jsonl"


def publication_manifests() -> list[tuple[Path, dict[str, Any]]]:
    """Return original publication manifests in stable path order."""
    manifests = []
    for path in sorted((PROJECT_ROOT / "data/manifests").glob("arXiv-*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if payload.get("translation_of"):
            continue
        manifests.append((path, payload))
    return manifests


def source_body(document_id: str) -> str:
    """Load the canonical text extracted from the authoritative PDF."""
    path = PROJECT_ROOT / "data/processed" / f"{document_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = payload.get("body_markdown")
    if not isinstance(body, str) or not body.strip():
        raise ValueError(f"Canonical publication has no body_markdown: {path}")
    return body


def translation_prompt(body: str) -> str:
    """Build a faithful full-text scientific translation request."""
    return (
        "Translate this scientific publication from English to Spanish. Return "
        "only translated Markdown, without a preamble or code fence. Preserve "
        "headings, lists, tables, equations, mathematical notation, citations, "
        "references, URLs, DOIs, arXiv IDs, code, data values, units, names, and "
        "Markdown syntax. Keep the publication title and cited publication titles "
        "unchanged. Translate all explanatory prose faithfully and naturally. Do "
        "not invent, omit, summarize, or merge content. This is a derived retrieval "
        "translation; the English PDF remains authoritative.\n\n"
        "--- BEGIN SOURCE ---\n"
        f"{body}\n"
        "--- END SOURCE ---\n"
    )


def target_source_path(manifest_path: Path) -> Path:
    """Return the repository-relative Markdown path for a Spanish paper."""
    return Path("data/raw/publications_es") / f"{manifest_path.stem}_es.md"


def target_manifest_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}_es.yaml")


def prepare() -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    input_tokens = output_tokens = 0
    for manifest_path, manifest in publication_manifests():
        document_id = manifest["document_id"]
        body = source_body(document_id)
        prompt = translation_prompt(body)
        maximum = estimate_tokens(body) + 2048
        translated_document_id = f"{document_id}_es"
        custom_id = f"publication__{translated_document_id}"
        requests.append(request(custom_id, prompt, maximum))
        tasks.append(
            {
                "custom_id": custom_id,
                "document_id": document_id,
                "translated_document_id": translated_document_id,
                "manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)),
                "translated_manifest_path": str(
                    target_manifest_path(manifest_path).relative_to(PROJECT_ROOT)
                ),
                "translated_source_path": str(target_source_path(manifest_path)),
                "source_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "max_output_tokens": maximum,
            }
        )
        input_tokens += estimate_tokens(prompt)
        output_tokens += maximum

    manifest = {
        "schema_version": "1",
        "model": MODEL,
        "request_count": len(requests),
        "estimated_input_tokens_upper_bound": input_tokens,
        "estimated_output_tokens_upper_bound": output_tokens,
        "estimated_cost_usd_upper_bound": (
            input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE
        )
        / 1_000_000,
        "tasks": tasks,
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_PATH.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests),
        encoding="utf-8",
    )
    TASKS_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_manifest() -> dict[str, Any]:
    if not INPUT_PATH.is_file() or not TASKS_PATH.is_file():
        raise SystemExit("No prepared publication job. Run prepare first.")
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        raise SystemExit("No submitted publication job. Run submit first.")
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def submit(env_file: str, api_key_env: str, max_cost: float) -> None:
    manifest = load_manifest()
    estimate = float(manifest["estimated_cost_usd_upper_bound"])
    if estimate > max_cost:
        raise SystemExit(
            f"Refusing to submit: estimated US${estimate:.4f} exceeds "
            f"US${max_cost:.2f}."
        )
    openai = client(env_file, api_key_env)
    with INPUT_PATH.open("rb") as handle:
        uploaded = openai.files.create(file=handle, purpose="batch")
    batch = openai.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"project": "askml-rag", "purpose": "publication-bilingual"},
    )
    STATE_PATH.write_text(
        json.dumps(
            {
                "batch_id": batch.id,
                "input_file_id": uploaded.id,
                "estimated_cost_usd_upper_bound": estimate,
                "max_cost_usd": max_cost,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Submitted batch: {batch.id}")
    print(f"Requests: {manifest['request_count']}")
    print(f"Estimated upper-bound cost: US${estimate:.4f}")


def status(env_file: str, api_key_env: str) -> None:
    batch = client(env_file, api_key_env).batches.retrieve(load_state()["batch_id"])
    print(
        json.dumps(
            {
                "batch_id": batch.id,
                "status": batch.status,
                "request_counts": batch.request_counts.model_dump()
                if batch.request_counts
                else None,
                "output_file_id": batch.output_file_id,
                "error_file_id": batch.error_file_id,
            },
            indent=2,
        )
    )


def apply(env_file: str, api_key_env: str) -> None:
    manifest = load_manifest()
    openai = client(env_file, api_key_env)
    batch = openai.batches.retrieve(load_state()["batch_id"])
    if batch.status != "completed" or not batch.output_file_id:
        raise SystemExit(f"Batch is not ready (status: {batch.status}).")
    raw_output = openai.files.content(batch.output_file_id).read()
    OUTPUT_PATH.write_bytes(raw_output)
    results = {
        item["custom_id"]: item
        for line in raw_output.decode("utf-8").splitlines()
        if (item := json.loads(line))
    }
    translated: dict[str, str] = {}
    errors = []
    for task in manifest["tasks"]:
        result = results.get(task["custom_id"])
        response = result.get("response") if result else None
        if not response or response.get("status_code") != 200:
            errors.append(task["custom_id"])
            continue
        if (
            hashlib.sha256(source_body(task["document_id"]).encode()).hexdigest()
            != task["source_body_sha256"]
        ):
            errors.append(f"changed source: {task['document_id']}")
            continue
        try:
            translated[task["custom_id"]] = output_text(response["body"])
        except (KeyError, ValueError) as error:
            errors.append(f"{task['custom_id']}: {error}")
    if errors or len(translated) != len(manifest["tasks"]):
        raise SystemExit("Refusing partial application: " + "; ".join(errors))

    for task in manifest["tasks"]:
        source_path = PROJECT_ROOT / task["translated_source_path"]
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(translated[task["custom_id"]], encoding="utf-8")
        original_path = PROJECT_ROOT / task["manifest_path"]
        original = yaml.safe_load(original_path.read_text(encoding="utf-8"))
        if original.get("language") != "en":
            raise SystemExit(f"Original publication is not marked en: {original_path}")
        derived = dict(original)
        derived.update(
            {
                "document_id": task["translated_document_id"],
                "source_path": task["translated_source_path"],
                "language": "es",
                "translation_of": task["document_id"],
            }
        )
        target_path = PROJECT_ROOT / task["translated_manifest_path"]
        target_path.write_text(
            yaml.safe_dump(derived, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    print(f"Applied {len(translated)} full-text publication translations.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "submit", "status", "apply"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="OPENAI_TRANSLATE_API_KEY")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST)
    arguments = parser.parse_args()
    if arguments.max_cost_usd <= 0:
        raise SystemExit("--max-cost-usd must be positive.")
    if arguments.command == "prepare":
        prepared = prepare()
        print(f"Prepared {prepared['request_count']} requests.")
        print(
            "Estimated upper-bound cost: "
            f"US${prepared['estimated_cost_usd_upper_bound']:.4f}"
        )
    elif arguments.command == "submit":
        submit(arguments.env_file, arguments.api_key_env, arguments.max_cost_usd)
    elif arguments.command == "status":
        status(arguments.env_file, arguments.api_key_env)
    else:
        apply(arguments.env_file, arguments.api_key_env)


if __name__ == "__main__":
    main()
