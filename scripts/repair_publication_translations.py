#!/usr/bin/env python3
"""Repair incomplete publication translations using bounded text segments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
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
from translate_publication_sources import source_body, translation_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "publication_translation_repair"
INPUT_PATH = ARTIFACT_DIR / "batch_input.jsonl"
TASKS_PATH = ARTIFACT_DIR / "tasks.json"
STATE_PATH = ARTIFACT_DIR / "batch_state.json"
OUTPUT_PATH = ARTIFACT_DIR / "batch_output.jsonl"
SEGMENT_CHARACTERS = 12_000
MINIMUM_LENGTH_RATIO = 0.75


def translated_source_path(document_id: str) -> Path:
    manifest_paths = sorted(
        (PROJECT_ROOT / "data/manifests").glob("arXiv-*_es.yaml")
    )
    for manifest_path in manifest_paths:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("translation_of") == document_id:
            return PROJECT_ROOT / manifest["source_path"]
    raise ValueError(f"No Spanish manifest found for {document_id}")


def incomplete_document_ids() -> list[str]:
    """Identify applied translations too short to represent the full source."""
    incomplete = []
    for path in sorted((PROJECT_ROOT / "data/processed").glob("arxiv_*_es.json")):
        translated = json.loads(path.read_text(encoding="utf-8"))
        original_id = translated.get("translation_of")
        if not original_id:
            continue
        ratio = len(translated["body_markdown"]) / len(source_body(original_id))
        if ratio < MINIMUM_LENGTH_RATIO:
            incomplete.append(original_id)
    return incomplete


def split_text(text: str, *, maximum_characters: int) -> list[str]:
    """Split Markdown at paragraph boundaries under a hard character ceiling."""
    segments: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        pieces = [
            paragraph[start : start + maximum_characters]
            for start in range(0, len(paragraph), maximum_characters)
        ] or [""]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > maximum_characters:
                segments.append(current)
                current = piece
            else:
                current = candidate
    if current:
        segments.append(current)
    return segments


def prepare() -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    input_tokens = output_tokens = 0
    for document_id in incomplete_document_ids():
        body = source_body(document_id)
        segments = split_text(body, maximum_characters=SEGMENT_CHARACTERS)
        for index, segment in enumerate(segments):
            prompt = translation_prompt(segment)
            maximum = estimate_tokens(segment) + 1024
            custom_id = f"repair__{document_id}__{index:03d}"
            requests.append(request(custom_id, prompt, maximum))
            tasks.append(
                {
                    "custom_id": custom_id,
                    "document_id": document_id,
                    "segment_index": index,
                    "segment_count": len(segments),
                    "segment_sha256": hashlib.sha256(segment.encode()).hexdigest(),
                    "max_output_tokens": maximum,
                }
            )
            input_tokens += estimate_tokens(prompt)
            output_tokens += maximum
    manifest = {
        "schema_version": "1",
        "model": MODEL,
        "request_count": len(requests),
        "document_ids": incomplete_document_ids(),
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


def load_json(path: Path, missing_message: str) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(missing_message)
    return json.loads(path.read_text(encoding="utf-8"))


def submit(env_file: str, api_key_env: str, max_cost: float) -> None:
    manifest = load_json(TASKS_PATH, "No prepared repair job.")
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
        metadata={"project": "askml-rag", "purpose": "publication-repair"},
    )
    STATE_PATH.write_text(
        json.dumps(
            {
                "batch_id": batch.id,
                "input_file_id": uploaded.id,
                "estimated_cost_usd_upper_bound": estimate,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Submitted batch: {batch.id}")
    print(f"Requests: {manifest['request_count']}")
    print(f"Estimated upper-bound cost: US${estimate:.4f}")


def get_batch(env_file: str, api_key_env: str) -> tuple[Any, Any]:
    state = load_json(STATE_PATH, "No submitted repair job.")
    openai = client(env_file, api_key_env)
    return openai, openai.batches.retrieve(state["batch_id"])


def status(env_file: str, api_key_env: str) -> None:
    _, batch = get_batch(env_file, api_key_env)
    print(
        json.dumps(
            {
                "batch_id": batch.id,
                "status": batch.status,
                "request_counts": batch.request_counts.model_dump()
                if batch.request_counts
                else None,
                "output_file_id": batch.output_file_id,
            },
            indent=2,
        )
    )


def apply(env_file: str, api_key_env: str) -> None:
    manifest = load_json(TASKS_PATH, "No prepared repair job.")
    openai, batch = get_batch(env_file, api_key_env)
    if batch.status != "completed" or not batch.output_file_id:
        raise SystemExit(f"Batch is not ready (status: {batch.status}).")
    raw_output = openai.files.content(batch.output_file_id).read()
    OUTPUT_PATH.write_bytes(raw_output)
    results = {
        item["custom_id"]: item
        for line in raw_output.decode("utf-8").splitlines()
        if (item := json.loads(line))
    }
    segments: defaultdict[str, dict[int, str]] = defaultdict(dict)
    errors = []
    for task in manifest["tasks"]:
        result = results.get(task["custom_id"])
        response = result.get("response") if result else None
        if not response or response.get("status_code") != 200:
            errors.append(task["custom_id"])
            continue
        original_segments = split_text(
            source_body(task["document_id"]),
            maximum_characters=SEGMENT_CHARACTERS,
        )
        original_segment = original_segments[task["segment_index"]]
        if hashlib.sha256(original_segment.encode()).hexdigest() != task["segment_sha256"]:
            errors.append(f"changed segment: {task['custom_id']}")
            continue
        try:
            segments[task["document_id"]][task["segment_index"]] = output_text(
                response["body"]
            ).strip()
        except (KeyError, ValueError) as error:
            errors.append(f"{task['custom_id']}: {error}")
    if errors:
        raise SystemExit("Refusing partial repair: " + "; ".join(errors))
    for document_id in manifest["document_ids"]:
        expected = len(split_text(source_body(document_id), maximum_characters=SEGMENT_CHARACTERS))
        if set(segments[document_id]) != set(range(expected)):
            raise SystemExit(f"Missing translated segments for {document_id}")
        translated = "\n\n".join(
            segments[document_id][index] for index in range(expected)
        ).strip() + "\n"
        ratio = len(translated) / len(source_body(document_id))
        if ratio < MINIMUM_LENGTH_RATIO:
            raise SystemExit(
                f"Repaired translation remains incomplete: {document_id} ({ratio:.3f})"
            )
        translated_source_path(document_id).write_text(translated, encoding="utf-8")
        print(f"Repaired {document_id}: length ratio {ratio:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "submit", "status", "apply"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="OPENAI_TRANSLATE_API_KEY")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        prepared = prepare()
        print(f"Documents: {', '.join(prepared['document_ids'])}")
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
