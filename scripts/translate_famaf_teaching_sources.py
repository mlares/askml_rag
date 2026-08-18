#!/usr/bin/env python3
"""Prepare, submit, inspect, and apply bilingual FAMAF teaching translations."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from translate_bilingual_dataset import (
    DEFAULT_MAX_COST,
    INPUT_PRICE,
    MODEL,
    OUTPUT_PRICE,
    Source,
    client,
    estimate_tokens,
    output_text,
    request,
    source_prompt,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "famaf_teaching_translation"
INPUT_PATH = ARTIFACT_DIR / "batch_input.jsonl"
TASKS_PATH = ARTIFACT_DIR / "tasks.json"
STATE_PATH = ARTIFACT_DIR / "batch_state.json"
OUTPUT_PATH = ARTIFACT_DIR / "batch_output.jsonl"

SOURCES = (
    Source(
        "famaf_teaching_activity",
        "data/processed/famaf_teaching_activity.json",
        "en",
        "famaf_teaching_activity_es",
        "data/raw/famaf_teaching_activity_es.md",
        "Actividad docente en FaMAF — Marcelo Lares",
        "data/manifests/famaf_teaching_activity.yaml",
    ),
    Source(
        "famaf_program_algebra_i_2026",
        "data/processed/famaf_program_algebra_i_2026.json",
        "es",
        "famaf_program_algebra_i_2026_en",
        "data/raw/famaf_program_algebra_i_2026_en.md",
        "Programme: Algebra I",
        "data/manifests/famaf_program_algebra_i_2026.yaml",
    ),
    Source(
        "famaf_program_analisis_matematico_iii_2026",
        "data/processed/famaf_program_analisis_matematico_iii_2026.json",
        "es",
        "famaf_program_analisis_matematico_iii_2026_en",
        "data/raw/famaf_program_analisis_matematico_iii_2026_en.md",
        "Programme: Mathematical Analysis III",
        "data/manifests/famaf_program_analisis_matematico_iii_2026.yaml",
    ),
    Source(
        "famaf_program_ciencia_datos_2026",
        "data/processed/famaf_program_ciencia_datos_2026.json",
        "es",
        "famaf_program_ciencia_datos_2026_en",
        "data/raw/famaf_program_ciencia_datos_2026_en.md",
        "Programme: Data Science (Bachelor's in Applied Mathematics)",
        "data/manifests/famaf_program_ciencia_datos_2026.yaml",
    ),
    Source(
        "famaf_program_computacion_2026",
        "data/processed/famaf_program_computacion_2026.json",
        "es",
        "famaf_program_computacion_2026_en",
        "data/raw/famaf_program_computacion_2026_en.md",
        "Programme: Computation (Physics and Mathematics Teacher-Training Programmes)",
        "data/manifests/famaf_program_computacion_2026.yaml",
    ),
    Source(
        "famaf_program_introduccion_fisica_2026",
        "data/processed/famaf_program_introduccion_fisica_2026.json",
        "es",
        "famaf_program_introduccion_fisica_2026_en",
        "data/raw/famaf_program_introduccion_fisica_2026_en.md",
        "Programme: Introduction to Physics",
        "data/manifests/famaf_program_introduccion_fisica_2026.yaml",
    ),
    Source(
        "famaf_program_machine_learning_2026",
        "data/processed/famaf_program_machine_learning_2026.json",
        "es",
        "famaf_program_machine_learning_2026_en",
        "data/raw/famaf_program_machine_learning_2026_en.md",
        "Programme: Introduction to Machine Learning",
        "data/manifests/famaf_program_machine_learning_2026.yaml",
    ),
    Source(
        "famaf_program_modelos_simulacion_2026",
        "data/processed/famaf_program_modelos_simulacion_2026.json",
        "es",
        "famaf_program_modelos_simulacion_2026_en",
        "data/raw/famaf_program_modelos_simulacion_2026_en.md",
        "Programme: Models and Simulation",
        "data/manifests/famaf_program_modelos_simulacion_2026.yaml",
    ),
    Source(
        "famaf_program_probabilidad_estadistica_2025",
        "data/processed/famaf_program_probabilidad_estadistica_2025.json",
        "es",
        "famaf_program_probabilidad_estadistica_2025_en",
        "data/raw/famaf_program_probabilidad_estadistica_2025_en.md",
        "Programme: Introduction to Probability and Statistics / Probability and Statistics",
        "data/manifests/famaf_program_probabilidad_estadistica_2025.yaml",
    ),
)


def source_body(source: Source) -> str:
    """Load normalized canonical text rather than translating a PDF binary."""
    payload = json.loads(
        (PROJECT_ROOT / source.source_path).read_text(encoding="utf-8")
    )
    body = payload.get("body_markdown")
    if not isinstance(body, str) or not body.strip():
        raise ValueError(f"Canonical source has no body_markdown: {source.source_path}")
    return body


def prepare() -> dict[str, Any]:
    requests: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    input_tokens = output_tokens = 0
    for source in SOURCES:
        body = source_body(source)
        prompt = source_prompt(source, body)
        maximum = estimate_tokens(body) + 1024
        custom_id = f"source__{source.translated_document_id}"
        requests.append(request(custom_id, prompt, maximum))
        tasks.append(
            {
                "custom_id": custom_id,
                "source": asdict(source),
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
        raise SystemExit("No prepared job. Run prepare first.")
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def submit(env_file: str, api_key_env: str, max_cost: float) -> None:
    manifest = load_manifest()
    estimate = float(manifest["estimated_cost_usd_upper_bound"])
    if estimate > max_cost:
        raise SystemExit(
            f"Refusing to submit: estimated US${estimate:.4f} exceeds US${max_cost:.2f}."
        )
    openai = client(env_file, api_key_env)
    with INPUT_PATH.open("rb") as handle:
        uploaded = openai.files.create(file=handle, purpose="batch")
    batch = openai.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"project": "askml-rag", "purpose": "famaf-teaching-bilingual"},
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
    print(f"Estimated upper-bound cost: US${estimate:.4f}")


def status(env_file: str, api_key_env: str) -> None:
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    batch = client(env_file, api_key_env).batches.retrieve(state["batch_id"])
    print(
        json.dumps(
            {
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
    manifest = load_manifest()
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    openai = client(env_file, api_key_env)
    batch = openai.batches.retrieve(state["batch_id"])
    if batch.status != "completed" or not batch.output_file_id:
        raise SystemExit(f"Batch is not ready (status: {batch.status}).")
    raw_output = openai.files.content(batch.output_file_id).read()
    OUTPUT_PATH.write_bytes(raw_output)
    results = {
        json.loads(line)["custom_id"]: json.loads(line)
        for line in raw_output.decode("utf-8").splitlines()
    }
    for task in manifest["tasks"]:
        result = results.get(task["custom_id"])
        response = result.get("response") if result else None
        if not response or response.get("status_code") != 200:
            raise SystemExit(f"Refusing partial application: {task['custom_id']}")
        source = Source(**task["source"])
        if (
            hashlib.sha256(source_body(source).encode()).hexdigest()
            != task["source_body_sha256"]
        ):
            raise SystemExit(f"Refusing to apply changed source: {source.document_id}")
        translated_path = PROJECT_ROOT / source.translated_path
        translated_path.parent.mkdir(parents=True, exist_ok=True)
        translated_path.write_text(output_text(response["body"]), encoding="utf-8")
        original_manifest = yaml.safe_load(
            (PROJECT_ROOT / source.manifest_path).read_text(encoding="utf-8")
        )
        original_manifest.update(
            {
                "document_id": source.translated_document_id,
                "title": source.translated_title,
                "source_path": source.translated_path,
                "language": source.target_language,
                "translation_of": source.document_id,
            }
        )
        target_manifest = (PROJECT_ROOT / source.manifest_path).with_name(
            Path(source.manifest_path).stem + f"_{source.target_language}.yaml"
        )
        target_manifest.write_text(
            yaml.safe_dump(original_manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    print(f"Applied {len(manifest['tasks'])} translations.")


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
