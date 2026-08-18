#!/usr/bin/env python3
"""Translate the reviewed IThreex portfolio sources to Spanish with Batch API."""

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
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "ithreex_translation"
INPUT_PATH = ARTIFACT_DIR / "batch_input.jsonl"
TASKS_PATH = ARTIFACT_DIR / "tasks.json"
STATE_PATH = ARTIFACT_DIR / "batch_state.json"
OUTPUT_PATH = ARTIFACT_DIR / "batch_output.jsonl"

SOURCES = (
    Source(
        "ithreex_kolektor_mvp1",
        "data/raw/ithreex_kolektor_mvp1.md",
        "en",
        "ithreex_kolektor_mvp1_es",
        "data/raw/ithreex_kolektor_mvp1_es.md",
        "Predicción de comportamiento de pago y segmentación de contribuyentes (Kolektor MVP I)",
        "data/manifests/ithreex_kolektor_mvp1.yaml",
    ),
    Source(
        "ithreex_kolektor_mvp2",
        "data/raw/ithreex_kolektor_mvp2.md",
        "en",
        "ithreex_kolektor_mvp2_es",
        "data/raw/ithreex_kolektor_mvp2_es.md",
        "Pronóstico mensual de recaudación fiscal y métricas de comportamiento (Kolektor MVP II)",
        "data/manifests/ithreex_kolektor_mvp2.yaml",
    ),
    Source(
        "ithreex_pueblo_nativo",
        "data/raw/ithreex_pueblo_nativo.md",
        "en",
        "ithreex_pueblo_nativo_es",
        "data/raw/ithreex_pueblo_nativo_es.md",
        "Segmentación de clientes y análisis de voz del cliente (Pueblo Nativo)",
        "data/manifests/ithreex_pueblo_nativo.yaml",
    ),
    Source(
        "ithreex_inverfin",
        "data/raw/ithreex_inverfin.md",
        "en",
        "ithreex_inverfin_es",
        "data/raw/ithreex_inverfin_es.md",
        "Enriquecimiento geoespacial y analítica de segmentación de sucursales (Inverfin)",
        "data/manifests/ithreex_inverfin.yaml",
    ),
    Source(
        "ithreex_procordoba",
        "data/raw/ithreex_procordoba.md",
        "en",
        "ithreex_procordoba_es",
        "data/raw/ithreex_procordoba_es.md",
        "Analítica de oportunidades de exportación mediante clustering de productos y países (ProCórdoba)",
        "data/manifests/ithreex_procordoba.yaml",
    ),
    Source(
        "ithreex_ss_servicios",
        "data/raw/ithreex_ss_servicios.md",
        "en",
        "ithreex_ss_servicios_es",
        "data/raw/ithreex_ss_servicios_es.md",
        "Integración de datos de clientes e ingeniería de variables (SS Servicios)",
        "data/manifests/ithreex_ss_servicios.yaml",
    ),
    Source(
        "ithreex_animalia",
        "data/raw/ithreex_animalia.md",
        "en",
        "ithreex_animalia_es",
        "data/raw/ithreex_animalia_es.md",
        "Estimación de peso bovino asistida por visión por computadora (AnimalIA / SmartFarm)",
        "data/manifests/ithreex_animalia.yaml",
    ),
    Source(
        "ithreex_molibdeno",
        "data/raw/ithreex_molibdeno.md",
        "en",
        "ithreex_molibdeno_es",
        "data/raw/ithreex_molibdeno_es.md",
        "Producto y marketplace de modelos de IA Molibdeno",
        "data/manifests/ithreex_molibdeno.yaml",
    ),
)


def prepare() -> dict[str, Any]:
    requests, tasks = [], []
    input_tokens = output_tokens = 0
    for source in SOURCES:
        body = (PROJECT_ROOT / source.source_path).read_text(encoding="utf-8")
        prompt = source_prompt(source, body)
        maximum = estimate_tokens(body) + 1024
        custom_id = f"source__{source.translated_document_id}"
        requests.append(request(custom_id, prompt, maximum))
        tasks.append(
            {
                "custom_id": custom_id,
                "source": asdict(source),
                "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
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
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_manifest() -> dict[str, Any]:
    if not TASKS_PATH.is_file():
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
        metadata={"project": "askml-rag", "purpose": "ithreex-spanish-translation"},
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
        original = PROJECT_ROOT / source.source_path
        if hashlib.sha256(original.read_bytes()).hexdigest() != task["source_sha256"]:
            raise SystemExit(f"Refusing to apply changed source: {source.document_id}")
        translated = output_text(response["body"])
        (PROJECT_ROOT / source.translated_path).write_text(translated, encoding="utf-8")
        original_manifest = yaml.safe_load(
            (PROJECT_ROOT / source.manifest_path).read_text(encoding="utf-8")
        )
        original_manifest.update(
            {
                "document_id": source.translated_document_id,
                "title": source.translated_title,
                "source_path": source.translated_path,
                "language": "es",
                "translation_of": source.document_id,
            }
        )
        target_manifest = (PROJECT_ROOT / source.manifest_path).with_name(
            Path(source.manifest_path).stem + "_es.yaml"
        )
        target_manifest.write_text(
            yaml.safe_dump(original_manifest, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
    print(f"Applied {len(manifest['tasks'])} translations.")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "submit", "status", "apply"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="OPENAI_TRANSLATE_API_KEY")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST)
    args = parser.parse_args()
    if args.command == "prepare":
        manifest = prepare()
        print(f"Prepared {manifest['request_count']} requests.")
        print(
            f"Estimated upper-bound cost: US${manifest['estimated_cost_usd_upper_bound']:.4f}"
        )
    elif args.command == "submit":
        submit(args.env_file, args.api_key_env, args.max_cost_usd)
    elif args.command == "status":
        status(args.env_file, args.api_key_env)
    else:
        apply(args.env_file, args.api_key_env)


if __name__ == "__main__":
    main()
