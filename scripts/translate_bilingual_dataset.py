#!/usr/bin/env python3
"""Prepare, submit, inspect, and apply a bounded Batch API translation job.

It covers curated CV, web, skills, and personal-traits sources only: never
individual paper PDFs.  The result is a derived retrieval translation, not an
independent source of evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "reports" / "bilingual_translation"
INPUT_PATH = ARTIFACT_DIR / "batch_input.jsonl"
TASKS_PATH = ARTIFACT_DIR / "tasks.json"
STATE_PATH = ARTIFACT_DIR / "batch_state.json"
OUTPUT_PATH = ARTIFACT_DIR / "batch_output.jsonl"
MODEL = "gpt-5.6-luna"
INPUT_PRICE = 0.10  # Batch price, USD per million tokens.
OUTPUT_PRICE = 0.60  # Batch price, USD per million tokens.
DEFAULT_MAX_COST = 0.50


@dataclass(frozen=True)
class Source:
    document_id: str
    source_path: str
    language: Literal["en", "es"]
    translated_document_id: str
    translated_path: str
    translated_title: str
    manifest_path: str

    @property
    def target_language(self) -> Literal["en", "es"]:
        return "es" if self.language == "en" else "en"


SOURCES = (
    Source("cv_expertise", "sources/public/00_perfil_y_experticia.md", "es", "cv_expertise_en", "sources/public/00_perfil_y_experticia_en.md", "Public profile and expertise", "data/manifests/00_perfil_y_experticia.yaml"),
    Source("cv_education", "sources/public/01_formacion.md", "es", "cv_education_en", "sources/public/01_formacion_en.md", "Education, studies and certifications", "data/manifests/01_formacion.yaml"),
    Source("cv_rrhh", "sources/public/02_cargos_y_rrhh.md", "es", "cv_rrhh_en", "sources/public/02_cargos_y_rrhh_en.md", "Academic positions, employment and mentoring", "data/manifests/02_cargos_y_rrhh.yaml"),
    Source("cv_projects", "sources/public/03_proyectos_y_financiamiento.md", "es", "cv_projects_en", "sources/public/03_proyectos_y_financiamiento_en.md", "Projects and research funding", "data/manifests/03_proyectos_y_financiamiento.yaml"),
    Source("cv_outreach", "sources/public/05_evaluacion.md", "es", "cv_outreach_en", "sources/public/05_evaluacion_en.md", "Academic evaluation and outreach", "data/manifests/05_evaluacion.yaml"),
    Source("cv_papers", "sources/public/06_produccion_y_publicaciones.md", "es", "cv_papers_en", "sources/public/06_produccion_y_publicaciones_en.md", "Publication record", "data/manifests/06_produccion_y_publicaciones.yaml"),
    Source("cv_meetings", "sources/public/07_servicios_y_redes.md", "es", "cv_meetings_en", "sources/public/07_servicios_y_redes_en.md", "Services, professional networks and meetings", "data/manifests/07_servicios_y_redes.yaml"),
    Source("personal_traits", "data/raw/personal_traits.md", "es", "personal_traits_en", "data/raw/personal_traits_en.md", "Personal profile and collaboration style", "data/manifests/personal_traits.yaml"),
    Source("skills", "data/raw/skills.md", "en", "skills_es", "data/raw/skills_es.md", "Marcelo Lares — Perfil de habilidades y palabras clave de recuperación", "data/manifests/skills.yaml"),
    Source("website_home", "data/raw/website_home.md", "en", "website_home_es", "data/raw/website_home_es.md", "Marcelo Lares", "data/manifests/website_home.yaml"),
    Source("website_projects", "data/raw/website_projects.md", "en", "website_projects_es", "data/raw/website_projects_es.md", "Proyectos seleccionados", "data/manifests/website_projects.yaml"),
    Source("website_research", "data/raw/website_research.md", "en", "website_research_es", "data/raw/website_research_es.md", "Investigación", "data/manifests/website_research.yaml"),
    Source("website_teaching", "data/raw/website_teaching.md", "en", "website_teaching_es", "data/raw/website_teaching_es.md", "Docencia y mentoría", "data/manifests/website_teaching.yaml"),
)


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"Environment file not found: {path.resolve()}")
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip().removeprefix("export ").lstrip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise SystemExit(f"Invalid .env entry at {path}:{number}")
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def estimate_tokens(text: str) -> int:
    """Conservative estimate used only for refusing an over-budget submission."""
    return math.ceil(len(text) / 3)


def source_prompt(source: Source, body: str) -> str:
    names = {"en": "English", "es": "Spanish"}
    return (
        f"Translate this curated public-professional source from {names[source.language]} "
        f"to {names[source.target_language]}. Return only translated Markdown, "
        "without a preamble or code fence. Preserve headings, lists, tables, "
        "links, URLs, dates, IDs, numbers, names, institutions, software names, "
        "file paths, and Markdown syntax. Translate prose and labels naturally. "
        "Do not invent, omit, summarize, or merge facts. Do not translate cited "
        "publication titles, DOIs, arXiv IDs, or code. This is a derived retrieval "
        "translation, not a new source of evidence.\n\n--- BEGIN SOURCE ---\n"
        f"{body}\n--- END SOURCE ---\n"
    )


def questions_prompt(body: str) -> str:
    return (
        "Translate this evaluation-question YAML from English to Spanish. Return "
        "only valid YAML. Preserve the root key `questions`, all YAML keys, IDs, "
        "categories, answer modes, booleans, numeric values, document IDs, and "
        "chunk IDs exactly. Translate only `question`, `expected_claims[].claim`, "
        "and `notes`. Do not add/remove questions, claims, or labels.\n\n"
        "--- BEGIN YAML ---\n" + body + "\n--- END YAML ---\n"
    )


def request(custom_id: str, prompt: str, max_output_tokens: int) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "input": [{"role": "user", "content": prompt}],
            "max_output_tokens": max_output_tokens,
        },
    }


def build_tasks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    input_tokens = output_tokens = 0
    for source in SOURCES:
        body = (PROJECT_ROOT / source.source_path).read_text(encoding="utf-8")
        prompt = source_prompt(source, body)
        maximum = estimate_tokens(body) + 1024
        custom_id = f"source__{source.translated_document_id}"
        requests.append(request(custom_id, prompt, maximum))
        tasks.append({"custom_id": custom_id, "kind": "source", "source": asdict(source), "source_sha256": hashlib.sha256(body.encode()).hexdigest(), "max_output_tokens": maximum})
        input_tokens += estimate_tokens(prompt)
        output_tokens += maximum

    questions_path = PROJECT_ROOT / "data/evaluation/questions.yaml"
    questions = questions_path.read_text(encoding="utf-8")
    prompt = questions_prompt(questions)
    maximum = estimate_tokens(questions) + 1024
    requests.append(request("evaluation_questions_es", prompt, maximum))
    tasks.append({"custom_id": "evaluation_questions_es", "kind": "questions", "source_path": "data/evaluation/questions.yaml", "target_path": "data/evaluation/questions_es.draft.yaml", "source_sha256": hashlib.sha256(questions.encode()).hexdigest(), "max_output_tokens": maximum})
    input_tokens += estimate_tokens(prompt)
    output_tokens += maximum
    cost = (input_tokens * INPUT_PRICE + output_tokens * OUTPUT_PRICE) / 1_000_000
    return requests, {"schema_version": "1", "model": MODEL, "request_count": len(requests), "estimated_input_tokens_upper_bound": input_tokens, "estimated_output_tokens_upper_bound": output_tokens, "estimated_cost_usd_upper_bound": cost, "tasks": tasks}


def prepare() -> dict[str, Any]:
    requests, manifest = build_tasks()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_PATH.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in requests), encoding="utf-8")
    TASKS_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def load_manifest() -> dict[str, Any]:
    if not INPUT_PATH.is_file() or not TASKS_PATH.is_file():
        raise SystemExit("No prepared job. Run `prepare` first.")
    return json.loads(TASKS_PATH.read_text(encoding="utf-8"))


def client(env_file: str, api_key_env: str) -> Any:
    load_dotenv(PROJECT_ROOT / env_file)
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"{api_key_env} is not set.")
    from openai import OpenAI
    return OpenAI(api_key=api_key)


def submit(env_file: str, api_key_env: str, max_cost: float) -> None:
    manifest = load_manifest()
    estimated = float(manifest["estimated_cost_usd_upper_bound"])
    if estimated > max_cost:
        raise SystemExit(f"Refusing to submit: estimated US${estimated:.4f} exceeds US${max_cost:.2f}.")
    openai = client(env_file, api_key_env)
    try:
        with INPUT_PATH.open("rb") as handle:
            uploaded = openai.files.create(file=handle, purpose="batch")
        batch = openai.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"project": "askml-rag", "purpose": "bilingual-corpus"},
        )
    except Exception as error:
        message = str(error)
        if "api.files.write" in message:
            raise SystemExit(
                "The API key cannot upload Batch input files. Create or update a "
                "project key with Files write/read and Batch/Responses permissions, "
                "then update the selected key environment variable and run submit again."
            ) from None
        raise
    STATE_PATH.write_text(json.dumps({"batch_id": batch.id, "input_file_id": uploaded.id, "max_cost_usd": max_cost, "estimated_cost_usd_upper_bound": estimated}, indent=2) + "\n", encoding="utf-8")
    print(f"Submitted batch: {batch.id}")
    print(f"Requests: {manifest['request_count']}")
    print(f"Estimated upper-bound cost: US${estimated:.4f}")


def load_state() -> dict[str, Any]:
    if not STATE_PATH.is_file():
        raise SystemExit("No submitted batch. Run `submit` after `prepare`.")
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def status(env_file: str, api_key_env: str) -> Any:
    batch = client(env_file, api_key_env).batches.retrieve(load_state()["batch_id"])
    result = {"batch_id": batch.id, "status": batch.status, "output_file_id": batch.output_file_id, "error_file_id": batch.error_file_id, "request_counts": batch.request_counts.model_dump() if batch.request_counts else None}
    print(json.dumps(result, indent=2))
    return batch


def output_text(body: dict[str, Any]) -> str:
    text = "".join(content["text"] for item in body.get("output", []) for content in item.get("content", []) if content.get("type") == "output_text" and isinstance(content.get("text"), str)).strip()
    if not text or text.startswith("```"):
        raise ValueError("Expected unwrapped output_text.")
    return text + "\n"


def original_manifest(source: Source) -> dict[str, Any]:
    path = PROJECT_ROOT / source.manifest_path
    if path.is_file():
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    if source.document_id != "personal_traits":
        raise ValueError(f"Original manifest missing: {path}")
    payload = {"document_id": "personal_traits", "title": "Perfil personal y estilo de colaboración", "document_type": "biography", "authors": ["Marcelo Lares"], "source_path": source.source_path, "language": "es", "visibility": "public", "topics": ["professional-profile", "communication", "leadership"]}
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return payload


def apply(env_file: str, api_key_env: str) -> None:
    manifest, state, openai = load_manifest(), load_state(), client(env_file, api_key_env)
    batch = openai.batches.retrieve(state["batch_id"])
    if batch.status != "completed" or not batch.output_file_id:
        raise SystemExit(f"Batch is not ready (status: {batch.status}).")
    raw_output = openai.files.content(batch.output_file_id).read()
    OUTPUT_PATH.write_bytes(raw_output)
    tasks = {task["custom_id"]: task for task in manifest["tasks"]}
    translated: dict[str, str] = {}
    errors: list[str] = []
    for line in raw_output.decode("utf-8").splitlines():
        result = json.loads(line)
        task = tasks.get(result.get("custom_id"))
        response = result.get("response") or {}
        if task is None or response.get("status_code") != 200:
            errors.append(f"Failed task: {result.get('custom_id')}")
            continue
        try:
            translated[task["custom_id"]] = output_text(response["body"])
        except (KeyError, ValueError) as error:
            errors.append(f"{task['custom_id']}: {error}")
    missing = set(tasks) - set(translated)
    if errors or missing:
        raise SystemExit("Refusing partial application: " + "; ".join(errors + sorted(missing)))

    for task in manifest["tasks"]:
        current_source = PROJECT_ROOT / task["source_path"] if task["kind"] == "questions" else PROJECT_ROOT / task["source"]["source_path"]
        current_hash = hashlib.sha256(current_source.read_bytes()).hexdigest()
        if current_hash != task["source_sha256"]:
            raise SystemExit(
                f"Refusing to apply {task['custom_id']}: its source changed after "
                "the batch was prepared. Prepare and submit a new batch."
            )
        text = translated[task["custom_id"]]
        if task["kind"] == "questions":
            payload = yaml.safe_load(text)
            if not isinstance(payload, dict) or set(payload) != {"questions"}:
                raise SystemExit("Translated question output is not valid questions YAML.")
            (PROJECT_ROOT / task["target_path"]).write_text(text, encoding="utf-8")
            continue
        source = Source(**task["source"])
        (PROJECT_ROOT / source.translated_path).write_text(text, encoding="utf-8")
        derived = original_manifest(source)
        derived.update({"document_id": source.translated_document_id, "title": source.translated_title, "source_path": source.translated_path, "language": source.target_language, "translation_of": source.document_id})
        manifest_path = (PROJECT_ROOT / source.manifest_path).with_name(Path(source.manifest_path).stem + f"_{source.target_language}.yaml")
        manifest_path.write_text(yaml.safe_dump(derived, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Applied {len(translated)} translations.")
    print("Questions are a draft: review exact relevant chunk IDs after re-chunking.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "submit", "status", "apply"))
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key-env", default="OPENAI_TRANSLATE_API_KEY")
    parser.add_argument("--max-cost-usd", type=float, default=DEFAULT_MAX_COST)
    args = parser.parse_args()
    if args.max_cost_usd <= 0:
        raise SystemExit("--max-cost-usd must be positive.")
    if args.command == "prepare":
        manifest = prepare()
        print(f"Prepared {manifest['request_count']} requests.")
        print(f"Estimated upper-bound cost: US${manifest['estimated_cost_usd_upper_bound']:.4f}")
    elif args.command == "submit":
        submit(args.env_file, args.api_key_env, args.max_cost_usd)
    elif args.command == "status":
        status(args.env_file, args.api_key_env)
    else:
        apply(args.env_file, args.api_key_env)


if __name__ == "__main__":
    main()
