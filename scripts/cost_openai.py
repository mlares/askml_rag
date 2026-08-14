#!/usr/bin/env python3
"""Measure token consumption and estimated cost across OpenAI text models.

The script loads OPENAI_API_KEY from a local .env file. By default, it tests a
small set of affordable models. Use --all-priced to test every text-generation
model in PRICE_CATALOG that is also visible to the current API project.

Install:
    python -m pip install --upgrade openai

Examples:
    python openai_model_costs.py --list-models
    python openai_model_costs.py
    python openai_model_costs.py --models gpt-5-mini gpt-5.6-luna
    python openai_model_costs.py --all-priced
    python openai_model_costs.py --prompt "Explain RAG in two sentences."

Important: PRICE_CATALOG is a local snapshot of standard API prices in USD per
1 million tokens. Check https://developers.openai.com/api/docs/pricing before
using the estimates for budgeting, because prices can change.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# USD per 1 million tokens. Updated 2026-08-14 from the official OpenAI model
# and pricing pages. These are text-token rates and exclude separately priced
# tools such as web search, file search, containers, audio, images, and video.
PRICE_CATALOG: dict[str, dict[str, float]] = {
    "gpt-5.6-sol": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.6-terra": {"input": 2.00, "cached_input": 0.20, "output": 12.00},
    "gpt-5.6-luna": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
    "gpt-5.5": {"input": 5.00, "cached_input": 0.50, "output": 30.00},
    "gpt-5.4": {"input": 2.50, "cached_input": 0.25, "output": 15.00},
    "gpt-5.4-mini": {"input": 0.75, "cached_input": 0.075, "output": 4.50},
    "gpt-5.4-nano": {"input": 0.20, "cached_input": 0.02, "output": 1.25},
    "gpt-5.2": {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "cached_input": 0.005, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
}

DEFAULT_MODELS = [
    "gpt-4o-mini",
    "gpt-5.4-nano",
    "gpt-5.6-luna",
    "gpt-5-mini",
]

DEFAULT_PROMPT = (
    "In no more than 80 words, explain what retrieval-augmented generation "
    "(RAG) is and mention one limitation."
)


@dataclass
class Measurement:
    requested_model: str
    response_model: str | None
    status: str
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    estimated_cost_usd: float | None
    latency_seconds: float
    output_text: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the dotenv file (default: .env).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model IDs to test. Overrides the affordable defaults.",
    )
    parser.add_argument(
        "--all-priced",
        action="store_true",
        help=(
            "Test every model in PRICE_CATALOG that is visible to the API "
            "project. This can be expensive."
        ),
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models visible to the API project and exit without testing.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Identical prompt sent to every tested model.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=512,
        help="Maximum generated tokens per request, including reasoning (default: 512).",
    )
    parser.add_argument(
        "--csv",
        default="openai_model_costs.csv",
        help="CSV output path (default: openai_model_costs.csv).",
    )
    parser.add_argument(
        "--json",
        default="openai_model_costs.json",
        help="JSON output path (default: openai_model_costs.json).",
    )
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without adding a dotenv dependency."""
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        if "=" not in line:
            raise SystemExit(f"Invalid .env entry at {path}:{line_number}")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_client(env_file: str) -> Any:
    path = Path(env_file)
    if not path.is_file():
        raise SystemExit(f"Environment file not found: {path.resolve()}")

    load_env_file(path)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit(f"OPENAI_API_KEY is missing from {path.resolve()}")

    # The SDK reads OPENAI_API_KEY from the environment automatically.
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The OpenAI SDK is not installed. Run: "
            "python -m pip install --upgrade openai"
        ) from exc
    return OpenAI()


def visible_model_ids(client: Any) -> set[str]:
    return {model.id for model in client.models.list().data}


def nested_int(obj: Any, attribute: str) -> int:
    if obj is None:
        return 0
    value = getattr(obj, attribute, 0)
    return int(value or 0)


def estimate_cost(
    model: str,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float | None:
    prices = PRICE_CATALOG.get(model)
    if prices is None:
        return None

    uncached_input_tokens = max(input_tokens - cached_input_tokens, 0)
    return (
        uncached_input_tokens * prices["input"]
        + cached_input_tokens * prices["cached_input"]
        + output_tokens * prices["output"]
    ) / 1_000_000


def measure_model(
    client: Any,
    model: str,
    prompt: str,
    max_output_tokens: int,
) -> Measurement:
    started = time.perf_counter()
    try:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
            max_output_tokens=max_output_tokens,
        )
        latency = time.perf_counter() - started
        usage = response.usage

        if usage is None:
            raise RuntimeError("The response did not include a usage object.")

        cached = nested_int(usage.input_tokens_details, "cached_tokens")
        reasoning = nested_int(usage.output_tokens_details, "reasoning_tokens")
        cost = estimate_cost(
            model=model,
            input_tokens=usage.input_tokens,
            cached_input_tokens=cached,
            output_tokens=usage.output_tokens,
        )

        return Measurement(
            requested_model=model,
            response_model=response.model,
            status=response.status or "unknown",
            input_tokens=usage.input_tokens,
            cached_input_tokens=cached,
            output_tokens=usage.output_tokens,
            reasoning_tokens=reasoning,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=cost,
            latency_seconds=latency,
            output_text=response.output_text,
            error="",
        )
    except Exception as exc:  # Continue so one unavailable model does not stop the run.
        return Measurement(
            requested_model=model,
            response_model=None,
            status="error",
            input_tokens=0,
            cached_input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            total_tokens=0,
            estimated_cost_usd=None,
            latency_seconds=time.perf_counter() - started,
            output_text="",
            error=f"{type(exc).__name__}: {exc}",
        )


def write_csv(path: Path, measurements: list[Measurement]) -> None:
    rows = [asdict(item) for item in measurements]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    prompt: str,
    measurements: list[Measurement],
) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pricing_snapshot_date": "2026-08-14",
        "prompt": prompt,
        "measurements": [asdict(item) for item in measurements],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def format_cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.8f}"


def main() -> int:
    args = parse_args()
    client = load_client(args.env_file)

    try:
        visible = visible_model_ids(client)
    except Exception as exc:
        print(f"Could not list models: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.list_models:
        for model_id in sorted(visible):
            marker = "priced" if model_id in PRICE_CATALOG else "unpriced"
            print(f"{model_id}\t{marker}")
        return 0

    if args.models and args.all_priced:
        print("Use either --models or --all-priced, not both.", file=sys.stderr)
        return 2

    if args.models:
        requested_models = args.models
    elif args.all_priced:
        requested_models = sorted(set(PRICE_CATALOG).intersection(visible))
    else:
        requested_models = DEFAULT_MODELS

    if not requested_models:
        print("No models selected.", file=sys.stderr)
        return 2

    unavailable = [model for model in requested_models if model not in visible]
    if unavailable:
        print(
            "Warning: these models were not returned by models.list() and may fail: "
            + ", ".join(unavailable),
            file=sys.stderr,
        )

    measurements: list[Measurement] = []
    for model in requested_models:
        print(f"Testing {model}...", flush=True)
        result = measure_model(
            client=client,
            model=model,
            prompt=args.prompt,
            max_output_tokens=args.max_output_tokens,
        )
        measurements.append(result)
        print(
            f"  status={result.status} input={result.input_tokens} "
            f"output={result.output_tokens} reasoning={result.reasoning_tokens} "
            f"cost={format_cost(result.estimated_cost_usd)} "
            f"latency={result.latency_seconds:.2f}s"
        )
        if result.error:
            print(f"  {result.error}", file=sys.stderr)

    csv_path = Path(args.csv)
    json_path = Path(args.json)
    write_csv(csv_path, measurements)
    write_json(json_path, args.prompt, measurements)

    total_cost = sum(item.estimated_cost_usd or 0.0 for item in measurements)
    print(f"\nEstimated total: ${total_cost:.8f}")
    print(f"CSV:  {csv_path.resolve()}")
    print(f"JSON: {json_path.resolve()}")
    return 0 if all(item.status != "error" for item in measurements) else 1


if __name__ == "__main__":
    raise SystemExit(main())
