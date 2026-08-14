"""Ask the local BM25 corpus through the grounded OpenAI adapter."""

import argparse
from pathlib import Path

from askml_rag.config import load_dotenv
from askml_rag.evaluation.runner import load_chunks
from askml_rag.generation.grounded import GroundedGenerator
from askml_rag.generation.openai_provider import DEFAULT_MODEL, OpenAIResponsesLLM
from askml_rag.retrieval.bm25 import BM25Retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the local public corpus with BM25 retrieval and OpenAI generation."
    )
    parser.add_argument("question")
    parser.add_argument(
        "--chunks-path",
        default="data/processed/chunks/chunks.jsonl",
        help="Path to the generated full-text chunk corpus.",
    )
    parser.add_argument("--limit", type=int, default=7, help="BM25 chunk limit.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dotenv_keys = load_dotenv(Path(".env"))
    if "API_KEY" in dotenv_keys and "OPENAI_API_KEY" not in dotenv_keys:
        raise ValueError(
            "Your .env file uses API_KEY. Rename it to OPENAI_API_KEY so the "
            "OpenAI SDK can read it."
        )
    chunks = load_chunks(Path(args.chunks_path))
    retrieved_chunks = BM25Retriever(chunks).search(args.question, limit=args.limit)
    generator = GroundedGenerator(OpenAIResponsesLLM(model=args.model))
    answer = generator.answer(args.question, retrieved_chunks)
    print(answer.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
