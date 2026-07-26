from pathlib import Path

import yaml
from sentence_transformers import SentenceTransformer

from askml_rag.evaluation.retrieval import recall_at_k
from askml_rag.models import Chunk, EvaluationQuestion
from askml_rag.retrieval.bm25 import BM25Retriever
from askml_rag.retrieval.hybrid import HybridRetriever
from askml_rag.retrieval.semantic import SemanticRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "chunks.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.yaml"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K = 3


def load_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as file:
        return [Chunk.model_validate_json(line) for line in file if line.strip()]


def load_questions(path: Path) -> list[EvaluationQuestion]:
    with path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)

    return [
        EvaluationQuestion.model_validate(question) for question in payload["questions"]
    ]


def main() -> None:
    chunks = load_chunks(CHUNKS_PATH)
    questions = load_questions(QUESTIONS_PATH)

    print(f"Loading embedding model: {MODEL_NAME}")
    embedder = SentenceTransformer(MODEL_NAME)

    lexical_retriever = BM25Retriever(chunks)
    semantic_retriever = SemanticRetriever(chunks, embedder)
    retriever = HybridRetriever(
        lexical_retriever,
        semantic_retriever,
        candidate_limit=10,
    )

    scoreable_questions = [
        question
        for question in questions
        if question.answerable and question.relevant_chunk_ids
    ]

    scores: list[float] = []

    for question in scoreable_questions:
        results = retriever.search(question.question, limit=K)
        score = recall_at_k(question, results, k=K)
        scores.append(score)

        print(f"\n{question.question_id}: Recall@{K} = {score:.0f}")
        print(f"Question: {question.question}")
        print("Retrieved:", [chunk.chunk_id for chunk in results])
        print("Expected:", question.relevant_chunk_ids)

    average_recall = sum(scores) / len(scores) if scores else 0.0

    print("\n--- Hybrid retrieval baseline ---")
    print("Methods: BM25 + all-MiniLM-L6-v2 with RRF")
    print(f"Evaluable answerable questions: {len(scoreable_questions)}")
    print(f"Average Recall@{K}: {average_recall:.3f}")


if __name__ == "__main__":
    main()
