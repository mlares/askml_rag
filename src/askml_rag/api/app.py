"""FastAPI application factory for grounded question answering."""

import asyncio
import logging
import math
import os
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from askml_rag.config import load_dotenv
from askml_rag.evaluation.runner import load_chunks
from askml_rag.generation.grounded import (
    GroundedAnswer,
    GroundedGenerator,
    PROMPT_VERSION,
)
from askml_rag.generation.openai_provider import OpenAIResponsesLLM
from askml_rag.models import Language, RetrievalFilters
from askml_rag.retrieval.hybrid import Retriever
from askml_rag.retrieval.planned_bm25 import PlannedBM25Retriever


DEFAULT_CHUNKS_PATH = Path("data/processed/chunks/chunks.jsonl")
RETRIEVAL_LIMIT = 7
STATIC_DIRECTORY = Path(__file__).parent / "static"
DEFAULT_BOOKING_URL = "https://www.mlares.space/contact/"
LOGGER = logging.getLogger("askml_rag.api")
GENERATION_FALLBACK_MESSAGES = {
    Language.english: "The source documents do not include information related to this question.",
    Language.spanish: "Los documentos fuente no incluyen información relacionada con esta pregunta.",
}

SPANISH_LANGUAGE_SIGNALS = frozenset(
    {
        "qué", "cómo", "cuál", "cuáles", "tenés", "tienes", "trabajaste",
        "investigación", "docencia", "materia", "curso", "podés", "puedes",
        "sobre", "con", "para",
    }
)
ENGLISH_LANGUAGE_SIGNALS = frozenset(
    {
        "what", "which", "how", "have", "has", "you", "your", "worked",
        "research", "teaching", "course", "about", "with", "for", "does", "do",
    }
)


def detect_question_language(question: str) -> Language:
    """Infer English or Spanish for clients that do not supply a language.

    Ambiguous or mixed short questions retain the Spanish default used by the
    public UI. The browser also updates its compact language selector while a
    visitor types, so normal browser requests remain explicit and inspectable.
    """
    lowered = question.casefold()
    tokens = set(re.findall(r"\b\w+\b", lowered, flags=re.UNICODE))
    spanish_score = len(tokens & SPANISH_LANGUAGE_SIGNALS)
    english_score = len(tokens & ENGLISH_LANGUAGE_SIGNALS)
    if any(character in lowered for character in "áéíóúñ¿¡"):
        spanish_score += 1
    if english_score > spanish_score:
        return Language.english
    return Language.spanish


@dataclass(frozen=True)
class OperationalSettings:
    """Deliberately small, local defaults for the public demonstration API."""

    rate_limit_requests: int = 10
    rate_limit_window_seconds: float = 60.0
    request_timeout_seconds: float = 25.0
    max_request_bytes: int = 4_096

    def __post_init__(self) -> None:
        if self.rate_limit_requests <= 0:
            raise ValueError("rate_limit_requests must be greater than zero.")
        if self.rate_limit_window_seconds <= 0:
            raise ValueError("rate_limit_window_seconds must be greater than zero.")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero.")
        if self.max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be greater than zero.")


class InMemoryRateLimiter:
    """Thread-safe fixed-window protection for one application process."""

    def __init__(self, settings: OperationalSettings) -> None:
        self.settings = settings
        self.requests_by_client: dict[str, deque[float]] = defaultdict(deque)
        self.lock = Lock()

    def allow(self, client_key: str) -> tuple[bool, int]:
        """Return whether one request is permitted and a retry delay if not."""
        now = time.monotonic()
        cutoff = now - self.settings.rate_limit_window_seconds
        with self.lock:
            timestamps = self.requests_by_client[client_key]
            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()
            if len(timestamps) >= self.settings.rate_limit_requests:
                retry_after = math.ceil(
                    self.settings.rate_limit_window_seconds - (now - timestamps[0])
                )
                return False, max(1, retry_after)
            timestamps.append(now)
        return True, 0


class AskRequest(BaseModel):
    """The public input contract for one grounded question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1_000)
    language: Language | None = None

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must contain non-whitespace text.")
        return normalized


class AskResponse(GroundedAnswer):
    """A grounded answer plus enough retrieval metadata for clients and logs."""

    retriever: str = "bm25"
    retrieval_limit: int = RETRIEVAL_LIMIT
    retrieved_chunk_ids: list[str] = Field(default_factory=list)
    query_language: Language


class AskService:
    """Keep the retrieval index and generator alive across HTTP requests."""

    def __init__(
        self,
        retriever: Retriever,
        generator: GroundedGenerator,
        *,
        retrieval_limit: int = RETRIEVAL_LIMIT,
        retriever_name: str = "bm25",
    ) -> None:
        if retrieval_limit <= 0:
            raise ValueError("retrieval_limit must be greater than zero.")
        self.retriever = retriever
        self.generator = generator
        self.retrieval_limit = retrieval_limit
        self.retriever_name = retriever_name

    def ask(self, question: str, language: Language) -> AskResponse:
        retrieved_chunks = self.retriever.search(
            question,
            limit=self.retrieval_limit,
            filters=RetrievalFilters(
                languages=[language],
                include_unknown_language=True,
            ),
        )
        answer = self.generator.answer(
            question,
            retrieved_chunks,
            answer_language=language,
        )
        return AskResponse(
            **answer.model_dump(),
            retriever=self.retriever_name,
            retrieval_limit=self.retrieval_limit,
            retrieved_chunk_ids=[chunk.chunk_id for chunk in retrieved_chunks],
            query_language=language,
        )


def build_default_service() -> AskService:
    """Construct the production service from local corpus files and environment."""
    dotenv_keys = load_dotenv(Path(".env"))
    if "API_KEY" in dotenv_keys and "OPENAI_API_KEY" not in dotenv_keys:
        raise ValueError(
            "Your .env file uses API_KEY. Rename it to OPENAI_API_KEY so the "
            "OpenAI SDK can read it."
        )

    chunks = load_chunks(DEFAULT_CHUNKS_PATH)
    return AskService(
        PlannedBM25Retriever(chunks),
        GroundedGenerator(OpenAIResponsesLLM()),
        retriever_name="planned_bm25",
    )


def generation_fallback_response(
    language: Language,
    service: AskService,
) -> AskResponse:
    """Return a user-safe abstention after the provider retry is exhausted."""
    message = GENERATION_FALLBACK_MESSAGES[language]
    return AskResponse(
        answerable=False,
        answer=message,
        limitations=[message],
        prompt_version=PROMPT_VERSION,
        model_version="generation-unavailable",
        retriever=service.retriever_name,
        retrieval_limit=service.retrieval_limit,
        query_language=language,
    )


def create_app(
    service: AskService | None = None,
    *,
    settings: OperationalSettings | None = None,
) -> FastAPI:
    """Create the application, allowing fake dependencies in integration tests."""
    resolved_service = service or build_default_service()
    resolved_settings = settings or OperationalSettings()
    rate_limiter = InMemoryRateLimiter(resolved_settings)
    app = FastAPI(
        title="AskML RAG API",
        version="0.1.0",
        description="Grounded answers over Marcelo Lares's public corpus.",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

    @app.middleware("http")
    async def add_operational_safeguards(request: Request, call_next: object):
        request_id = str(uuid4())
        request.state.request_id = request_id
        started_at = time.perf_counter()

        if request.url.path == "/ask":
            content_length = request.headers.get("content-length")
            try:
                body_is_too_large = bool(content_length) and int(content_length) > resolved_settings.max_request_bytes
            except ValueError:
                body_is_too_large = True
            if body_is_too_large:
                response = JSONResponse(
                    status_code=413, content={"detail": "Request body is too large."}
                )
            elif request.headers.get("content-type", "").split(";", 1)[0] != "application/json":
                response = JSONResponse(
                    status_code=415,
                    content={"detail": "POST /ask requires application/json."},
                )
            else:
                client_key = request.client.host if request.client else "unknown"
                allowed, retry_after = rate_limiter.allow(client_key)
                if not allowed:
                    response = JSONResponse(
                        status_code=429,
                        content={"detail": "Too many requests. Please try again later."},
                        headers={"Retry-After": str(retry_after)},
                    )
                else:
                    response = await call_next(request)
        else:
            response = await call_next(request)

        duration_ms = (time.perf_counter() - started_at) * 1_000
        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "event=http_request request_id=%s method=%s path=%s status_code=%s "
            "duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    @app.get("/health", include_in_schema=False)
    def health() -> dict[str, str]:
        """Liveness probe: never calls retrieval or the provider."""
        return {"status": "ok"}

    @app.get("/ready", include_in_schema=False)
    def ready() -> dict[str, str]:
        """The service constructor has loaded the corpus and retrieval index."""
        return {"status": "ready", "retriever": resolved_service.retriever_name}

    @app.get("/book", include_in_schema=False)
    def book() -> RedirectResponse:
        """Stable conversion URL; configure BOOKING_URL with Marcelo's Calendly."""
        return RedirectResponse(os.environ.get("BOOKING_URL", DEFAULT_BOOKING_URL))

    @app.get("/", include_in_schema=False)
    def web_ui() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "index.html")

    @app.get("/privacy", include_in_schema=False)
    def privacy_notice() -> FileResponse:
        return FileResponse(STATIC_DIRECTORY / "privacy.html")

    @app.post("/ask", response_model=AskResponse)
    async def ask(payload: AskRequest, http_request: Request) -> AskResponse:
        language = payload.language or detect_question_language(payload.question)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(resolved_service.ask, payload.question, language),
                timeout=resolved_settings.request_timeout_seconds,
            )
        except TimeoutError:
            LOGGER.warning(
                "event=request_timeout request_id=%s",
                http_request.state.request_id,
            )
            raise HTTPException(
                status_code=504,
                detail="The request timed out. Please try again.",
            ) from None
        except Exception as error:
            provider_status = getattr(error, "status_code", None)
            LOGGER.warning(
                "event=generation_failure request_id=%s error_type=%s provider_status=%s",
                http_request.state.request_id,
                type(error).__name__,
                provider_status if provider_status is not None else "unknown",
            )
            return generation_fallback_response(language, resolved_service)

    return app
