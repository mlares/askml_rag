import time

from fastapi.testclient import TestClient

from askml_rag.api.app import (
    AskResponse,
    AskService,
    OperationalSettings,
    create_app,
    detect_question_language,
)
from askml_rag.generation.grounded import (
    GroundedGenerator,
    LLMGenerationResponse,
    StaticLLM,
)
from askml_rag.models import Chunk, Language
from askml_rag.retrieval.bm25 import BM25Retriever


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id="website_projects_chunk_001",
        document_id="website_projects",
        chunk_index=0,
        document_content_hash="a" * 64,
        title="Selected projects",
        document_type="website",
        source_url="https://www.mlares.space/projects/",
        language=Language.spanish,
        text="Marcelo built reproducible data-processing pipelines with validation rules.",
    )


def make_decoy_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.removesuffix("_chunk_001"),
        chunk_index=0,
        document_content_hash="b" * 64,
        title="Unrelated source",
        document_type="website",
        text=text,
    )


def make_client(settings: OperationalSettings | None = None) -> TestClient:
    llm = StaticLLM(
        LLMGenerationResponse(
            answerable=True,
            answer="Marcelo built reproducible data-processing pipelines.",
            claims=[
                {
                    "text": "Marcelo built reproducible data-processing pipelines.",
                    "citation_ids": ["website_projects_chunk_001"],
                }
            ],
            citations=[
                {
                    "chunk_id": "website_projects_chunk_001",
                    "quote": "built reproducible data-processing pipelines",
                }
            ],
        )
    )
    service = AskService(
        BM25Retriever(
            [
                make_chunk(),
                make_decoy_chunk(
                    "teaching_chunk_001", "This source describes teaching."
                ),
                make_decoy_chunk(
                    "biography_chunk_001", "This source contains a biography."
                ),
            ]
        ),
        GroundedGenerator(llm),
    )
    return TestClient(create_app(service, settings=settings))


def test_post_ask_runs_bm25_and_returns_a_grounded_answer() -> None:
    response = make_client().post(
        "/ask",
        json={
            "question": "Which reproducible data-processing pipelines did Marcelo build?",
            "language": "es",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answerable"] is True
    assert payload["retriever"] == "bm25"
    assert payload["retrieval_limit"] == 7
    assert payload["query_language"] == "es"
    assert payload["retrieved_chunk_ids"] == ["website_projects_chunk_001"]
    assert payload["citations"][0]["chunk_id"] == "website_projects_chunk_001"


def test_get_root_serves_the_web_interface() -> None:
    response = make_client().get("/")

    assert response.status_code == 200
    assert "AskML · Marcelo Lares" in response.text
    assert "Ask Marcelo" not in response.text
    assert "/static/app.js" in response.text
    assert "https://www.mlares.space/" in response.text
    assert 'id="example-prompts"' in response.text
    assert 'name="language"' in response.text
    assert 'value="es" checked' in response.text
    assert 'class="question-workspace"' in response.text
    assert 'class="prompt-carousel"' in response.text
    assert "Evidence used" not in response.text
    assert "How this assistant works" not in response.text
    assert 'href="/privacy"' in response.text
    assert 'href="/book"' in response.text


def test_health_ready_privacy_and_booking_routes_do_not_call_the_model() -> None:
    client = make_client()

    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert "Privacy and data use" in client.get("/privacy").text
    booking = client.get("/book", follow_redirects=False)
    assert booking.status_code == 307
    assert booking.headers["location"] == "https://www.mlares.space/contact/"


def test_get_static_javascript_serves_the_api_client() -> None:
    response = make_client().get("/static/app.js")

    assert response.status_code == 200
    assert 'fetch("/ask"' in response.text
    assert "language: selectedLanguage()" in response.text
    assert "applyInterfaceLanguage" in response.text
    assert "renderPrompts" in response.text
    assert "emptyState.hidden = true" in response.text
    assert 'payload.model_version === "generation-unavailable"' in response.text


def test_static_styles_keep_hidden_empty_state_invisible() -> None:
    response = make_client().get("/static/styles.css")

    assert response.status_code == 200
    assert ".empty-state[hidden] { display:none; }" in response.text


def test_post_ask_rejects_invalid_request_data() -> None:
    response = make_client().post("/ask", json={"question": "   "})

    assert response.status_code == 422
    assert "question must contain non-whitespace text" in response.text


def test_post_ask_rejects_unexpected_request_fields() -> None:
    response = make_client().post(
        "/ask",
        json={
            "question": "Which reproducible data-processing pipelines did Marcelo build?",
            "limit": 20,
        },
    )

    assert response.status_code == 422


def test_post_ask_rejects_non_json_and_oversized_bodies() -> None:
    client = make_client(OperationalSettings(max_request_bytes=40))

    assert client.post("/ask", content="question=hello").status_code == 415
    assert (
        client.post(
            "/ask",
            json={"question": "a" * 100},
        ).status_code
        == 413
    )


def test_post_ask_filters_to_the_requested_language() -> None:
    client = make_client()

    response = client.post(
        "/ask",
        json={
            "question": "Which reproducible data-processing pipelines did Marcelo build?",
            "language": "en",
        },
    )

    assert response.status_code == 200
    assert response.json()["retrieved_chunk_ids"] == []
    assert response.json()["query_language"] == "en"


def test_language_detection_uses_question_signals_for_unspecified_requests() -> None:
    assert detect_question_language("Have you worked at CONICET?") == Language.english
    assert detect_question_language("¿Trabajaste en CONICET?") == Language.spanish


def test_post_ask_rate_limits_one_client() -> None:
    client = make_client(
        OperationalSettings(
            rate_limit_requests=1,
            rate_limit_window_seconds=60,
            request_timeout_seconds=1,
        )
    )
    payload = {
        "question": "Which reproducible data-processing pipelines did Marcelo build?"
    }

    assert client.post("/ask", json=payload).status_code == 200
    limited = client.post("/ask", json=payload)

    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "60"
    assert limited.headers["X-Request-ID"]


def test_post_ask_returns_an_abstention_when_generation_fails() -> None:
    class FailedService:
        retriever_name = "planned_bm25"
        retrieval_limit = 7

        def ask(self, question: str, language: Language) -> AskResponse:
            raise RuntimeError((question, language))

    client = TestClient(create_app(FailedService()))  # type: ignore[arg-type]

    response = client.post(
        "/ask",
        json={"question": "What did Marcelo build?", "language": "en"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["answerable"] is False
    assert payload["limitations"] == [
        "The answer could not be generated or validated. Please try again."
    ]
    assert payload["model_version"] == "generation-unavailable"


def test_request_logging_omits_question_and_headers(caplog) -> None:
    secret_question = "This text must not appear in logs."

    with caplog.at_level("INFO", logger="askml_rag.api"):
        response = make_client().post(
            "/ask",
            json={"question": secret_question},
            headers={"Authorization": "Bearer must-not-appear"},
        )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"]
    log_output = caplog.text
    assert "event=http_request" in log_output
    assert secret_question not in log_output
    assert "must-not-appear" not in log_output


def test_post_ask_times_out_before_a_slow_service_finishes() -> None:
    class SlowService:
        def ask(self, question: str, language: Language) -> AskResponse:
            time.sleep(0.05)
            raise AssertionError((question, language))

    client = TestClient(
        create_app(
            SlowService(),  # type: ignore[arg-type]
            settings=OperationalSettings(
                rate_limit_requests=10,
                rate_limit_window_seconds=60,
                request_timeout_seconds=0.001,
            ),
        )
    )

    response = client.post("/ask", json={"question": "Will this timeout?"})

    assert response.status_code == 504
    assert response.headers["X-Request-ID"]
