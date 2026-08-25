from types import SimpleNamespace

import pytest

from askml_rag.generation.grounded import GenerationRequest
from askml_rag.generation.openai_provider import (
    DEFAULT_MODEL,
    OpenAIParsedResponse,
    OpenAIResponsesLLM,
)


class FakeResponsesAPI:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_parsed=OpenAIParsedResponse(
                answerable=True,
                answer="Marcelo built reproducible pipelines.",
                claims=[
                    {
                        "text": "Marcelo built reproducible pipelines.",
                        "citation_ids": ["retrieved_chunk_001"],
                    }
                ],
                citations=[
                    {
                        "chunk_id": "retrieved_chunk_001",
                        "quote": "built reproducible pipelines",
                    }
                ],
                limitations=[],
            )
        )


class FlakyResponsesAPI(FakeResponsesAPI):
    def parse(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            raise TimeoutError("temporary provider timeout")
        return SimpleNamespace(
            output_parsed=OpenAIParsedResponse(
                answerable=True,
                answer="Marcelo built reproducible pipelines.",
                claims=[
                    {
                        "text": "Marcelo built reproducible pipelines.",
                        "citation_ids": ["retrieved_chunk_001"],
                    }
                ],
                citations=[
                    {
                        "chunk_id": "retrieved_chunk_001",
                        "quote": "built reproducible pipelines",
                    }
                ],
                limitations=[],
            )
        )


def make_request() -> GenerationRequest:
    return GenerationRequest(
        question="What did Marcelo build?",
        prompt="Only use chunk retrieved_chunk_001.",
        prompt_version="grounded-answer-v1",
        context=[
            {
                "chunk_id": "retrieved_chunk_001",
                "document_id": "website_projects",
                "title": "Selected projects",
                "source_url": "https://www.mlares.space/projects/",
                "text": "Marcelo built reproducible pipelines.",
            }
        ],
    )


def test_openai_provider_sends_prompt_and_pydantic_schema() -> None:
    responses_api = FakeResponsesAPI()
    client = SimpleNamespace(responses=responses_api)
    llm = OpenAIResponsesLLM(client=client, model="test-model")

    answer = llm.generate(make_request())

    assert answer.answerable is True
    assert answer.citations[0].chunk_id == "retrieved_chunk_001"
    assert responses_api.calls == [
        {
            "model": "test-model",
            "input": [
                {"role": "user", "content": "Only use chunk retrieved_chunk_001."}
            ],
            "text_format": OpenAIParsedResponse,
            "max_output_tokens": 600,
        }
    ]


def test_openai_provider_retries_once_after_a_generation_failure() -> None:
    responses_api = FlakyResponsesAPI()
    llm = OpenAIResponsesLLM(client=SimpleNamespace(responses=responses_api))

    answer = llm.generate(make_request())

    assert answer.answerable is True
    assert len(responses_api.calls) == 2


def test_openai_provider_requires_a_key_without_an_injected_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        OpenAIResponsesLLM()


def test_openai_provider_uses_a_cost_conscious_default_model() -> None:
    llm = OpenAIResponsesLLM(client=SimpleNamespace(responses=FakeResponsesAPI()))

    assert llm.model_version == DEFAULT_MODEL


def test_openai_provider_rejects_a_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout_seconds must be greater than zero"):
        OpenAIResponsesLLM(
            client=SimpleNamespace(responses=FakeResponsesAPI()),
            timeout_seconds=0,
        )


def test_openai_provider_rejects_a_non_positive_output_cap() -> None:
    with pytest.raises(ValueError, match="max_output_tokens must be greater than zero"):
        OpenAIResponsesLLM(
            client=SimpleNamespace(responses=FakeResponsesAPI()),
            max_output_tokens=0,
        )
