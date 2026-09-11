"""OpenAI Responses API adapter for the grounded-generation contract."""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict

from askml_rag.generation.grounded import (
    CitationReference,
    GeneratedClaim,
    GenerationRequest,
    LLMGenerationResponse,
)


# DEFAULT_MODEL = "gpt-5-mini"
# DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_MODEL = "gpt-5.6-luna"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_OUTPUT_TOKENS = 1_200
MAX_GENERATION_ATTEMPTS = 2


class OpenAIParsedResponse(BaseModel):
    """All fields are required for OpenAI Structured Outputs strict mode."""

    model_config = ConfigDict(extra="forbid")

    answerable: bool
    answer: str
    claims: list[GeneratedClaim]
    citations: list[CitationReference]
    limitations: list[str]


class OpenAIResponsesLLM:
    """Adapt OpenAI structured responses to the local `LanguageModel` protocol."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        client: Any | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be greater than zero.")
        self.model_version = model
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        if client is not None:
            self.client = client
            return

        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Export it in your shell before "
                "running a real OpenAI request."
            )

        from openai import OpenAI

        self.client = OpenAI(
            api_key=resolved_api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def generate(self, request: GenerationRequest) -> LLMGenerationResponse:
        for attempt in range(MAX_GENERATION_ATTEMPTS):
            try:
                response = self.client.responses.parse(
                    model=self.model_version,
                    input=[{"role": "user", "content": request.prompt}],
                    text_format=OpenAIParsedResponse,
                    max_output_tokens=self.max_output_tokens * (attempt + 1),
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                )
                parsed = response.output_parsed
                if parsed is None:
                    raise RuntimeError(
                        "OpenAI returned no parsed structured output; inspect the "
                        "response status or refusal before retrying."
                    )
                return LLMGenerationResponse.model_validate(parsed.model_dump())
            except Exception:
                if attempt + 1 == MAX_GENERATION_ATTEMPTS:
                    raise

        raise AssertionError("The generation retry loop must return or raise.")
