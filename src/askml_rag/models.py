from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class DocumentType(StrEnum):
    publication = "publication"
    project = "project"
    professional_experience = "professional_experience"
    teaching = "teaching"
    biography = "biography"
    website = "website"


class Visibility(StrEnum):
    public = "public"
    excluded = "excluded"


class AbstractSource(StrEnum):
    """Origin of publication abstract text."""

    bibtex = "bibtex"
    pdf = "pdf"


class SourceManifest(BaseModel):
    """Human-curated metadata for one raw source document."""

    document_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    title: str = Field(min_length=1)
    document_type: DocumentType
    year: int | None = Field(default=None, ge=1900)
    authors: list[str] = Field(default_factory=list)
    source_url: HttpUrl | None = None
    source_path: str
    visibility: Visibility = Visibility.public
    topics: list[str] = Field(default_factory=list)


class CanonicalDocument(BaseModel):
    """A normalized, public document ready for chunking and retrieval."""

    document_id: str = Field(
        pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
        examples=["paper_deepz_2022"],
    )
    title: str = Field(min_length=1)
    document_type: DocumentType
    year: int | None = Field(default=None, ge=1900)
    authors: list[str] = Field(default_factory=list)
    source_url: HttpUrl | None = None
    source_path: str
    content_hash: str = Field(min_length=64, max_length=64)
    visibility: Visibility = Visibility.public
    topics: list[str] = Field(default_factory=list)
    body_markdown: str = Field(min_length=1)


class PublicationMetadata(BaseModel):
    """Normalized metadata for one publication included in the corpus."""

    document_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    bibtex_key: str = Field(min_length=1)
    entry_type: str = Field(min_length=1)
    title: str = Field(min_length=1)
    authors: list[str] = Field(min_length=1)
    year: int = Field(ge=1900)
    journal: str | None = None
    journal_abbreviation: str | None = None
    abstract: str | None = None
    abstract_source: AbstractSource | None = None
    keywords: list[str] = Field(default_factory=list)
    doi: str | None = None
    arxiv_id: str = Field(min_length=1)
    primary_class: str | None = None
    volume: str | None = None
    number: str | None = None
    pages: str | None = None
    source_url: HttpUrl | None = None
    ads_url: HttpUrl | None = None


class Chunk(BaseModel):
    """A retrievable text fragment linked to one canonical document."""

    chunk_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    document_id: str = Field(pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
    chunk_index: int = Field(ge=0)
    document_content_hash: str = Field(min_length=64, max_length=64)
    title: str = Field(min_length=1)
    document_type: DocumentType
    year: int | None = Field(default=None, ge=1900)
    visibility: Visibility = Visibility.public
    source_url: HttpUrl | None = None
    topics: list[str] = Field(default_factory=list)
    text: str = Field(min_length=1)


class RetrievalFilters(BaseModel):
    """Optional constraints applied consistently during retrieval."""

    document_ids: list[str] = Field(default_factory=list)
    document_types: list[DocumentType] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    year_from: int | None = Field(default=None, ge=1900)
    year_to: int | None = Field(default=None, ge=1900)

    @model_validator(mode="after")
    def validate_year_range(self) -> "RetrievalFilters":
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise ValueError("year_from cannot be greater than year_to.")

        return self


class QuestionCategory(StrEnum):
    direct_fact = "direct_fact"
    multi_document = "multi_document"
    comparison = "comparison"
    quantitative = "quantitative"
    temporal = "temporal"
    ambiguous = "ambiguous"
    unanswerable = "unanswerable"
    false_premise = "false_premise"


class ExpectedClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(min_length=1)
    source_document_ids: list[str] = Field(min_length=1)


class EvaluationQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(
        pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
        examples=["ml_experience_001"],
    )
    question: str = Field(min_length=1)
    answerable: bool
    category: QuestionCategory
    expected_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    expected_claims: list[ExpectedClaim] = Field(default_factory=list)
    difficulty: str = Field(min_length=1)
    notes: str | None = None
