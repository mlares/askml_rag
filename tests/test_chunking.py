from askml_rag.retrieval.chunking import split_into_word_chunks
from askml_rag.models import CanonicalDocument
from askml_rag.retrieval.chunking import chunk_document


def test_split_into_word_chunks_uses_overlap() -> None:
    text = "one two three four five six seven eight nine ten"

    chunks = split_into_word_chunks(
        text,
        chunk_size_words=4,
        overlap_words=1,
    )

    assert chunks == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]


def test_chunk_document_preserves_citation_metadata() -> None:
    document = CanonicalDocument(
        document_id="example_source",
        title="Example source",
        document_type="website",
        source_url="https://www.mlares.space/",
        source_path="data/raw/example.md",
        content_hash="a" * 64,
        topics=["example"],
        body_markdown="one two three four five six seven eight nine ten",
    )

    chunks = chunk_document(
        document,
        chunk_size_words=4,
        overlap_words=1,
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "example_source_chunk_000",
        "example_source_chunk_001",
        "example_source_chunk_002",
    ]
    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]
    assert all(chunk.document_id == "example_source" for chunk in chunks)
    assert all(chunk.title == "Example source" for chunk in chunks)
    assert all(chunk.document_content_hash == "a" * 64 for chunk in chunks)

    assert all(chunk.year == document.year for chunk in chunks)
    assert all(chunk.visibility == document.visibility for chunk in chunks)
