from askml_rag.models import CanonicalDocument, Chunk


def split_into_word_chunks(
    text: str,
    *,
    chunk_size_words: int,
    overlap_words: int,
) -> list[str]:
    """Split text into overlapping chunks using whitespace-separated words."""
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero.")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    words = text.split()

    if not words:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap_words

    return chunks


def chunk_document(
    document: CanonicalDocument,
    *,
    chunk_size_words: int,
    overlap_words: int,
) -> list[Chunk]:
    """Split one canonical document into citation-ready chunks."""
    chunk_texts = split_into_word_chunks(
        document.body_markdown,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    return [
        Chunk(
            chunk_id=f"{document.document_id}_chunk_{index:03d}",
            document_id=document.document_id,
            chunk_index=index,
            document_content_hash=document.content_hash,
            title=document.title,
            document_type=document.document_type,
            year=document.year,
            visibility=document.visibility,
            source_url=document.source_url,
            language=document.language,
            translation_of=document.translation_of,
            topics=document.topics,
            text=text,
        )
        for index, text in enumerate(chunk_texts)
    ]
