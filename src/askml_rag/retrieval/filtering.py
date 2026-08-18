from askml_rag.models import Chunk, RetrievalFilters, Visibility


def matches_filters(chunk: Chunk, filters: RetrievalFilters) -> bool:
    """Return whether a public chunk satisfies all retrieval filters."""
    if chunk.visibility != Visibility.public:
        return False

    if filters.document_ids:
        if chunk.document_id not in filters.document_ids:
            return False

    if filters.document_types:
        if chunk.document_type not in filters.document_types:
            return False

    if filters.languages:
        language_matches = chunk.language in filters.languages
        unknown_language_matches = (
            filters.include_unknown_language and chunk.language is None
        )
        if not language_matches and not unknown_language_matches:
            return False

    if filters.topics:
        if not set(chunk.topics) & set(filters.topics):
            return False

    if filters.year_from is not None:
        if chunk.year is None or chunk.year < filters.year_from:
            return False

    if filters.year_to is not None:
        if chunk.year is None or chunk.year > filters.year_to:
            return False

    return True
