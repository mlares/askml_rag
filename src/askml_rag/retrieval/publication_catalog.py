"""Retrieval over structured publication metadata.

This is intentionally separate from full-text retrieval.  A question such as
"which papers did A publish with B?" is a set-membership query over author
lists, not a relevance-ranking query over PDF passages.
"""

import re
import unicodedata
from collections.abc import Sequence

from askml_rag.ingestion.publications import publication_summary_chunk
from askml_rag.models import Chunk, PublicationMetadata, RetrievalFilters
from askml_rag.retrieval.bm25 import BM25Retriever


def normalize_author_name(name: str) -> tuple[str, ...]:
    """Normalize accents, punctuation, and whitespace in a person name."""
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return tuple(re.findall(r"[a-z0-9]+", without_accents.casefold()))


def author_name_matches(author: str, query: str) -> bool:
    """Match a query name against one author, allowing initials for given names.

    For example, ``Marcelo Lares`` matches ``M. Lares`` and
    ``Daza-Perilla`` matches ``I. V. Daza-Perilla``.  All query tokens must be
    represented, so a surname-only match remains deliberate and inspectable.
    """
    author_tokens = normalize_author_name(author)
    query_tokens = normalize_author_name(query)
    if not query_tokens:
        raise ValueError("An author query must contain letters or numbers.")

    # An initial alone is not enough evidence for a name match. In particular,
    # searching "Gramajo" must not match every author written as "G. ...".
    # Requiring one exact non-initial token still lets "Marcelo Lares" match
    # "M. Lares", because the surname is exact.
    if not any(
        len(query_token) > 1 and query_token in author_tokens
        for query_token in query_tokens
    ):
        return False

    return all(
        any(
            author_token == query_token
            or (len(author_token) == 1 and author_token == query_token[0])
            or (len(query_token) == 1 and query_token == author_token[0])
            for author_token in author_tokens
        )
        for query_token in query_tokens
    )


class PublicationCatalogRetriever:
    """Search publication summaries and deterministically filter coauthor lists."""

    def __init__(self, publications: Sequence[PublicationMetadata]) -> None:
        self.publications = list(publications)
        self.publications_by_id = {
            publication.document_id: publication for publication in self.publications
        }
        if len(self.publications_by_id) != len(self.publications):
            raise ValueError("Publication document IDs must be unique.")

        self.summary_chunks_by_id = {
            publication.document_id: publication_summary_chunk(publication)
            for publication in self.publications
        }
        self.summary_retriever = BM25Retriever(
            list(self.summary_chunks_by_id.values())
        )

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilters | None = None,
    ) -> list[Chunk]:
        """Rank title, author, keyword, and abstract summary chunks with BM25."""
        return self.summary_retriever.search(query, limit=limit, filters=filters)

    def publications_by_authors(
        self,
        author_queries: Sequence[str],
        *,
        limit: int | None = None,
    ) -> list[PublicationMetadata]:
        """Return every paper containing all requested author identities.

        This is a logical AND over author lists, rather than a ranked text
        search.  The default has no result cap because list questions need an
        exhaustive answer relative to the bounded catalogue.
        """
        if not author_queries:
            raise ValueError("At least one author query is required.")
        if limit is not None and limit <= 0:
            raise ValueError("limit must be greater than zero when provided.")

        normalized_queries = tuple(author_queries)
        for query in normalized_queries:
            if not normalize_author_name(query):
                raise ValueError("An author query must contain letters or numbers.")
        matches = [
            publication
            for publication in self.publications
            if all(
                any(author_name_matches(author, query) for author in publication.authors)
                for query in normalized_queries
            )
        ]
        ranked = sorted(
            matches,
            key=lambda publication: (-publication.year, publication.title.casefold()),
        )
        return ranked if limit is None else ranked[:limit]

    def chunks_by_authors(
        self,
        author_queries: Sequence[str],
        *,
        limit: int | None = None,
    ) -> list[Chunk]:
        """Return summary evidence for every exact author-list match."""
        return [
            self.summary_chunks_by_id[publication.document_id]
            for publication in self.publications_by_authors(author_queries, limit=limit)
        ]
