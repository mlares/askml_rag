import re
from collections.abc import Sequence
from rank_bm25 import BM25Okapi
from askml_rag.models import Chunk, RetrievalFilters
from askml_rag.retrieval.filtering import matches_filters


TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase lexical-search tokens."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Lexical retriever over a fixed collection of chunks."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)

        if self.chunks:
            tokenized_chunks = [tokenize(chunk.text) for chunk in self.chunks]
            self.index: BM25Okapi | None = BM25Okapi(tokenized_chunks)
        else:
            self.index = None

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilters | None = None,
    ) -> list[Chunk]:
        """Return public lexical matches satisfying optional filters."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        query_tokens = tokenize(query)

        if not query_tokens or self.index is None:
            return []

        active_filters = filters or RetrievalFilters()
        scores = self.index.get_scores(query_tokens)
        ranked_results = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            self.chunks[index]
            for index, score in ranked_results
            if score > 0 and matches_filters(self.chunks[index], active_filters)
        ][:limit]
