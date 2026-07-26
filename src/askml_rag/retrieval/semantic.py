from collections.abc import Sequence
from typing import Protocol

import numpy as np

from askml_rag.models import Chunk, RetrievalFilters
from askml_rag.retrieval.filtering import matches_filters


class Embedder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
    ) -> np.ndarray: ...


class SemanticRetriever:
    """Vector-similarity retriever over a fixed collection of chunks."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder

        if self.chunks:
            self.chunk_embeddings = np.asarray(
                self.embedder.encode(
                    [chunk.text for chunk in self.chunks],
                    normalize_embeddings=True,
                )
            )
        else:
            self.chunk_embeddings = np.empty((0, 0))

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilters | None = None,
    ) -> list[Chunk]:
        """Return public semantic matches satisfying optional filters."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        if not query.strip() or not self.chunks:
            return []

        active_filters = filters or RetrievalFilters()

        query_embedding = np.asarray(
            self.embedder.encode(
                [query],
                normalize_embeddings=True,
            )[0]
        )

        scores = self.chunk_embeddings @ query_embedding
        ranked_indices = np.argsort(scores)[::-1]

        return [
            self.chunks[index]
            for index in ranked_indices
            if matches_filters(self.chunks[index], active_filters)
        ][:limit]
