from collections import defaultdict
from typing import Protocol

from askml_rag.models import Chunk, RetrievalFilters


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilters | None = None,
    ) -> list[Chunk]: ...


class HybridRetriever:
    """Fuse lexical and semantic retrieval rankings with RRF."""

    def __init__(
        self,
        lexical_retriever: Retriever,
        semantic_retriever: Retriever,
        *,
        rank_constant: int = 60,
        candidate_limit: int = 10,
    ) -> None:
        if rank_constant <= 0:
            raise ValueError("rank_constant must be greater than zero.")
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than zero.")

        self.lexical_retriever = lexical_retriever
        self.semantic_retriever = semantic_retriever
        self.rank_constant = rank_constant
        self.candidate_limit = candidate_limit

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilters | None = None,
    ) -> list[Chunk]:
        """Return a fused ranking from lexical and semantic retrieval."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        lexical_results = self.lexical_retriever.search(
            query,
            limit=self.candidate_limit,
            filters=filters,
        )
        semantic_results = self.semantic_retriever.search(
            query,
            limit=self.candidate_limit,
            filters=filters,
        )

        scores: defaultdict[str, float] = defaultdict(float)
        chunks_by_id: dict[str, Chunk] = {}

        for results in (lexical_results, semantic_results):
            for rank, chunk in enumerate(results, start=1):
                chunks_by_id[chunk.chunk_id] = chunk
                scores[chunk.chunk_id] += 1 / (self.rank_constant + rank)

        ranked_chunk_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

        return [chunks_by_id[chunk_id] for chunk_id in ranked_chunk_ids[:limit]]
