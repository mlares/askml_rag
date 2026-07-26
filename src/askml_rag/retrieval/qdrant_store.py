from collections.abc import Sequence
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from askml_rag.models import Chunk, RetrievalFilters, Visibility


class Embedder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        normalize_embeddings: bool,
    ) -> np.ndarray: ...


def build_qdrant_filter(filters: RetrievalFilters) -> Filter:
    """Translate shared retrieval filters into Qdrant payload conditions."""
    conditions = [
        FieldCondition(
            key="visibility",
            match=MatchValue(value=Visibility.public.value),
        )
    ]

    if filters.document_types:
        conditions.append(
            FieldCondition(
                key="document_type",
                match=MatchAny(
                    any=[
                        document_type.value for document_type in filters.document_types
                    ]
                ),
            )
        )

    if filters.topics:
        conditions.append(
            FieldCondition(
                key="topics",
                match=MatchAny(any=filters.topics),
            )
        )

    if filters.year_from is not None or filters.year_to is not None:
        conditions.append(
            FieldCondition(
                key="year",
                range=Range(gte=filters.year_from, lte=filters.year_to),
            )
        )

    return Filter(must=conditions)


class QdrantSemanticRetriever:
    """Semantic retrieval backed by one Qdrant collection."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedder: Embedder,
        *,
        model_name: str,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedder = embedder
        self.model_name = model_name

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Embed chunks and replace the collection with their vectors."""
        chunks = list(chunks)

        if not chunks:
            raise ValueError("Cannot index an empty chunk collection.")

        embeddings = np.asarray(
            self.embedder.encode(
                [chunk.text for chunk in chunks],
                normalize_embeddings=True,
            )
        )

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embeddings.shape[1],
                distance=Distance.COSINE,
            ),
            metadata={
                "embedding_model": self.model_name,
                "vector_dimension": embeddings.shape[1],
            },
        )

        points = [
            PointStruct(
                id=index,
                vector=embedding.tolist(),
                payload=chunk.model_dump(mode="json"),
            )
            for index, (chunk, embedding) in enumerate(
                zip(chunks, embeddings, strict=True),
                start=1,
            )
        ]

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilters | None = None,
    ) -> list[Chunk]:
        """Return public chunks matching optional metadata filters."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")

        if not query.strip():
            return []

        active_filters = filters or RetrievalFilters()

        query_embedding = np.asarray(
            self.embedder.encode(
                [query],
                normalize_embeddings=True,
            )[0]
        )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding.tolist(),
            query_filter=build_qdrant_filter(active_filters),
            limit=limit,
            with_payload=True,
        )

        return [Chunk.model_validate(point.payload) for point in response.points]
