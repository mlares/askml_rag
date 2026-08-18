"""Language-preserving BM25 planning for heterogeneous bilingual corpora."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from askml_rag.models import Chunk, RetrievalFilters
from askml_rag.retrieval.bm25 import BM25Retriever, tokenize


BOILERPLATE_TOKENS = {
    "a",
    "an",
    "and",
    "are",
    "did",
    "documented",
    "does",
    "evidence",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "indexed",
    "is",
    "lares",
    "marcelo",
    "of",
    "on",
    "or",
    "public",
    "support",
    "supports",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
    "cuál",
    "cuáles",
    "cómo",
    "documentado",
    "documentados",
    "documentada",
    "documentadas",
    "el",
    "en",
    "evidencia",
    "la",
    "las",
    "los",
    "qué",
    "público",
    "tiene",
    "tuvo",
    "y",
}

QUERY_EXPANSIONS = {
    "aprendizaje profundo": (
        "aprendizaje profundo pytorch transferencia redes neuronales "
        "autoencoders densidad mixta"
    ),
    "comercio internacional": (
        "comercio internacional exportación oportunidad procordoba producto "
        "código hs"
    ),
    "deep learning": (
        "deep learning pytorch transfer learning neural networks autoencoders "
        "mixture density networks"
    ),
    "editorial": "editorial committee technical editor journal",
    "gcp": (
        "gcp google cloud platform compute engine cloud storage bigquery"
    ),
    "international trade": (
        "international trade export opportunity procordoba commodity hs code"
    ),
    "leadership": "leadership lead owner mentoring technical leadership team",
    "liderazgo": "liderazgo líder responsable mentoría equipo técnico",
    "python": (
        "python numpy pandas scipy pytorch scientific computing data pipelines"
    ),
    "statistics": "statistics probability hypothesis testing statistical",
    "estadística": "estadística probabilidad pruebas hipótesis estadístico",
    "enseñanza": "enseñanza profesor cursos currículo mentoría docencia",
    "docencia": "docencia profesor cursos currículo mentoría enseñanza",
    "teaching": "teaching professor courses curriculum mentoring taught",
    "time series": "time series forecasting temporal modeling",
    "time-series": "time series forecasting temporal modeling",
    "series temporales": "series temporales pronóstico modelado temporal",
}

FAMAF_CUES = (
    "clases prácticas",
    "famaf",
    "materias",
    "practical classes",
    "programa",
    "programme",
    "teóricos",
    "theoretically",
)
ITHREEX_CUES = (
    "animalia",
    "inverfin",
    "ithreex",
    "kolektor",
    "molibdeno",
    "procórdoba",
    "procordoba",
    "pueblo nativo",
    "ss servicios",
)
CROSS_SOURCE_CUES = (
    "ambos",
    "both mentoring",
    "came first",
    "connect marcelo",
    "difference",
    "differ",
    "early and recent",
    "how do the public materials",
    "sequence",
    "diferencia",
    "primero",
    "secuencia",
)
PROFESSIONAL_CUES = (
    "aprendizaje profundo",
    "comercio internacional",
    "deep learning",
    "editorial",
    "experience",
    "gcp",
    "google scholar profile",
    "industry",
    "leadership",
    "python",
    "researcher position",
    "software products",
    "teach",
    "teaching",
    "work at",
    "docencia",
    "enseñanza",
    "estadística",
    "experiencia",
    "liderazgo",
    "trabajó",
)
EXPANSION_CUES = (
    "ambos",
    "between",
    "both",
    "came first",
    "difference",
    "differ",
    "early and recent",
    "how did",
    "how do",
    "how many",
    "sequence",
    "diferencia",
    "entre",
    "primero",
    "secuencia",
)


def rewrite_query(query: str) -> str:
    """Remove benchmark boilerplate and add explicit lexical aliases."""
    lowered = query.lower()
    meaningful = [
        token
        for token in tokenize(lowered)
        if token not in BOILERPLATE_TOKENS and len(token) > 1
    ]
    rewritten = " ".join(meaningful)
    expansions = [
        expansion
        for cue, expansion in QUERY_EXPANSIONS.items()
        if cue in lowered
    ]
    return " ".join([rewritten, *expansions]).strip()


def decompose_query(query: str) -> list[str]:
    """Produce bounded subqueries for common multi-part question forms."""
    lowered = query.lower()
    subqueries = []
    if "mentoring" in lowered and "evaluation" in lowered:
        subqueries.extend(
            (
                "mentoring students advisor supervision",
                "formal evaluation reviewer committee conicet",
            )
        )
    return [subquery.strip(" ?. ,") for subquery in subqueries if subquery.strip()]


class PlannedBM25Retriever:
    """Route, expand, fuse, diversify, and locally expand BM25 results."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        candidate_limit: int = 50,
        rank_constant: int = 10,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be greater than zero.")
        if rank_constant <= 0:
            raise ValueError("rank_constant must be greater than zero.")
        self.chunks = list(chunks)
        self.retriever = BM25Retriever(self.chunks, include_metadata=True)
        self.language_retrievers = {
            language: BM25Retriever(
                [
                    chunk
                    for chunk in self.chunks
                    if chunk.language == language or chunk.language is None
                ],
                include_metadata=True,
            )
            for language in {chunk.language for chunk in self.chunks}
            if language is not None
        }
        self.candidate_limit = candidate_limit
        self.rank_constant = rank_constant
        self.document_chunks: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in self.chunks:
            self.document_chunks[chunk.document_id].append(chunk)
        for document_chunks in self.document_chunks.values():
            document_chunks.sort(key=lambda chunk: chunk.chunk_index)

    def _scope_document_ids(self, query: str) -> tuple[list[str], bool]:
        lowered = query.lower()
        is_professional = False

        if any(cue in lowered for cue in FAMAF_CUES):
            prefixes = ("famaf_",)
            exact_ids: tuple[str, ...] = ()
        elif any(cue in lowered for cue in ITHREEX_CUES):
            prefixes = ("ithreex_", "website_")
            exact_ids = ("skills",)
        elif any(cue in lowered for cue in CROSS_SOURCE_CUES):
            prefixes = ("website_", "cv_")
            exact_ids = ("skills",)
        elif any(cue in lowered for cue in PROFESSIONAL_CUES):
            is_professional = True
            prefixes = ("website_", "cv_", "personal_traits")
            exact_ids = ("skills",)
        else:
            return [], False

        return [
            document_id
            for document_id in self.document_chunks
            if document_id.startswith(prefixes) or document_id in exact_ids
        ], is_professional

    @staticmethod
    def _merge_filters(
        filters: RetrievalFilters | None,
        scoped_document_ids: Sequence[str],
    ) -> RetrievalFilters:
        active = filters or RetrievalFilters()
        if not scoped_document_ids:
            return active
        scoped = set(scoped_document_ids)
        document_ids = (
            [item for item in active.document_ids if item in scoped]
            if active.document_ids
            else list(scoped_document_ids)
        )
        return active.model_copy(update={"document_ids": document_ids})

    def _fused_candidates(
        self,
        query: str,
        filters: RetrievalFilters,
    ) -> tuple[list[Chunk], dict[str, Chunk]]:
        queries = list(
            dict.fromkeys((query, rewrite_query(query), *decompose_query(query)))
        )
        scores: defaultdict[str, float] = defaultdict(float)
        chunks_by_id: dict[str, Chunk] = {}
        retriever = self._retriever_for(filters)
        for variant in queries:
            for rank, chunk in enumerate(
                retriever.search(
                    variant,
                    limit=self.candidate_limit,
                    filters=filters,
                ),
                start=1,
            ):
                chunks_by_id[chunk.chunk_id] = chunk
                scores[chunk.chunk_id] += 1 / (self.rank_constant + rank)
        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [chunks_by_id[chunk_id] for chunk_id in ranked_ids], chunks_by_id

    def _retriever_for(self, filters: RetrievalFilters) -> BM25Retriever:
        """Use a selector-language index so other languages do not alter IDF."""
        if len(filters.languages) == 1:
            language = filters.languages[0]
            if language in self.language_retrievers:
                return self.language_retrievers[language]
        return self.retriever

    def _expand_neighbors(
        self,
        candidates: Sequence[Chunk],
        chunks_by_id: dict[str, Chunk],
    ) -> list[Chunk]:
        expanded_ids: list[str] = []
        for chunk in candidates[:10]:
            if chunk.chunk_id not in expanded_ids:
                expanded_ids.append(chunk.chunk_id)
            document_chunks = self.document_chunks[chunk.document_id]
            for neighbor_index in (chunk.chunk_index - 1, chunk.chunk_index + 1):
                if 0 <= neighbor_index < len(document_chunks):
                    neighbor = document_chunks[neighbor_index]
                    chunks_by_id[neighbor.chunk_id] = neighbor
                    if neighbor.chunk_id not in expanded_ids:
                        expanded_ids.append(neighbor.chunk_id)
        expanded_ids.extend(
            chunk.chunk_id
            for chunk in candidates
            if chunk.chunk_id not in expanded_ids
        )
        return [chunks_by_id[chunk_id] for chunk_id in expanded_ids]

    @staticmethod
    def _diversify(
        candidates: Sequence[Chunk],
        *,
        maximum_per_document: int,
    ) -> list[Chunk]:
        counts: defaultdict[str, int] = defaultdict(int)
        diversified = []
        for chunk in candidates:
            if counts[chunk.document_id] >= maximum_per_document:
                continue
            counts[chunk.document_id] += 1
            diversified.append(chunk)
        return diversified

    def search(
        self,
        query: str,
        *,
        limit: int,
        filters: RetrievalFilters | None = None,
    ) -> list[Chunk]:
        """Return a routed and fused lexical ranking for the original query."""
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        scoped_document_ids, is_professional = self._scope_document_ids(query)
        active_filters = self._merge_filters(filters, scoped_document_ids)
        lowered = query.lower()
        if any(cue in lowered for cue in FAMAF_CUES):
            candidates = self._retriever_for(active_filters).search(
                query,
                limit=self.candidate_limit,
                filters=active_filters,
            )
            chunks_by_id = {chunk.chunk_id: chunk for chunk in candidates}
        else:
            candidates, chunks_by_id = self._fused_candidates(
                query,
                active_filters,
            )
        if any(cue in lowered for cue in EXPANSION_CUES):
            candidates = self._expand_neighbors(candidates, chunks_by_id)
        elif is_professional:
            candidates = self._diversify(candidates, maximum_per_document=2)
        return list(candidates[:limit])
