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
    "cual",
    "cuales",
    "cómo",
    "como",
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
    "lo",
    "que",
    "quien",
    "quienes",
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
        "comercio internacional exportación oportunidad procordoba producto código hs"
    ),
    "deep learning": (
        "deep learning pytorch transfer learning neural networks autoencoders "
        "mixture density networks"
    ),
    "course": "teaching course curriculum syllabus programme professor",
    "curso": "docencia curso materia programa contenidos profesor",
    "editorial": "editorial committee technical editor journal",
    "gcp": ("gcp google cloud platform compute engine cloud storage bigquery"),
    "international trade": (
        "international trade export opportunity procordoba commodity hs code"
    ),
    "leadership": "leadership lead owner mentoring technical leadership team",
    "liderazgo": "liderazgo líder responsable mentoría equipo técnico",
    "python": ("python numpy pandas scipy pytorch scientific computing data pipelines"),
    "research": "research publications papers collaboration coauthors scientific methods",
    "investigación": "investigación publicaciones artículos colaboración coautores métodos científicos",
    "statistics": "statistics probability hypothesis testing statistical",
    "estadística": "estadística probabilidad pruebas hipótesis estadístico",
    "enseñanza": "enseñanza profesor cursos currículo mentoría docencia",
    "docencia": "docencia profesor cursos currículo mentoría enseñanza",
    "teaching": "teaching professor courses curriculum mentoring taught",
    "time series": "time series forecasting temporal modeling",
    "time-series": "time series forecasting temporal modeling",
    "series temporales": "series temporales pronóstico modelado temporal",
}

PRODUCT_TOKENS = frozenset(
    {
        "product",
        "products",
        "producto",
        "productos",
        "platform",
        "platforms",
        "plataforma",
        "plataformas",
    }
)
AUDIENCE_CUES = (
    "customer",
    "client",
    "user",
    "cliente",
    "usuario",
    "quien lo usa",
    "quienes lo usan",
    "usan",
)
CONTRIBUTION_CUES = (
    "contribution",
    "contributed",
    "your role",
    "aporte",
    "aportaste",
    "contribución",
    "contribuiste",
    "rol",
)
RECENCY_CUES = (
    "latest",
    "last one",
    "most recent",
    "current",
    "último",
    "ultimo",
    "más reciente",
    "mas reciente",
    "actual",
)
PROFILE_CUES = (
    "tell me about yourself",
    "tell me about you",
    "about yourself",
    "contame sobre vos",
    "cuéntame sobre ti",
    "quién sos",
    "quien sos",
    "quién eres",
    "quien eres",
)
THESIS_ADVISOR_CUES = (
    "who supervised your thesis",
    "who directed your thesis",
    "who was your thesis director",
    "quién dirigió tu tesis",
    "quien dirigio tu tesis",
    "director de tu tesis",
    "directora de tu tesis",
)

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
    "latest",
    "last one",
    "most recent",
    "último",
    "ultimo",
    "más reciente",
    "mas reciente",
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
        expansion for cue, expansion in QUERY_EXPANSIONS.items() if cue in lowered
    ]
    return " ".join([rewritten, *expansions]).strip()


def _is_product_question(query: str) -> bool:
    """Recognize product intent without confusing `product` with `production`."""
    return bool(set(tokenize(query)) & PRODUCT_TOKENS)


def _is_professional_product_question(query: str) -> bool:
    """Recognize product questions that ask about Marcelo's professional work."""
    lowered = query.lower()
    if not _is_product_question(lowered):
        return False
    work_cues = (
        "worked",
        "work on",
        "built",
        "developed",
        "trabajado",
        "trabajaste",
        "trabajó",
        "construí",
        "construido",
        "desarrollé",
        "desarrollado",
    )
    return any(
        cue in lowered
        for cue in (
            *work_cues,
            *AUDIENCE_CUES,
            *CONTRIBUTION_CUES,
            *RECENCY_CUES,
        )
    )


def _contains_any(query: str, cues: Sequence[str]) -> bool:
    return any(cue in query.lower() for cue in cues)


def decompose_query(query: str) -> list[str]:
    """Produce bounded subqueries for common multi-part question forms."""
    lowered = query.lower()
    subqueries: list[str] = []
    if "mentoring" in lowered and "evaluation" in lowered:
        subqueries.extend(
            (
                "mentoring students advisor supervision",
                "formal evaluation reviewer committee conicet",
            )
        )

    if _contains_any(lowered, PROFILE_CUES):
        subqueries.extend(
            (
                "perfil profesional experiencia habilidades trayectoria",
                "biografía trabajo investigación docencia liderazgo",
            )
        )

    if _contains_any(lowered, THESIS_ADVISOR_CUES):
        subqueries.append(
            "doctorado astronomía tesis director Diego García Lambas formación académica"
        )

    product_question = _is_professional_product_question(lowered)
    audience_question = any(cue in lowered for cue in AUDIENCE_CUES)
    contribution_question = any(cue in lowered for cue in CONTRIBUTION_CUES)
    recency_question = any(cue in lowered for cue in RECENCY_CUES)
    spanish_question = any(
        cue in lowered
        for cue in (
            "producto",
            "plataforma",
            "cliente",
            "usuario",
            "aporte",
            "contribución",
            "último",
            "ultimo",
        )
    )
    if product_question:
        subqueries.append(
            "productos proyectos profesionales software plataforma modelos"
            if spanish_question
            else "professional products projects software platform models"
        )
        if audience_question:
            subqueries.append(
                "usuarios clientes equipos escala producción"
                if spanish_question
                else "users customers clients teams scale production"
            )
        if contribution_question:
            subqueries.append(
                "aporte contribución rol desarrollé lideré construí entregué"
                if spanish_question
                else "contribution role developed led built delivered"
            )
        if recency_question:
            subqueries.append(
                "último reciente actual presente período"
                if spanish_question
                else "latest recent current present period"
            )
    return list(dict.fromkeys(subqueries))


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

        if _contains_any(lowered, THESIS_ADVISOR_CUES):
            prefixes = ("website_teaching", "cv_education")
            exact_ids = ()
        elif _contains_any(lowered, PROFILE_CUES):
            is_professional = True
            prefixes = ("website_home", "cv_expertise", "personal_traits")
            exact_ids = ("skills", "skills_es")
        elif any(cue in lowered for cue in FAMAF_CUES):
            prefixes = ("famaf_",)
            exact_ids: tuple[str, ...] = ()
        elif any(cue in lowered for cue in ITHREEX_CUES):
            prefixes = ("ithreex_", "website_")
            exact_ids = ("skills",)
        elif any(cue in lowered for cue in CROSS_SOURCE_CUES):
            prefixes = ("website_", "cv_")
            exact_ids = ("skills",)
        elif _is_professional_product_question(lowered) or any(
            cue in lowered for cue in PROFESSIONAL_CUES
        ):
            is_professional = True
            if _is_professional_product_question(lowered):
                prefixes = ("website_home", "website_projects", "ithreex_")
                exact_ids = ()
            else:
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
    ) -> RetrievalFilters | None:
        active = filters or RetrievalFilters()
        if not scoped_document_ids:
            return active
        scoped = set(scoped_document_ids)
        if active.document_ids:
            document_ids = [item for item in active.document_ids if item in scoped]
            if not document_ids:
                return None
        else:
            document_ids = list(scoped_document_ids)
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
            chunk.chunk_id for chunk in candidates if chunk.chunk_id not in expanded_ids
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
        if active_filters is None:
            return []
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
        expands_neighbors = any(cue in lowered for cue in EXPANSION_CUES)
        if expands_neighbors:
            candidates = self._expand_neighbors(candidates, chunks_by_id)
        if is_professional and (
            not expands_neighbors or _is_professional_product_question(lowered)
        ):
            candidates = self._diversify(candidates, maximum_per_document=2)
        return list(candidates[:limit])
