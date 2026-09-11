from askml_rag.models import Chunk, RetrievalFilters
from askml_rag.retrieval.planned_bm25 import (
    PlannedBM25Retriever,
    decompose_query,
    rewrite_query,
)


def make_chunk(
    chunk_id: str,
    document_id: str,
    chunk_index: int,
    text: str,
    *,
    language: str = "en",
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=chunk_index,
        document_content_hash="a" * 64,
        title=document_id,
        document_type="website",
        language=language,
        topics=[],
        text=text,
    )


def test_rewrite_query_removes_boilerplate_and_expands_gcp() -> None:
    rewritten = rewrite_query(
        "Which GCP services are documented in Marcelo's indexed public corpus?"
    )

    assert "marcelo" not in rewritten
    assert "documented" not in rewritten
    assert "google cloud platform" in rewritten
    assert "compute engine" in rewritten
    assert "bigquery" in rewritten


def test_rewrite_query_expands_course_terms_for_teaching_material() -> None:
    rewritten = rewrite_query("What does the machine learning course cover?")

    assert "curriculum" in rewritten
    assert "syllabus" in rewritten


def test_rewrite_query_expands_research_terms_without_profile_only_routing() -> None:
    rewritten = rewrite_query("What research has Marcelo done?")

    assert "publications" in rewritten
    assert "coauthors" in rewritten


def test_decompose_query_separates_multi_part_evidence_needs() -> None:
    assert decompose_query(
        "What evidence documents both mentoring and formal evaluation work?"
    ) == [
        "mentoring students advisor supervision",
        "formal evaluation reviewer committee conicet",
    ]


def test_decompose_spanish_product_question_into_evidence_facets() -> None:
    subqueries = decompose_query(
        "en que productos has trabajado, quienes lo usan y que aportaste en el ultimo"
    )

    assert subqueries == [
        "productos proyectos profesionales software plataforma modelos",
        "usuarios clientes equipos escala producción",
        "aporte contribución rol desarrollé lideré construí entregué",
        "último reciente actual presente período",
    ]


def test_decompose_profile_and_thesis_advisor_questions() -> None:
    assert decompose_query("Contame sobre vos.") == [
        "perfil profesional experiencia habilidades trayectoria",
        "biografía trabajo investigación docencia liderazgo",
    ]
    assert decompose_query("¿Quién dirigió tu tesis?") == [
        "doctorado astronomía tesis director Diego García Lambas formación académica"
    ]


def test_professional_routing_excludes_famaf_candidates() -> None:
    chunks = [
        make_chunk(
            "website_teaching_chunk_000",
            "website_teaching",
            0,
            "university teaching statistics data science",
        ),
        make_chunk(
            "famaf_teaching_activity_chunk_000",
            "famaf_teaching_activity",
            0,
            "university teaching statistics data science teaching",
        ),
        make_chunk("noise_one_chunk_000", "noise_one", 0, "astronomy"),
        make_chunk("noise_two_chunk_000", "noise_two", 0, "software"),
        make_chunk("noise_three_chunk_000", "noise_three", 0, "galaxies"),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "What teaching experience does Marcelo have in statistics?",
        limit=3,
        filters=RetrievalFilters(languages=["en"]),
    )

    assert [chunk.chunk_id for chunk in results] == ["website_teaching_chunk_000"]


def test_comparison_queries_expand_adjacent_chunks() -> None:
    chunks = [
        make_chunk(
            "website_projects_chunk_000",
            "website_projects",
            0,
            "cosmic void detection methodology",
        ),
        make_chunk(
            "website_projects_chunk_001",
            "website_projects",
            1,
            "galaxy spin alignment inference",
        ),
        make_chunk("noise_one_chunk_000", "noise_one", 0, "astronomy"),
        make_chunk("noise_two_chunk_000", "noise_two", 0, "software"),
        make_chunk("noise_three_chunk_000", "noise_three", 0, "statistics"),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "What is the difference between cosmic void detection and galaxy spin?",
        limit=2,
    )

    assert {chunk.chunk_id for chunk in results} == {
        "website_projects_chunk_000",
        "website_projects_chunk_001",
    }


def test_professional_results_are_diversified_by_document() -> None:
    chunks = [
        *[
            make_chunk(
                f"website_home_chunk_00{index}",
                "website_home",
                index,
                f"python experience software {index}",
            )
            for index in range(3)
        ],
        make_chunk(
            "skills_chunk_000",
            "skills",
            0,
            "python scientific computing",
        ),
        make_chunk("noise_one_chunk_000", "noise_one", 0, "astronomy"),
        make_chunk("noise_two_chunk_000", "noise_two", 0, "galaxies"),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search("What Python experience does Marcelo have?", limit=4)

    assert sum(chunk.document_id == "website_home" for chunk in results) <= 2
    assert any(chunk.document_id == "skills" for chunk in results)


def test_selected_language_uses_a_separate_lexical_index() -> None:
    chunks = [
        make_chunk(
            "website_home_chunk_000",
            "website_home",
            0,
            "python scientific computing",
        ),
        make_chunk(
            "website_home_es_chunk_000",
            "website_home_es",
            0,
            "python computación científica python",
            language="es",
        ),
        make_chunk("noise_one_chunk_000", "noise_one", 0, "astronomy"),
        make_chunk("noise_two_chunk_000", "noise_two", 0, "galaxies"),
        make_chunk("noise_three_chunk_000", "noise_three", 0, "statistics"),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "Python experience",
        limit=2,
        filters=RetrievalFilters(languages=["en"]),
    )

    assert [chunk.language for chunk in results] == ["en"]


def test_spanish_product_question_routes_and_covers_required_evidence() -> None:
    chunks = [
        make_chunk(
            "website_projects_es_chunk_000",
            "website_projects_es",
            0,
            "Sistema de recomendaciones en producción usado por millones de usuarios.",
            language="es",
        ),
        make_chunk(
            "website_projects_es_chunk_001",
            "website_projects_es",
            1,
            "Mi contribución en el proyecto actual fue desarrollar y optimizar el modelo.",
            language="es",
        ),
        make_chunk(
            "ithreex_molibdeno_es_chunk_000",
            "ithreex_molibdeno_es",
            0,
            "Molibdeno es una plataforma de modelos utilizada por clientes de IThreex.",
            language="es",
        ),
        make_chunk(
            "famaf_productos_chunk_000",
            "famaf_productos",
            0,
            "Materia universitaria sobre productos y usuarios.",
            language="es",
        ),
        make_chunk(
            "arxiv_productos_chunk_000",
            "arxiv_productos",
            0,
            "Artículo sobre productos, usuarios y contribuciones recientes.",
            language="es",
        ),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "en que productos has trabajado, quienes lo usan y que aportaste en el ultimo",
        limit=4,
        filters=RetrievalFilters(languages=["es"]),
    )
    result_ids = {chunk.chunk_id for chunk in results}

    assert "website_projects_es_chunk_000" in result_ids
    assert "website_projects_es_chunk_001" in result_ids
    assert "ithreex_molibdeno_es_chunk_000" in result_ids
    assert "famaf_productos_chunk_000" not in result_ids
    assert "arxiv_productos_chunk_000" not in result_ids


def test_disjoint_caller_and_planner_scopes_return_no_results() -> None:
    chunks = [
        make_chunk(
            "website_projects_es_chunk_000",
            "website_projects_es",
            0,
            "Productos profesionales de software.",
            language="es",
        ),
        make_chunk(
            "famaf_productos_chunk_000",
            "famaf_productos",
            0,
            "Productos profesionales de software.",
            language="es",
        ),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "¿En qué productos has trabajado?",
        limit=2,
        filters=RetrievalFilters(
            document_ids=["famaf_productos"],
            languages=["es"],
        ),
    )

    assert results == []


def test_profile_question_routes_to_profile_sources() -> None:
    chunks = [
        make_chunk(
            "website_home_es_chunk_000",
            "website_home_es",
            0,
            "Perfil profesional con experiencia en industria, investigación y docencia.",
            language="es",
        ),
        make_chunk(
            "personal_traits_chunk_000",
            "personal_traits",
            0,
            "Biografía, habilidades y estilo de liderazgo.",
            language="es",
        ),
        make_chunk(
            "arxiv_biografia_chunk_000",
            "arxiv_biografia",
            0,
            "Un artículo que menciona una biografía profesional.",
            language="es",
        ),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "Contame sobre vos.",
        limit=3,
        filters=RetrievalFilters(languages=["es"]),
    )

    assert {chunk.document_id for chunk in results} == {
        "website_home_es",
        "personal_traits",
    }


def test_thesis_advisor_question_routes_to_education_sources() -> None:
    chunks = [
        make_chunk(
            "website_teaching_es_chunk_000",
            "website_teaching_es",
            0,
            "Doctorado en Astronomía. Tesis dirigida por Diego García Lambas.",
            language="es",
        ),
        make_chunk(
            "famaf_teaching_activity_es_chunk_000",
            "famaf_teaching_activity_es",
            0,
            "Marcelo dirigió tesis doctorales de otros investigadores.",
            language="es",
        ),
        make_chunk(
            "arxiv_tesis_chunk_000",
            "arxiv_tesis",
            0,
            "Una referencia bibliográfica a una tesis doctoral.",
            language="es",
        ),
    ]
    retriever = PlannedBM25Retriever(chunks)

    results = retriever.search(
        "¿Quién dirigió tu tesis?",
        limit=3,
        filters=RetrievalFilters(languages=["es"]),
    )

    assert [chunk.document_id for chunk in results] == ["website_teaching_es"]
