# Reproducir el corpus y la evaluación de retrieval

Esta guía reconstruye el corpus local y ejecuta la configuración elegida:
BM25 sobre `full_text` con `k=7`. Ejecuta los comandos desde la raíz del
repositorio (`askml_rag/`).

## Antes de empezar

```bash
uv sync
```

Los archivos de `data/raw/` y `sources/public/` no se versionan porque forman
parte del corpus local. Deben existir en las rutas `source_path` declaradas por
los manifests. Los manifests sí se versionan y son la lista de fuentes
autorizadas; no sustituyas una fuente ausente por otra sin actualizar y revisar
su manifest.

## Release baseline versionado

`data/evaluation/release.yaml` fija el corpus, el tamaño de chunk, los tres
archivos de preguntas oficiales y la configuración de retrieval para una
release. El identificador actual es `2026.08.0`; no cambies sus entradas sin
crear una nueva versión y regenerar el índice de identidad.

Los archivos `questions_es.draft.yaml` y los índices o conjuntos de preguntas
IThreex separados no forman parte de la release: el primero aún necesita
revisión humana y los segundos ya están incluidos en el corpus canónico y en
`questions.yaml`. No combines los directorios de chunks suplementarios con el
corpus canónico, porque repiten IDs.

Al crear el commit de una release, incluye los manifests revisados, los tres
conjuntos de preguntas oficiales, `corpus_index.json`, `release.yaml`, el
validador y sus pruebas. No añadas `data/raw/`, `data/processed/`, informes de
ejecución, el borrador español ni los índices suplementarios: son entradas
locales, derivados o material aún no aprobado.

## Flujo reproducible

| Etapa | Entrada versionada o controlada | Script | Salida |
| --- | --- | --- | --- |
| Fuentes | Markdown y PDF locales aprobados | — | archivos en sus `source_path` |
| Manifests | `data/manifests/*.yaml` | — | identidad, procedencia, idioma y visibilidad |
| Ingesta | fuentes + manifests | `ingest_markdown.py`, `ingest_pdf.py` | `data/processed/*.json` |
| Chunks | documentos canónicos | `chunk_documents.py` | `chunks.jsonl` |
| Preguntas | `data/evaluation/questions.yaml` | `generate_evaluation_corpus_index.py` | índice de identidad del corpus |
| Métricas | preguntas + chunks | `evaluate_retrieval.py` | `reports/retrieval/*.json` |

Ejecuta la reconstrucción en este orden:

```bash
# 1. Catálogo de publicaciones y un chunk-resumen por paper.
uv run python scripts/generate_publication_metadata.py

# 2. Documentos Markdown, incluidas las contrapartes bilingües disponibles.
for manifest in data/manifests/*.yaml; do
  if [[ "$manifest" != data/manifests/arXiv-* ]]; then
    uv run python scripts/ingest_markdown.py "$manifest"
  fi
done

# 3. PDFs de papers de arXiv.
for manifest in data/manifests/arXiv-*.yaml; do
  uv run python scripts/ingest_pdf.py "$manifest"
done

# 4. Corpus de retrieval: parámetros fijados por el experimento actual.
uv run python scripts/chunk_documents.py \
  --chunk-size-words 250 \
  --overlap-words 40

# 5. Huellas e IDs con los que se validan las etiquetas del benchmark.
uv run python scripts/generate_evaluation_corpus_index.py

# 6. Verifica que el corpus local coincide exactamente con la release.
uv run python scripts/validate_release.py

# 7. Evaluación oficial elegida. El nombre evita sobrescribir otra corrida.
uv run python scripts/evaluate_retrieval.py \
  --method planned_bm25 \
  --corpus full_text \
  --questions \
    data/evaluation/questions.yaml \
    data/evaluation/questions_famaf_teaching_en.yaml \
    data/evaluation/questions_famaf_teaching_es.yaml \
  --full-text-chunks data/processed/chunks/chunks.jsonl \
  --k 7 \
  --output reports/retrieval/planned_bm25_full_text_k7_by_topic_language.json
```

El último comando muestra `chunk recall`, `MRR`, `nDCG`, `document recall`,
latencia mediana y P95. También muestra el agregado completo y los cortes por
idioma y por tema-idioma. El informe JSON conserva esos agregados y la
recuperación de cada pregunta: úsalo para inspeccionar fallos, no solo el
promedio.

## Consultas exhaustivas de coautoría

`PublicationCatalogRetriever` no busca nombres dentro de los cuerpos de PDF.
Filtra la lista estructurada de autores y devuelve todos los papers que
contienen cada autor solicitado. Por ejemplo:

```bash
uv run python scripts/search_publications.py \
  --author "Marcelo Lares" \
  --author "Daza-Perilla" \
  --limit 20
```

La coincidencia admite acentos, guiones y nombres escritos como iniciales; el
resultado es exhaustivo dentro del catálogo local, sujeto a `--limit`.

## Corpus suplementario sin modificar el benchmark

Para preparar nuevas fuentes antes de etiquetar preguntas o cambiar las
métricas oficiales, escribe sus chunks en un directorio separado. Por ejemplo,
los registros de IThreex se pueden crear sin modificar
`data/processed/chunks/chunks.jsonl`:

```bash
uv run python scripts/chunk_documents.py \
  --document-id ithreex_kolektor_mvp1 \
  --document-id ithreex_kolektor_mvp2 \
  --document-id ithreex_pueblo_nativo \
  --document-id ithreex_inverfin \
  --document-id ithreex_procordoba \
  --document-id ithreex_ss_servicios \
  --document-id ithreex_animalia \
  --document-id ithreex_molibdeno \
  --output-directory data/processed/ithreex_chunks
```

El resultado queda en `data/processed/ithreex_chunks/`. No lo mezcles con el
corpus oficial: una vez que sus documentos entren en el corpus canónico, sus
IDs deben dejar de pasarse como un segundo archivo de chunks.

## Controles antes de aceptar un resultado

```bash
uv run pytest -q
uv run ruff check src tests scripts
git diff --check
```

Cuando cambies una fuente o el tamaño de chunk, vuelve a ejecutar desde la
ingesta y revisa las etiquetas `relevant_chunk_ids`: los límites de los chunks
pueden haber cambiado aunque el texto sea el mismo.

## Lo que aún falta

1. **Adquisición versionada o automatizada de fuentes.** El repositorio no
   puede reconstruir un clone vacío porque los inputs locales están ignorados.
   Falta un inventario transferible de archivos públicos, con hashes y un
   proceso explícito de descarga o de copia aprobada.
2. **Un comando orquestador.** Hoy el proceso es reproducible con los scripts
   anteriores, pero los bucles se ejecutan desde la shell. Un futuro
   `rebuild_corpus.py` o `Makefile` puede encapsular el orden y comprobar que
   no falte ninguna fuente.
3. **Benchmark español curado.**
   `data/evaluation/questions_es.draft.yaml` es una traducción candidata: sus
   `relevant_chunk_ids` conservan etiquetas inglesas y deben revisarse contra
   los chunks traducidos antes de renombrarla a `questions_es.yaml`.
4. **Revisión humana de traducciones y procedencia de citas.** Las
   traducciones facilitan retrieval; la fuente original sigue siendo la
   autoridad. Antes de una publicación pública conviene mostrar esa relación
   junto a una cita derivada.

La traducción por Batch y su revisión están documentadas en
[bilingual_dataset.md](bilingual_dataset.md). La explicación detallada de las
métricas está en [development.md](development.md).
