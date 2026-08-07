# AskML RAG notebooks

Run Jupyter from the repository root:

```bash
uv run jupyter lab
```

Open the notebooks in this order:

1. `01_corpus_and_chunking.ipynb` — source normalization, canonical documents,
   chunk metadata, overlap, and chunking reproducibility.
2. `02_retrieval_comparison.ipynb` — manually labelled questions, BM25,
   semantic retrieval, hybrid RRF, and Recall@k.
3. `03_embedding_visualization.ipynb` — chunk embeddings, a PCA projection,
   query neighbours, and chunk-to-chunk similarity.

Matplotlib is already included in the locked project dependencies.

The notebooks only read generated data. Regenerate it first when source files,
manifests, or chunking settings change:

```bash
uv run python scripts/ingest_markdown.py data/manifests/<manifest>.yaml
uv run python scripts/ingest_pdf.py data/manifests/<publication-manifest>.yaml
uv run python scripts/generate_publication_metadata.py
uv run python scripts/chunk_documents.py
```

For reproducible benchmark runs, use `scripts/evaluate_retrieval.py`. The
notebooks are intended for exploration and visualization, not as the canonical
evaluation pipeline.
