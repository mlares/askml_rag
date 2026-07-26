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

The third notebook uses Matplotlib. Add it to the project before opening that
notebook:

```bash
uv add matplotlib
```

The notebooks only read generated data. Regenerate it first when source files,
manifests, or chunking settings change:

```bash
uv run python scripts/ingest_markdown.py data/manifests/<manifest>.yaml
uv run python scripts/chunk_documents.py
```
