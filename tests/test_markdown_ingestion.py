import json
from askml_rag.ingestion.markdown import (
    ingest_markdown,
    normalize_markdown,
    write_canonical_document,
)
from textwrap import dedent


def test_normalize_markdown_removes_jekyll_front_matter_and_html() -> None:
    source = """
          ---
          layout: single
          title: Example
          ---

          <h1>Example heading</h1>

          <p>A short <strong>public</strong> biography.</p>
          """
    source = dedent(source).lstrip()

    normalized = normalize_markdown(source)

    assert "layout: single" not in normalized
    assert "title: Example" not in normalized
    assert "# Example heading" in normalized
    assert "public" in normalized


def test_ingest_markdown_creates_a_canonical_document(tmp_path) -> None:
    raw_path = tmp_path / "data" / "raw" / "example.md"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        dedent("""
        ---
        layout: single
        ---
        <h1>Example heading</h1>
        <p>A short <strong>public</strong> biography.</p>
        """).lstrip(),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "data" / "manifests" / "example.yaml"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        dedent("""
          document_id: example_source
          title: Example source
          document_type: website
          authors:
            - Marcelo Lares
          source_url: https://www.mlares.space/
          source_path: data/raw/example.md
          visibility: public
          topics:
            - example
          """).lstrip(),
        encoding="utf-8",
    )

    document = ingest_markdown(manifest_path, project_root=tmp_path)

    assert document.document_id == "example_source"
    assert document.content_hash
    assert len(document.content_hash) == 64
    assert "layout: single" not in document.body_markdown
    assert "# Example heading" in document.body_markdown

    output_path = write_canonical_document(
        document,
        output_directory=tmp_path / "data" / "processed",
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["document_id"] == "example_source"
    assert payload["content_hash"] == document.content_hash
