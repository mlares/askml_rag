import re
import hashlib
import json
from pathlib import Path
import yaml
from markdownify import markdownify
from askml_rag.models import CanonicalDocument, SourceManifest

FRONT_MATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n.*?\r?\n---[ \t]*\r?\n?",
    flags=re.DOTALL,
)


def normalize_markdown(source: str) -> str:
    """Remove Jekyll front matter and normalize HTML/Markdown to Markdown."""
    without_front_matter = FRONT_MATTER_PATTERN.sub("", source, count=1)
    converted = markdownify(without_front_matter, heading_style="ATX")

    lines = [line.rstrip() for line in converted.splitlines()]
    return "\n".join(lines).strip() + "\n"


def ingest_markdown(
    manifest_path: Path,
    *,
    project_root: Path,
) -> CanonicalDocument:
    """Create one canonical document from a YAML manifest and Markdown source."""
    manifest_data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = SourceManifest.model_validate(manifest_data)

    root = project_root.resolve()
    source_path = (root / manifest.source_path).resolve()

    try:
        source_path.relative_to(root)
    except ValueError as error:
        message = "The manifest source_path must stay inside the project root."
        raise ValueError(message) from error

    source = source_path.read_text(encoding="utf-8")
    body_markdown = normalize_markdown(source)
    content_hash = hashlib.sha256(body_markdown.encode("utf-8")).hexdigest()

    return CanonicalDocument(
        **manifest.model_dump(),
        content_hash=content_hash,
        body_markdown=body_markdown,
    )


def write_canonical_document(
    document: CanonicalDocument,
    *,
    output_directory: Path,
) -> Path:
    """Write a canonical document as reproducible JSON."""
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{document.document_id}.json"

    serialized = json.dumps(
        document.model_dump(mode="json"),
        indent=2,
        ensure_ascii=False,
    )
    output_path.write_text(f"{serialized}\n", encoding="utf-8")

    return output_path
