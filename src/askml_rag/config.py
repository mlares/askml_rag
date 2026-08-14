"""Small, dependency-free configuration helpers for local development."""

import os
from pathlib import Path


def load_dotenv(path: Path) -> set[str]:
    """Load simple KEY=VALUE entries without overwriting existing environment.

    The function intentionally supports the small `.env` format needed by this
    project. Production deployments should inject environment variables through
    their platform or secret manager instead.
    """
    if not path.is_file():
        return set()

    loaded_keys: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
        loaded_keys.add(key)

    return loaded_keys
