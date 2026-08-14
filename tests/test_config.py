import os
from pathlib import Path

from askml_rag.config import load_dotenv


def test_load_dotenv_loads_quoted_values_without_overwriting_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# local secrets\nOPENAI_API_KEY='local-key'\nEXISTING=from-file\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EXISTING", "from-environment")

    loaded = load_dotenv(dotenv)

    assert loaded == {"OPENAI_API_KEY", "EXISTING"}
    assert os.environ["OPENAI_API_KEY"] == "local-key"
    assert os.environ["EXISTING"] == "from-environment"
