from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_environment(path: Path | None = None) -> bool:
    """Load project variables without replacing values injected by Lightning or the shell."""
    return load_dotenv(dotenv_path=path, override=False)
