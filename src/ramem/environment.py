from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_environment(path: Path | None = None) -> bool:
    """Load project variables without replacing values injected by Lightning or the shell."""
    dotenv_path = str(path) if path else find_dotenv(usecwd=True)
    if not dotenv_path:
        return False
    return load_dotenv(dotenv_path=dotenv_path, override=False)
