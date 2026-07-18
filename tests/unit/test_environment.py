import os
from pathlib import Path

from pytest import MonkeyPatch

from ramem.environment import load_environment


def test_environment_is_discovered_in_parent_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    variable = "RAMEM_TEST_PARENT_ENV"
    monkeypatch.delenv(variable, raising=False)
    (tmp_path / ".env").write_text(f"{variable}=loaded\n", encoding="utf-8")
    child = tmp_path / "project"
    child.mkdir()
    monkeypatch.chdir(child)

    assert load_environment()
    assert os.environ[variable] == "loaded"
