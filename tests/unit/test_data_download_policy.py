from pathlib import Path

import pytest

from ramem.training.data import download_dataset


def test_final_holdout_requires_explicit_release_flag(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="reserved final holdout"):
        download_dataset("xquad-es-final", tmp_path)
