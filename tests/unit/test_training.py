from pathlib import Path

from ramem.training.train import latest_checkpoint


def test_latest_checkpoint_uses_numeric_step_and_ignores_invalid_names(tmp_path: Path) -> None:
    (tmp_path / "checkpoint-9").mkdir()
    expected = tmp_path / "checkpoint-120"
    expected.mkdir()
    (tmp_path / "checkpoint-invalid").mkdir()

    assert latest_checkpoint(tmp_path) == expected


def test_latest_checkpoint_returns_none_for_new_run(tmp_path: Path) -> None:
    assert latest_checkpoint(tmp_path) is None
