from pathlib import Path

import yaml
from typer.testing import CliRunner

from ramem.cli.app import app


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "retrieval": {
                    "embedding_provider": "hashing",
                    "embedding_model_id": "tests",
                    "embedding_dimension": 64,
                },
                "storage": {
                    "index_path": str(tmp_path / "ramem.sqlite3"),
                    "lance_path": str(tmp_path / "memory.lance"),
                },
                "telemetry": {"traces_dir": str(tmp_path / "traces")},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_setup_status_and_administrative_commands(tmp_path: Path) -> None:
    runner = CliRunner()
    config = _config(tmp_path)

    setup = runner.invoke(app, ["setup", "--config", str(config)])
    sessions = runner.invoke(app, ["session", "list", "--config", str(config)])
    stats = runner.invoke(app, ["memory", "stats", "--config", str(config)])
    doctor = runner.invoke(app, ["doctor", "--config", str(config)])

    assert setup.exit_code == 0, setup.output
    assert "RAMEM inicializado" in setup.output
    assert sessions.exit_code == 0, sessions.output
    assert stats.exit_code == 0, stats.output
    assert '"sessions": 0' in stats.output
    assert doctor.exit_code == 0, doctor.output
    assert '"sqlite_fts5": true' in doctor.output
