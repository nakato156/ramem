import json
from pathlib import Path

import pytest

from ramem.config import GenerationConfig
from ramem.generation.model_manager import MANIFEST_NAME, ModelManager, sha256


def test_model_manifest_verifies_declared_files(tmp_path: Path) -> None:
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf-test")
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": model.name,
                        "bytes": model.stat().st_size,
                        "sha256": sha256(model),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ModelManager(tmp_path, GenerationConfig()).verify()

    assert result["files"][0]["sha256"] == sha256(model)


def test_model_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf-test")
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": model.name,
                        "bytes": model.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256"):
        ModelManager(tmp_path, GenerationConfig()).verify()


def test_model_manifest_rejects_floating_recorded_snapshot(tmp_path: Path) -> None:
    model = tmp_path / "tiny.gguf"
    model.write_bytes(b"gguf-test")
    (tmp_path / ".ramem_revision").write_text("main\n", encoding="utf-8")
    (tmp_path / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": model.name,
                        "bytes": model.stat().st_size,
                        "sha256": sha256(model),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not immutable"):
        ModelManager(tmp_path, GenerationConfig()).verify()
