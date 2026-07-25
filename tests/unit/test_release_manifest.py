import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path("scripts/export/build_release_manifest.py")
    spec = importlib.util.spec_from_file_location("build_release_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_manifest_requires_all_runtime_artifacts(tmp_path: Path) -> None:
    module = _module()
    for name in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "ramem-q4_k_m.gguf",
    ):
        (tmp_path / name).write_bytes(name.encode())

    result = module.build_manifest(
        tmp_path,
        repo_id="owner/model",
        source_model_revision="a" * 40,
        source_commit="b" * 40,
    )

    assert len(result["files"]) == 5
    assert (tmp_path / module.MANIFEST_NAME).is_file()


def test_release_manifest_rejects_floating_source_revision(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(ValueError, match="immutable"):
        module.build_manifest(
            tmp_path,
            repo_id="owner/model",
            source_model_revision="main",
            source_commit="b" * 40,
        )
