from pathlib import Path

from ramem.training.export import ExportConfig


def test_export_config_keeps_artifact_paths() -> None:
    config = ExportConfig(
        base_model_id="google/gemma-3-1b-it",
        adapter_path=Path("artifacts/adapter"),
        output_dir=Path("artifacts/merged"),
    )

    assert config.adapter_path == Path("artifacts/adapter")
    assert config.output_dir == Path("artifacts/merged")
