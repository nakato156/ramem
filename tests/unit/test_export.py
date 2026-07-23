from pathlib import Path

from ramem.training.export import ExportConfig, _preferred_export_dtype


def test_export_config_keeps_artifact_paths() -> None:
    config = ExportConfig(
        base_model_id="google/gemma-3-1b-it",
        adapter_path=Path("artifacts/adapter"),
        output_dir=Path("artifacts/merged"),
    )

    assert config.adapter_path == Path("artifacts/adapter")
    assert config.output_dir == Path("artifacts/merged")


def test_export_dtype_does_not_probe_bf16_without_cuda() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def is_bf16_supported() -> bool:
            raise AssertionError("BF16 must not be probed when CUDA is disabled")

    class FakeTorch:
        cuda = FakeCuda()
        float16 = object()
        bfloat16 = object()

    assert _preferred_export_dtype(FakeTorch(), use_cuda=False) is FakeTorch.float16
    assert (
        _preferred_export_dtype(FakeTorch(), use_cuda=False, requested="bfloat16")
        is FakeTorch.bfloat16
    )
