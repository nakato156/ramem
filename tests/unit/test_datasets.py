from pathlib import Path

import pytest

from ramem.evaluation.datasets import authorize_dataset, load_manifest


def test_manifest_loads_with_explicit_approval_states() -> None:
    manifest = load_manifest(Path("data/datasets_manifest.yaml"))
    assert manifest.datasets
    approved = {dataset.name for dataset in manifest.datasets if not dataset.blocked}
    assert {"MIRACL-es", "MIRACL-corpus-es", "SQuAD-es"}.issubset(approved)


def test_test_only_dataset_cannot_train() -> None:
    manifest = load_manifest(Path("data/datasets_manifest.yaml"))
    dataset = next(item for item in manifest.datasets if item.name == "MLQA-es")
    with pytest.raises(PermissionError):
        authorize_dataset(dataset, "validation", "train")


def test_mlqa_validation_is_authorized_for_evaluation() -> None:
    manifest = load_manifest(Path("data/datasets_manifest.yaml"))
    dataset = next(item for item in manifest.datasets if item.name == "MLQA-es")

    authorize_dataset(dataset, "validation", "evaluate")
