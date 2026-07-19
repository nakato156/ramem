from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl


class DatasetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: str
    url: HttpUrl
    license: str
    license_status: str
    authorized_splits: tuple[str, ...]
    sha256: str | None = None
    function: str
    blocked: bool = True
    notes: str = ""


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    policy: str
    datasets: tuple[DatasetEntry, ...]


def load_manifest(path: Path) -> DatasetManifest:
    with path.open(encoding="utf-8") as handle:
        return DatasetManifest.model_validate(yaml.safe_load(handle))


def authorize_dataset(entry: DatasetEntry, split: str, purpose: str) -> None:
    if entry.blocked or entry.license_status != "verified" or not entry.sha256:
        raise PermissionError(f"dataset {entry.name} is blocked pending license/hash verification")
    if split not in entry.authorized_splits:
        raise PermissionError(f"split {split!r} is not authorized for {entry.name}")
    if purpose == "train" and entry.function not in {"train", "train_validation_test"}:
        raise PermissionError(f"dataset {entry.name} is not authorized for training")
