from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASETS: dict[str, dict[str, Any]] = {
    "squad-es": {
        "repo_id": "ccasimiro/squad_es",
        "config": "v1.1.0",
        "license": "cc-by-4.0",
        "splits": ("train",),
        "mode": "parquet",
        "revision_ref": "refs/convert/parquet",
        "files": {"train": ("v1.1.0/train/0000.parquet",)},
    },
    "mlqa-es-dev": {
        "repo_id": "facebook/mlqa",
        "config": "mlqa.es.es",
        "license": "cc-by-sa-3.0",
        "splits": ("validation",),
        "mode": "parquet",
        "revision_ref": "refs/convert/parquet",
        "files": {"validation": ("mlqa.es.es/validation/0000.parquet",)},
    },
    "xquad-es-final": {
        "repo_id": "google/xquad",
        "config": "xquad.es",
        "license": "cc-by-sa-4.0",
        "splits": ("validation",),
        "mode": "parquet",
        "revision_ref": "refs/convert/parquet",
        "files": {"validation": ("xquad.es/validation/0000.parquet",)},
        "release_only": True,
    },
    "miracl-es": {
        "repo_id": "miracl/miracl",
        "license": "apache-2.0",
        "patterns": ("miracl-v1.0-es/**", "README.md"),
        "mode": "snapshot",
    },
    "miracl-corpus-es": {
        "repo_id": "miracl/miracl-corpus",
        "license": "apache-2.0",
        "patterns": ("miracl-corpus-v1.0-es/**", "README.md"),
        "mode": "snapshot",
    },
}


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode())
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _license_value(card_data: Any) -> str:
    value = getattr(card_data, "license", None)
    if isinstance(value, list):
        return str(value[0]).casefold()
    return str(value or "").casefold()


def download_dataset(name: str, output_root: Path, *, release_test: bool = False) -> dict[str, Any]:
    spec = DATASETS[name]
    if spec.get("release_only") and not release_test:
        raise PermissionError(
            f"{name} is a reserved final holdout; pass --release-test only after "
            "freezing the candidate"
        )
    try:
        from datasets import DatasetDict, load_dataset  # type: ignore[import-not-found]
        from huggingface_hub import (  # type: ignore[import-not-found]
            HfApi,
            hf_hub_download,
            snapshot_download,
        )
    except ImportError as error:
        raise RuntimeError("Run `uv sync --extra training` before downloading data") from error

    repo_id = str(spec["repo_id"])
    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).dataset_info(repo_id)
    actual_license = _license_value(info.card_data)
    expected_license = str(spec["license"])
    if actual_license != expected_license:
        raise RuntimeError(
            f"License mismatch for {repo_id}: expected {expected_license}, got {actual_license!r}"
        )
    source_revision = str(info.sha)
    revision = source_revision
    if spec["mode"] == "parquet":
        revision = str(
            HfApi(token=token).dataset_info(repo_id, revision=str(spec["revision_ref"])).sha
        )
    destination = output_root / name
    if destination.exists():
        raise FileExistsError(f"{destination} already exists; raw datasets are immutable")
    destination.parent.mkdir(parents=True, exist_ok=True)

    if spec["mode"] == "datasets":
        loaded = DatasetDict()
        for split in spec["splits"]:
            loaded[str(split)] = load_dataset(
                repo_id,
                str(spec["config"]),
                split=str(split),
                revision=revision,
                token=token,
            )
        loaded.save_to_disk(destination)
    elif spec["mode"] == "parquet":
        parquet_files = {
            str(split): [
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=str(filename),
                    revision=revision,
                    token=token,
                )
                for filename in filenames
            ]
            for split, filenames in spec["files"].items()
        }
        loaded = load_dataset("parquet", data_files=parquet_files)
        loaded.save_to_disk(destination)
    else:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            allow_patterns=list(spec["patterns"]),
            local_dir=destination,
            token=token,
        )
    return {
        "name": name,
        "repo_id": repo_id,
        "revision": revision,
        "source_revision": source_revision,
        "license": actual_license,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "tree_sha256": _tree_hash(destination),
        "path": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download approved real datasets from Hugging Face"
    )
    parser.add_argument(
        "--dataset", choices=(*DATASETS, "all"), default="squad-es", help="Dataset to download"
    )
    parser.add_argument(
        "--output", type=Path, default=Path(os.environ.get("RAMEM_RAW_DATA_DIR", "data/raw"))
    )
    parser.add_argument(
        "--release-test",
        action="store_true",
        help="Permit an explicitly reserved final holdout after the candidate is frozen",
    )
    args = parser.parse_args()
    names = (
        tuple(
            name
            for name, spec in DATASETS.items()
            if args.release_test or not spec.get("release_only")
        )
        if args.dataset == "all"
        else (args.dataset,)
    )
    records = [
        download_dataset(name, args.output, release_test=args.release_test) for name in names
    ]
    manifest_path = args.output / "download_manifest.json"
    existing: list[dict[str, Any]] = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))["datasets"]
    by_name = {record["name"]: record for record in (*existing, *records)}
    manifest_path.write_text(
        json.dumps({"version": 1, "datasets": list(by_name.values())}, indent=2), encoding="utf-8"
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
