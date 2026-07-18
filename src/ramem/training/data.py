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
        "mode": "datasets",
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


def download_dataset(name: str, output_root: Path) -> dict[str, Any]:
    try:
        from datasets import DatasetDict, load_dataset  # type: ignore[import-not-found]
        from huggingface_hub import HfApi, snapshot_download  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Run `uv sync --extra training` before downloading data") from error

    spec = DATASETS[name]
    repo_id = str(spec["repo_id"])
    token = os.environ.get("HF_TOKEN")
    info = HfApi(token=token).dataset_info(repo_id)
    actual_license = _license_value(info.card_data)
    expected_license = str(spec["license"])
    if actual_license != expected_license:
        raise RuntimeError(
            f"License mismatch for {repo_id}: expected {expected_license}, got {actual_license!r}"
        )
    revision = str(info.sha)
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
    args = parser.parse_args()
    names = tuple(DATASETS) if args.dataset == "all" else (args.dataset,)
    records = [download_dataset(name, args.output) for name in names]
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
