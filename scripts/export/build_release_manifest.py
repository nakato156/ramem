from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

MANIFEST_NAME = "ramem_model_manifest.json"
REQUIRED_PATTERNS = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "*.safetensors",
    "*q4_k_m.gguf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    root: Path,
    *,
    repo_id: str,
    source_model_revision: str,
    source_commit: str,
) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    if not re.fullmatch(r"[0-9a-f]{40,64}", source_model_revision):
        raise ValueError(
            "source_model_revision must be an immutable 40-64 character hexadecimal commit"
        )
    if not re.fullmatch(r"[0-9a-f]{40,64}", source_commit):
        raise ValueError("source_commit must be an immutable 40-64 character hexadecimal commit")
    for pattern in REQUIRED_PATTERNS:
        if not list(root.glob(pattern)):
            raise FileNotFoundError(f"release artifact missing required pattern: {pattern}")
    files = []
    for path in sorted(item for item in root.iterdir() if item.is_file()):
        if path.name == MANIFEST_NAME:
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "repo_id": repo_id,
        "source_model_revision": source_model_revision,
        "source_commit": source_commit,
        "weights_terms": "Gemma Terms of Use",
        "files": files,
    }
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the immutable RAMEM model release manifest")
    parser.add_argument("root", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-model-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        args.root,
        repo_id=args.repo_id,
        source_model_revision=args.source_model_revision,
        source_commit=args.source_commit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
