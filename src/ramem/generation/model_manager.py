from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from ramem.config import GenerationConfig

MANIFEST_NAME = "ramem_model_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ModelManager:
    def __init__(self, root: Path, config: GenerationConfig) -> None:
        self.root = root
        self.config = config

    def pull(self, *, token: str | None = None) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        resolved_revision = (
            HfApi(token=token)
            .model_info(
                self.config.model_id,
                revision=self.config.model_revision,
            )
            .sha
        )
        if not resolved_revision:
            raise RuntimeError("Hugging Face did not resolve an immutable model revision")
        downloaded = snapshot_download(
            repo_id=self.config.model_id,
            revision=resolved_revision,
            local_dir=self.root,
            token=token,
        )
        (self.root / ".ramem_revision").write_text(resolved_revision + "\n", encoding="utf-8")
        self.verify(expected_revision=resolved_revision)
        return Path(downloaded)

    def verify(self, *, expected_revision: str | None = None) -> dict[str, Any]:
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Falta {MANIFEST_NAME} en {self.root}; no se aceptan pesos sin manifiesto."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest.get("files")
        if isinstance(files, dict):
            entries = [{"path": path, **metadata} for path, metadata in files.items()]
        elif isinstance(files, list):
            entries = files
        else:
            entries = []
        if not entries:
            raise ValueError("model manifest must declare a non-empty files collection")
        declared_repo = manifest.get("repo_id")
        if declared_repo is not None and declared_repo != self.config.model_id:
            raise ValueError("model manifest repo_id does not match configured model")
        recorded_revision_path = self.root / ".ramem_revision"
        recorded_revision = (
            recorded_revision_path.read_text(encoding="utf-8").strip()
            if recorded_revision_path.is_file()
            else None
        )
        snapshot_revision = expected_revision or recorded_revision
        if snapshot_revision and not re.fullmatch(r"[0-9a-f]{40,64}", snapshot_revision):
            raise ValueError("recorded Hugging Face snapshot revision is not immutable")
        verified: list[dict[str, Any]] = []
        root = self.root.resolve()
        for entry in entries:
            path = (self.root / str(entry["path"])).resolve()
            if path != root and root not in path.parents:
                raise ValueError(f"model manifest path escapes model root: {entry['path']}")
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            digest = sha256(path)
            if size != int(entry["bytes"]):
                raise ValueError(f"unexpected size for {path.name}: {size}")
            if digest.casefold() != str(entry["sha256"]).casefold():
                raise ValueError(f"unexpected SHA-256 for {path.name}")
            verified.append({"path": str(path), "bytes": size, "sha256": digest})
        return {
            "repo_id": self.config.model_id,
            "revision": snapshot_revision or self.config.model_revision,
            "source_model_revision": manifest.get("source_model_revision"),
            "files": verified,
        }
