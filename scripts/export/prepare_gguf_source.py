from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

IMAGE_TOKEN = "<image_soft_token>"


def _clean(value: Any) -> Any:
    if isinstance(value, list):
        return [
            _clean(item)
            for item in value
            if item != IMAGE_TOKEN
            and not (isinstance(item, dict) and item.get("content") == IMAGE_TOKEN)
        ]
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key == "image_token":
                continue
            if isinstance(item, dict) and item.get("content") == IMAGE_TOKEN:
                continue
            cleaned[key] = _clean(item)
        return cleaned
    return value


def prepare(source: Path, destination: Path) -> Path:
    """Create a text-only staging directory accepted by llama.cpp's Gemma converter."""
    destination.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        target = destination / path.name
        if path.name in {"tokenizer.json", "tokenizer_config.json"}:
            payload = json.loads(path.read_text(encoding="utf-8"))
            target.write_text(
                json.dumps(_clean(payload), ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
        elif path.name == "model.safetensors":
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)
        elif path.is_file():
            shutil.copy2(path, target)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the canonical text-only Gemma export for GGUF conversion."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare(args.source, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
