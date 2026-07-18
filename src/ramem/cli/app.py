from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Annotated

import typer

from ramem import __version__
from ramem.config import load_config
from ramem.domain.models import QueryRequest
from ramem.orchestration.pipeline import build_pipeline
from ramem.retrieval.ingest import load_jsonl_documents
from ramem.retrieval.store import SQLiteDocumentStore

app = typer.Typer(help="RaMem V0 local-first research CLI", no_args_is_help=True)

ConfigOption = Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)]


def _root() -> Path:
    return Path.cwd()


def _total_ram_gb() -> float | None:
    if platform.system() == "Windows":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        windll = getattr(ctypes, "windll", None)
        if windll is not None and windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(float(status.total_physical) / 1024**3, 2)
        return None
    if hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(float(page_size * pages) / 1024**3, 2)
    return None


@app.command()
def doctor(config: ConfigOption = Path("configs/default.yaml")) -> None:
    """Inspect the local runtime without downloading models."""
    loaded = load_config(config)
    disk = shutil.disk_usage(_root())
    fts5 = False
    with sqlite3.connect(":memory:") as connection:
        try:
            connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
            fts5 = True
        except sqlite3.OperationalError:
            pass
    gpu = "not_checked (install torch for hardware probing)"
    bf16 = "not_checked"
    try:
        import torch  # type: ignore[import-not-found]

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
        bf16 = str(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except ImportError:
        pass
    result = {
        "ramem": __version__,
        "python": sys.version.split()[0],
        "python_ok": sys.version_info >= (3, 12),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": _total_ram_gb(),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "sqlite_fts5": fts5,
        "gpu": gpu,
        "bf16": bf16,
        "bitsandbytes": importlib.util.find_spec("bitsandbytes") is not None,
        "index": str(loaded.storage.index_path),
    }
    typer.echo(json.dumps(result, indent=2))


@app.command()
def ingest(
    source: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: ConfigOption = Path("configs/default.yaml"),
) -> None:
    """Ingest validated JSONL documents into the local baseline index."""
    loaded = load_config(config)
    documents = load_jsonl_documents(source)
    store = SQLiteDocumentStore(loaded.storage.index_path, loaded.retrieval.embedding_dimension)
    count = store.ingest(documents)
    typer.echo(f"Ingested {count} documents; index now contains {store.count()} documents.")


@app.command()
def query(
    text: Annotated[str, typer.Argument(help="Question to answer")],
    config: ConfigOption = Path("configs/default.yaml"),
) -> None:
    """Run one query through the typed offline pipeline."""
    loaded = load_config(config)
    state = build_pipeline(loaded, root=_root()).run(QueryRequest(query=text))
    typer.echo(state.final_answer)
    typer.echo(
        json.dumps({"request_id": str(state.request_id), "routes": state.route_labels}, default=str)
    )


if __name__ == "__main__":
    app()
