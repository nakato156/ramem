from __future__ import annotations

import importlib.util
import json
import os
import platform
import shlex
import shutil
import signal
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Annotated

import typer
from platformdirs import user_data_path
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

from ramem import __version__
from ramem.config import AppConfig, load_config, resolve_data_paths
from ramem.conversation.models import ContextBundle, ConversationSession, RetrievedMemory
from ramem.conversation.service import ConversationService
from ramem.conversation.store import ConversationStore
from ramem.evaluation.memory import (
    evaluate_memory_retrieval,
    load_memory_benchmark,
    write_benchmark_result,
)
from ramem.generation.backends import GenerationCancelled
from ramem.generation.model_manager import ModelManager

app = typer.Typer(
    help="RAMEM V1 — chat local con memoria conversacional RAG",
    invoke_without_command=True,
    no_args_is_help=False,
)
session_app = typer.Typer(help="Administrar conversaciones")
memory_app = typer.Typer(help="Inspeccionar y reconstruir la memoria")
model_app = typer.Typer(help="Descargar y verificar el modelo finetuneado")
app.add_typer(session_app, name="session")
app.add_typer(memory_app, name="memory")
app.add_typer(model_app, name="model")

console = Console()
ConfigOption = Annotated[
    Path | None,
    typer.Option("--config", exists=True, dir_okay=False, help="YAML de configuración"),
]

SLASH_COMMANDS = [
    "/new",
    "/sessions",
    "/resume",
    "/rename",
    "/context",
    "/sources",
    "/memory",
    "/forget",
    "/model",
    "/status",
    "/help",
    "/exit",
]


class ExitRequested(RuntimeError):
    pass


def _config(path: Path | None) -> AppConfig:
    return load_config(path)


def _store(config: AppConfig) -> ConversationStore:
    sqlite_path, _, _ = resolve_data_paths(config)
    return ConversationStore(sqlite_path)


def _service(config: AppConfig) -> ConversationService:
    return ConversationService(config)


def _model_root(config: AppConfig) -> Path:
    sqlite_path, _, _ = resolve_data_paths(config)
    return sqlite_path.parent / "models" / "ramem-gemma-1b"


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
        return round(
            float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")) / 1024**3,
            2,
        )
    return None


def _gpu_details() -> dict[str, object]:
    if importlib.util.find_spec("torch") is None:
        return {"available": False, "device": None, "bf16": False}
    import torch  # type: ignore[import-not-found]

    available = bool(torch.cuda.is_available())
    return {
        "available": available,
        "device": torch.cuda.get_device_name(0) if available else None,
        "bf16": bool(available and torch.cuda.is_bf16_supported()),
    }


@app.callback()
def main(
    ctx: typer.Context,
    new: Annotated[bool, typer.Option("--new", help="Iniciar una conversación nueva")] = False,
    config: ConfigOption = None,
) -> None:
    """Abre el chat interactivo cuando no se especifica un subcomando."""
    if ctx.invoked_subcommand is None:
        run_repl(_config(config), new=new)


@app.command()
def setup(config: ConfigOption = None) -> None:
    """Inicializa el almacenamiento local y comprueba el índice."""
    loaded = _config(config)
    service = _service(loaded)
    completed, failed = service.worker.drain()
    sqlite_path, lance_path, _ = resolve_data_paths(loaded)
    console.print("[green]RAMEM inicializado.[/green]")
    console.print(f"SQLite: {sqlite_path}")
    console.print(f"LanceDB: {lance_path}")
    console.print(
        f"Índice: {service.index.count()} chunks; trabajos {completed} ok/{failed} fallidos"
    )


@app.command()
def ask(
    text: Annotated[str, typer.Argument(help="Mensaje para RAMEM")],
    session_id: Annotated[str | None, typer.Option("--session")] = None,
    new: Annotated[bool, typer.Option("--new")] = False,
    config: ConfigOption = None,
) -> None:
    """Ejecuta un turno sin abrir el REPL."""
    service = _service(_config(config))
    session = _resolve_session(service, session_id=session_id, new=new)
    if not _run_turn(service, session, text):
        raise typer.Exit(130)


@app.command()
def doctor(config: ConfigOption = None) -> None:
    """Inspecciona el runtime sin cargar modelos ni descargar archivos."""
    loaded = _config(config)
    sqlite_path, lance_path, _ = resolve_data_paths(loaded)
    disk = shutil.disk_usage(sqlite_path.parent if sqlite_path.parent.exists() else Path.cwd())
    with sqlite3.connect(":memory:") as connection:
        try:
            connection.execute("CREATE VIRTUAL TABLE probe USING fts5(text)")
            fts5 = True
        except sqlite3.OperationalError:
            fts5 = False
    gguf = _model_root(loaded) / loaded.generation.gguf_filename
    model_verified = False
    model_error: str | None = None
    try:
        ModelManager(_model_root(loaded), loaded.generation).verify()
        model_verified = True
    except (FileNotFoundError, ValueError) as error:
        model_error = type(error).__name__
    gpu = _gpu_details()
    result = {
        "ramem": __version__,
        "python": sys.version.split()[0],
        "python_ok": (3, 12) <= sys.version_info[:2] < (3, 13),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": _total_ram_gb(),
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "disk_ok": disk.free >= 5 * 1024**3,
        "gpu": gpu,
        "sqlite_fts5": fts5,
        "sqlite_path": str(sqlite_path),
        "lancedb_path": str(lance_path),
        "lancedb": importlib.util.find_spec("lancedb") is not None,
        "embeddinggemma_runtime": importlib.util.find_spec("sentence_transformers") is not None,
        "llama_cpp": importlib.util.find_spec("llama_cpp") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "gguf_present": gguf.is_file(),
        "gguf_path": str(gguf),
        "model_verified": model_verified,
        "model_error": model_error,
    }
    console.print_json(json.dumps(result))


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query")] = "¿Qué recuerdas de mí?",
    repeats: Annotated[int, typer.Option("--repeats", min=1)] = 10,
    dataset: Annotated[Path | None, typer.Option("--dataset", exists=True)] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    k: Annotated[int, typer.Option("--k", min=1)] = 10,
    config: ConfigOption = None,
) -> None:
    """Mide recuperación local o ejecuta RaMem-Memory-ES reproduciblemente."""
    loaded = _config(config)
    if dataset is not None:
        cases = load_memory_benchmark(dataset)
        if (
            any(case.split == "holdout" for case in cases)
            and os.environ.get("RAMEM_RELEASE_CANDIDATE_FROZEN") != "yes"
        ):
            raise typer.BadParameter("el holdout requiere RAMEM_RELEASE_CANDIDATE_FROZEN=yes")
        with tempfile.TemporaryDirectory(prefix="ramem-benchmark-") as temporary:
            root = Path(temporary)
            isolated = loaded.model_copy(
                update={
                    "storage": loaded.storage.model_copy(
                        update={
                            "index_path": root / "ramem.sqlite3",
                            "lance_path": root / "memory.lance",
                        }
                    ),
                    "telemetry": loaded.telemetry.model_copy(
                        update={"traces_dir": root / "traces"}
                    ),
                }
            )
            result = evaluate_memory_retrieval(
                _service(isolated),
                cases,
                k=k,
            )
        if output is not None:
            write_benchmark_result(result, output)
        console.print_json(result.model_dump_json())
        return

    service = _service(loaded)
    latencies: list[float] = []
    result_count = 0
    for _ in range(repeats):
        started = time.perf_counter()
        result_count = len(service.search(query))
        latencies.append((time.perf_counter() - started) * 1000)
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, int(len(ordered) * 0.95))
    console.print_json(
        json.dumps(
            {
                "query": query,
                "repeats": repeats,
                "results": result_count,
                "mean_ms": sum(latencies) / len(latencies),
                "p95_ms": ordered[p95_index],
                "indexed_chunks": service.index.count(),
            }
        )
    )


@session_app.command("list")
def session_list(
    config: ConfigOption = None,
) -> None:
    """Lista conversaciones guardadas."""
    _print_sessions(_store(_config(config)).list_sessions())


@session_app.command("show")
def session_show(
    session_id: str,
    config: ConfigOption = None,
) -> None:
    """Muestra una conversación completa."""
    store = _store(_config(config))
    session = store.get_session(session_id)
    if session is None:
        raise typer.BadParameter(f"sesión desconocida: {session_id}")
    console.print(f"[bold]{session.title}[/bold] [dim]{session.session_id}[/dim]")
    for message in store.messages(session_id):
        role = "Tú" if message.role == "user" else "RAMEM"
        console.print(f"\n[bold]{role}[/bold] · [dim]{message.created_at.isoformat()}[/dim]")
        console.print(Markdown(message.content))


@session_app.command("delete")
def session_delete(
    session_id: str,
    yes: Annotated[bool, typer.Option("--yes", help="Confirmar borrado")] = False,
    config: ConfigOption = None,
) -> None:
    """Borra una conversación y sus vectores."""
    if not yes:
        raise typer.BadParameter("usa --yes para confirmar el borrado")
    service = _service(_config(config))
    service.delete_session(session_id)
    console.print(f"[green]Sesión {session_id} eliminada.[/green]")


@memory_app.command("search")
def memory_search(
    query: str,
    config: ConfigOption = None,
) -> None:
    """Busca recuerdos sin generar una respuesta."""
    service = _service(_config(config))
    _print_memories(service.search(query), service)


@memory_app.command("rebuild")
def memory_rebuild(
    config: ConfigOption = None,
) -> None:
    """Reconstruye LanceDB completamente desde SQLite."""
    service = _service(_config(config))
    count = service.rebuild_index()
    console.print(f"[green]Índice reconstruido: {count} chunks.[/green]")


@memory_app.command("stats")
def memory_stats(
    config: ConfigOption = None,
) -> None:
    """Muestra estadísticas del almacenamiento y el índice."""
    service = _service(_config(config))
    payload = service.store.stats().model_dump(mode="json")
    payload["lancedb_chunks"] = service.index.count()
    console.print_json(json.dumps(payload))


@model_app.command("pull")
def model_pull(
    config: ConfigOption = None,
) -> None:
    """Descarga una revisión inmutable del modelo desde Hugging Face."""
    loaded = _config(config)
    manager = ModelManager(_model_root(loaded), loaded.generation)
    path = manager.pull(token=os.environ.get("HF_TOKEN"))
    console.print(f"[green]Modelo descargado y verificado en {path}.[/green]")


@model_app.command("verify")
def model_verify(
    config: ConfigOption = None,
) -> None:
    """Verifica todos los archivos declarados por el manifiesto."""
    loaded = _config(config)
    result = ModelManager(_model_root(loaded), loaded.generation).verify()
    console.print_json(json.dumps(result))


def run_repl(config: AppConfig, *, new: bool = False) -> None:
    service = _service(config)
    session = service.open_session(new=new)
    history_path = Path(user_data_path("ramem", "ramem")) / "prompt-history"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _accept_input(event) -> None:  # type: ignore[no-untyped-def]
        event.current_buffer.validate_and_handle()

    prompt = PromptSession[str](
        history=FileHistory(str(history_path)),
        completer=WordCompleter(SLASH_COMMANDS, sentence=True),
        key_bindings=bindings,
        multiline=True,
        bottom_toolbar="Esc+Enter enviar · Enter nueva línea · Ctrl+C cancelar",
    )
    _print_banner(service, session)
    interrupted = False
    while True:
        try:
            text = prompt.prompt("ramem > ").strip()
            interrupted = False
        except EOFError:
            break
        except KeyboardInterrupt:
            if interrupted:
                break
            interrupted = True
            console.print("[dim]Pulsa Ctrl+C otra vez o /exit para salir.[/dim]")
            continue
        if not text:
            continue
        if text.startswith("/"):
            try:
                session, should_exit = _slash(service, session, text)
            except (KeyError, ValueError) as error:
                console.print(f"[red]{error}[/red]")
                continue
            if should_exit:
                break
            continue
        try:
            _run_turn(service, session, text)
        except ExitRequested:
            break
        except KeyboardInterrupt:
            console.print("\n[yellow]Generación cancelada.[/yellow]")
        except Exception as error:
            console.print(f"\n[red]{type(error).__name__}: {error}[/red]")


def _run_turn(
    service: ConversationService,
    session: ConversationSession,
    text: str,
) -> bool:
    console.print()
    parts: list[str] = []
    cancelled = threading.Event()
    interrupts = 0
    original_handler = signal.getsignal(signal.SIGINT)

    def handle_interrupt(signum: int, frame: object) -> None:
        del signum, frame
        nonlocal interrupts
        interrupts += 1
        if interrupts == 1:
            cancelled.set()
            return
        raise ExitRequested()

    can_install_handler = threading.current_thread() is threading.main_thread()
    if can_install_handler:
        signal.signal(signal.SIGINT, handle_interrupt)
    try:
        with Live(Markdown(""), console=console, refresh_per_second=12) as live:

            def token(value: str) -> None:
                parts.append(value)
                live.update(Markdown("".join(parts)))

            try:
                result = service.ask(
                    session.session_id,
                    text,
                    on_token=token,
                    cancelled=cancelled.is_set,
                )
            except GenerationCancelled:
                console.print(
                    "\n[yellow]Generación cancelada; "
                    "el mensaje del usuario quedó guardado.[/yellow]"
                )
                return False
            live.update(Markdown(result.answer))
    finally:
        if can_install_handler:
            signal.signal(signal.SIGINT, original_handler)
    console.print(
        f"[dim]{len(result.context.memories)} recuerdos · "
        f"{result.context.memory_tokens} tokens de memoria[/dim]"
    )
    return True


def _slash(
    service: ConversationService,
    session: ConversationSession,
    raw: str,
) -> tuple[ConversationSession, bool]:
    args = shlex.split(raw)
    command = args[0].casefold()
    if command in {"/exit", "/quit"}:
        return session, True
    if command == "/help":
        console.print(Markdown(_slash_help()))
    elif command == "/new":
        session = service.open_session(new=True, title=" ".join(args[1:]) or None)
        console.print(
            f"[green]Nueva sesión:[/green] {session.title} [dim]{session.session_id}[/dim]"
        )
    elif command == "/sessions":
        _print_sessions(service.store.list_sessions())
    elif command == "/resume":
        if len(args) != 2:
            raise ValueError("uso: /resume <id>")
        found = service.store.get_session(args[1])
        if found is None:
            raise KeyError(f"sesión desconocida: {args[1]}")
        session = found
        console.print(f"[green]Sesión activa:[/green] {session.title}")
    elif command == "/rename":
        if len(args) < 2:
            raise ValueError("uso: /rename <nombre>")
        session = service.store.rename_session(session.session_id, " ".join(args[1:]))
        console.print(f"[green]Renombrada:[/green] {session.title}")
    elif command == "/context":
        _print_context(service.last_context)
    elif command == "/sources":
        _print_sources(
            service.last_context,
            service,
            service.last_cited_evidence_ids,
        )
    elif command == "/memory":
        if len(args) < 2:
            raise ValueError("uso: /memory <consulta>")
        _print_memories(service.search(" ".join(args[1:])), service)
    elif command == "/forget":
        if args[1:] == ["all"]:
            confirmation = input("Escribe BORRAR TODO para confirmar: ")
            if confirmation == "BORRAR TODO":
                service.delete_all()
                session = service.open_session(new=True)
                console.print("[green]Toda la memoria fue eliminada.[/green]")
        elif len(args) == 3 and args[1] == "session":
            confirmation = input(f"Escribe {args[2]} para confirmar: ")
            if confirmation == args[2]:
                service.delete_session(args[2])
                if str(session.session_id) == args[2]:
                    session = service.open_session(new=True)
                console.print("[green]Sesión eliminada.[/green]")
        else:
            raise ValueError("uso: /forget session <id> | /forget all")
    elif command == "/model":
        console.print_json(service.backend.info.model_dump_json())
    elif command == "/status":
        payload = service.store.stats().model_dump(mode="json")
        payload["lancedb_chunks"] = service.index.count()
        console.print_json(json.dumps(payload))
    else:
        raise ValueError(f"comando desconocido: {command}")
    return session, False


def _resolve_session(
    service: ConversationService,
    *,
    session_id: str | None,
    new: bool,
) -> ConversationSession:
    if session_id:
        session = service.store.get_session(session_id)
        if session is None:
            raise typer.BadParameter(f"sesión desconocida: {session_id}")
        return session
    return service.open_session(new=new)


def _print_banner(service: ConversationService, session: ConversationSession) -> None:
    console.print("[bold cyan]RAMEM V1[/bold cyan] · memoria conversacional local")
    console.print(f"Sesión: [bold]{session.title}[/bold] [dim]{session.session_id}[/dim]")
    console.print("[dim]Escribe /help para ver comandos.[/dim]\n")


def _print_sessions(sessions: tuple[ConversationSession, ...]) -> None:
    table = Table("ID", "Título", "Actualizada")
    for session in sessions:
        table.add_row(str(session.session_id), session.title, session.updated_at.isoformat())
    console.print(table)


def _print_memories(
    memories: tuple[RetrievedMemory, ...],
    service: ConversationService,
) -> None:
    if not memories:
        console.print("[dim]Sin recuerdos relevantes.[/dim]")
        return
    for number, memory in enumerate(memories, start=1):
        session = service.store.get_session(memory.chunk.session_id)
        title = session.title if session else str(memory.chunk.session_id)
        console.print(
            f"\n[bold][D{number}] {title}[/bold] · "
            f"[dim]{memory.chunk.created_at.isoformat()} · {memory.fused_score:.5f}[/dim]"
        )
        console.print(memory.chunk.text)


def _print_context(context: ContextBundle | None) -> None:
    if context is None:
        console.print("[dim]Aún no hay contexto recuperado.[/dim]")
        return
    console.print(Markdown(context.rendered_context or "_Sin memoria recuperada._"))
    console.print(
        f"[dim]{context.memory_tokens}/{context.token_budget} tokens de memoria; "
        f"{context.recent_tokens} recientes[/dim]"
    )


def _print_sources(
    context: ContextBundle | None,
    service: ConversationService,
    cited_evidence_ids: tuple[str, ...],
) -> None:
    cited = set(cited_evidence_ids)
    memories = (
        tuple(item for item in context.memories if item.evidence_id in cited)
        if context is not None
        else ()
    )
    if not memories:
        console.print("[dim]La última respuesta no tuvo fuentes de memoria.[/dim]")
        return
    table = Table("Evidencia", "Sesión", "Fecha", "Mensajes")
    for memory in memories:
        session = service.store.get_session(memory.chunk.session_id)
        source_messages = {
            item.message_id: item
            for item in service.store.messages(memory.chunk.session_id)
            if item.message_id in memory.chunk.message_ids
        }
        rendered_messages = "\n".join(
            f"{source_messages[message_id].role}: {source_messages[message_id].content[:160]}"
            for message_id in memory.chunk.message_ids
            if message_id in source_messages
        )
        table.add_row(
            memory.evidence_id,
            session.title if session else str(memory.chunk.session_id),
            memory.chunk.created_at.isoformat(),
            rendered_messages,
        )
    console.print(table)


def _slash_help() -> str:
    return """\
`/new [nombre]` · nueva sesión
`/sessions` · listar sesiones
`/resume <id>` · reanudar
`/rename <nombre>` · renombrar
`/context` · contexto de la última respuesta
`/sources` · procedencia de citas
`/memory <consulta>` · buscar recuerdos
`/forget session <id>` o `/forget all` · borrar
`/model` · backend activo
`/status` · almacenamiento e índice
`/help` · ayuda
`/exit` · salir
"""


if __name__ == "__main__":
    app()
