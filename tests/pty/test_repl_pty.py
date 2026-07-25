import os
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX pseudoterminal required")


def test_repl_starts_and_exits_through_pseudoterminal(tmp_path: Path) -> None:
    pexpect = pytest.importorskip("pexpect")
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "retrieval": {
                    "embedding_provider": "hashing",
                    "embedding_model_id": "tests",
                    "embedding_dimension": 64,
                },
                "storage": {
                    "index_path": str(tmp_path / "ramem.sqlite3"),
                    "lance_path": str(tmp_path / "memory.lance"),
                },
                "telemetry": {"traces_dir": str(tmp_path / "traces")},
            }
        ),
        encoding="utf-8",
    )
    child = pexpect.spawn(
        sys.executable,
        ["-m", "ramem.cli.app", "--config", str(config)],
        encoding="utf-8",
        timeout=20,
    )

    child.expect("RAMEM V1")
    child.expect("ramem >")
    child.send("/exit")
    child.send("\x1b\r")
    child.expect(pexpect.EOF)

    assert child.exitstatus == 0


def test_ctrl_c_cancels_generation_and_confirmed_message_survives(tmp_path: Path) -> None:
    pexpect = pytest.importorskip("pexpect")
    config = tmp_path / "config.yaml"
    database = tmp_path / "ramem.sqlite3"
    config.write_text(
        yaml.safe_dump(
            {
                "retrieval": {
                    "embedding_provider": "hashing",
                    "embedding_model_id": "tests",
                    "embedding_dimension": 64,
                },
                "storage": {
                    "index_path": str(database),
                    "lance_path": str(tmp_path / "memory.lance"),
                },
                "telemetry": {"traces_dir": str(tmp_path / "traces")},
            }
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["RAMEM_PTY_CONFIG"] = str(config)
    child = pexpect.spawn(
        sys.executable,
        ["tests/pty/repl_harness.py"],
        cwd=str(Path.cwd()),
        env=environment,
        encoding="utf-8",
        timeout=20,
    )

    child.expect("ramem >")
    child.send("mensaje confirmado")
    child.send("\x1b\r")
    child.expect("respuesta")
    child.sendcontrol("c")
    child.expect("mensaje del usuario quedó guardado")
    child.expect("ramem >")
    child.send("/exit")
    child.send("\x1b\r")
    child.expect(pexpect.EOF)

    from ramem.conversation.store import ConversationStore

    store = ConversationStore(database)
    sessions = store.list_sessions()
    assert len(sessions) == 1
    messages = store.messages(sessions[0].session_id)
    assert len(messages) == 1
    assert messages[0].content == "mensaje confirmado"
