"""``unchecked`` names the callers a session never read and the tests it never ran.

The activity log is opt-in: without ``.code-review-graph/activity.jsonl`` the
edit hook writes nothing and prints nothing extra.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

from code_review_graph import hook_update
from code_review_graph.activity import (
    activity_path,
    is_test_command,
    log_event,
    record_tool_use,
    unchecked,
)
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, get_db_path

LIB = "def helper(x):\n    return x + 1\n"
APP = "from lib import helper\n\n\ndef main():\n    return helper(1)\n"
TEST = "from lib import helper\n\n\ndef test_helper():\n    assert helper(1) == 2\n"


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=CRG test",
         "-c", "user.email=crg-test@example.invalid", "-c", "commit.gpgsign=false", *args],
        check=True, capture_output=True, text=True,
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text(".code-review-graph/\n", encoding="utf-8")
    (repo / "lib.py").write_text(LIB, encoding="utf-8")
    (repo / "app.py").write_text(APP, encoding="utf-8")
    (repo / "tests" / "test_lib.py").write_text(TEST, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    (repo / "lib.py").write_text(LIB.replace("x + 1", "x + 2"), encoding="utf-8")
    store = GraphStore(get_db_path(repo))
    try:
        full_build(repo, store)
    finally:
        store.close()
    return repo.resolve()


def _only(report: dict) -> dict:
    (entry,) = [s for s in report["symbols"] if s["symbol"].endswith("helper")]
    return entry


def test_unread_caller_and_unrun_test_are_listed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    entry = _only(unchecked(repo))
    assert entry["unread_caller_files"] == ["app.py"]
    assert entry["unrun_tests"] == ["tests/test_lib.py"]


def test_reading_the_caller_and_running_the_test_clears_them(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    record_tool_use(repo, {"tool_name": "Read", "session_id": "s1",
                           "tool_input": {"file_path": str(repo / "app.py")}})
    record_tool_use(repo, {"tool_name": "Bash", "session_id": "s1",
                           "tool_input": {"command": "uv run pytest tests/test_lib.py -q"}})
    entry = _only(unchecked(repo, session="s1"))
    assert entry["unread_caller_files"] == []
    assert entry["unrun_tests"] == []
    assert entry["caller_files_read"] == 1 and entry["tests_run"] == 1
    # Another session's reads do not count for this one.
    assert _only(unchecked(repo, session="s2"))["unread_caller_files"] == ["app.py"]


def test_test_commands_are_recognised() -> None:
    assert is_test_command("pytest")
    assert is_test_command("cd x && uv run pytest tests/ -q")
    assert is_test_command("go test ./...")
    assert not is_test_command("git status")


def test_edit_hook_without_a_log_writes_and_prints_nothing(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    payload = {"hook_event_name": "PostToolUse", "session_id": "s1", "cwd": str(repo),
               "tool_name": "Edit", "tool_input": {"file_path": str(repo / "lib.py")}}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(hook_update, "_update_repo", lambda root: None)
    hook_update.run_update_hook()
    assert capsys.readouterr().out == ""
    assert not activity_path(repo).exists()
    assert not log_event(repo, "edit", path=str(repo / "lib.py"))


def test_edit_hook_with_a_log_tells_the_agent(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    activity_path(repo).touch()
    payload = {"hook_event_name": "PostToolUse", "session_id": "s1", "cwd": str(repo),
               "tool_name": "Edit", "tool_input": {"file_path": str(repo / "lib.py")}}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    monkeypatch.setattr(hook_update, "_update_repo", lambda root: None)
    hook_update.run_update_hook()
    note = json.loads(capsys.readouterr().out)["hookSpecificOutput"]["additionalContext"]
    assert "helper changed" in note and "0 read" in note
    assert '"kind": "edit"' in activity_path(repo).read_text(encoding="utf-8")
