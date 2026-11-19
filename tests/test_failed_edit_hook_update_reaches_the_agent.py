"""A failed edit-hook update is told to the agent, not only to stderr.

The installed Claude Code hook ends in ``|| true``, so the exit code is
lost and stderr is not shown to the model. On PostToolUse the hook now
prints ``additionalContext`` JSON saying the graph was not updated.
"""

from __future__ import annotations

import io
import json
import subprocess

import pytest

from code_review_graph import hook_update
from code_review_graph.tools import build


def _repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")
    return repo


def _run(monkeypatch, payload, existing_only=False):
    monkeypatch.setattr(hook_update.sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(SystemExit) as exc:
        hook_update.run_update_hook(existing_only=existing_only)
    return exc.value.code


def test_failure_on_post_tool_use_prints_additional_context(monkeypatch, tmp_path, capsys):
    repo = _repo(tmp_path)

    def boom(**_kwargs):
        raise OSError("database is locked")

    monkeypatch.setattr(build, "build_or_update_graph", boom)
    payload = {"hook_event_name": "PostToolUse", "cwd": str(repo),
               "tool_input": {"file_path": str(repo / "a.py")}}
    assert _run(monkeypatch, payload) == 1
    out = capsys.readouterr()
    assert "Error: database is locked" in out.err
    context = json.loads(out.out)["hookSpecificOutput"]
    assert context["hookEventName"] == "PostToolUse"
    assert "NOT updated" in context["additionalContext"]
    assert "database is locked" in context["additionalContext"]


def test_error_status_is_reported_too(monkeypatch, tmp_path, capsys):
    repo = _repo(tmp_path)
    monkeypatch.setattr(build, "build_or_update_graph",
                        lambda **_k: {"status": "error", "summary": "parse failed"})
    payload = {"hook_event_name": "PostToolUse", "cwd": str(repo)}
    assert _run(monkeypatch, payload) == 1
    assert "parse failed" in json.loads(capsys.readouterr().out)[
        "hookSpecificOutput"]["additionalContext"]


@pytest.mark.parametrize("payload_event, existing_only", [
    ("SessionStart", False),
    ("PostToolUse", True),  # the Codex hook: stdout stays as it was
])
def test_other_events_keep_stdout_empty(monkeypatch, tmp_path, capsys,
                                        payload_event, existing_only):
    repo = _repo(tmp_path)
    (repo / ".code-review-graph").mkdir()
    (repo / ".code-review-graph" / "graph.db").write_bytes(b"")
    monkeypatch.setattr(build, "build_or_update_graph",
                        lambda **_k: {"status": "error", "summary": "x"})
    assert _run(monkeypatch, {"hook_event_name": payload_event, "cwd": str(repo)},
                existing_only) == 1
    assert capsys.readouterr().out == ""


def test_success_prints_nothing(monkeypatch, tmp_path, capsys):
    repo = _repo(tmp_path)
    monkeypatch.setattr(build, "build_or_update_graph", lambda **_k: {"status": "ok"})
    monkeypatch.setattr(hook_update.sys, "stdin", io.StringIO(json.dumps(
        {"hook_event_name": "PostToolUse", "cwd": str(repo)})))
    hook_update.run_update_hook()
    assert capsys.readouterr().out == ""
