"""``live`` draws the files a session edited, read and left unchecked."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from code_review_graph.activity import activity_path, record_tool_use
from code_review_graph.live import live_state, render_live
from tests.test_unchecked_impact_from_activity import _repo


def test_state_splits_edited_read_and_unchecked(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    record_tool_use(repo, {"tool_name": "Read", "session_id": "s1",
                           "tool_input": {"file_path": str(repo / "app.py")}})
    state = live_state(repo)
    assert state["edited"] == ["lib.py"]
    assert state["read"] == ["app.py"]
    assert state["unchecked"] == ["tests/test_lib.py"]


def test_page_reloads_and_colours_the_sets(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "live.html"
    render_live(repo, out, interval=2)
    page = out.read_text(encoding="utf-8")
    assert '<meta http-equiv="refresh" content="2">' in page
    assert '"edited": ["lib.py"]' in page
    assert "crg-live" in page and "node-shape" in page


def test_nothing_changed_writes_a_waiting_page(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "lib.py"], check=True)
    out = tmp_path / "live.html"
    render_live(repo, out)
    assert "Nothing changed" in out.read_text(encoding="utf-8")
    assert not activity_path(repo).exists()


def test_cli_once(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "page.html"
    result = subprocess.run(
        [sys.executable, "-m", "code_review_graph", "live", "--repo", str(repo),
         "--out", str(out), "--once"],
        capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"}, cwd=str(repo),
    )
    assert result.returncode == 0, result.stderr
    assert "1 edited" in result.stdout and out.is_file()
