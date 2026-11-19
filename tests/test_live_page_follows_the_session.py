"""``visualize --live`` draws the files a session edited, read and left unchecked."""

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


def _stamp(out: Path) -> str:
    text = (out.parent / (out.name + ".stamp.js")).read_text(encoding="utf-8")
    return text.split('"')[1]


def test_page_colours_the_sets_and_reloads_only_on_change(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    out = tmp_path / "live.html"
    render_live(repo, out, interval=2)
    page = out.read_text(encoding="utf-8")
    assert "http-equiv" not in page  # a timed reload would restart the layout
    assert '"edited": ["lib.py"]' in page
    assert "crg-live" in page and "node-shape" in page
    first = _stamp(out)
    assert f'var mine = "{first}"' in page and "setInterval(check, 2000)" in page

    render_live(repo, out, interval=2)
    assert _stamp(out) == first  # nothing changed: the open page stays

    record_tool_use(repo, {"tool_name": "Read", "session_id": "s1",
                           "tool_input": {"file_path": str(repo / "app.py")}})
    render_live(repo, out, interval=2)
    assert _stamp(out) != first


def test_nothing_changed_writes_a_waiting_page(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", "lib.py"], check=True)
    out = tmp_path / "live.html"
    render_live(repo, out)
    assert "Nothing changed" in out.read_text(encoding="utf-8")
    assert not activity_path(repo).exists()


def _visualize(repo: Path, *flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "code_review_graph", "visualize", "--repo", str(repo), *flags],
        capture_output=True, text=True, env={**os.environ, "PYTHONUTF8": "1"}, cwd=str(repo),
    )


def test_visualize_live_once_draws_graph_html(tmp_path: Path) -> None:
    """The live view is ``visualize --live``: the same graph.html, no second page."""
    repo = _repo(tmp_path)
    result = _visualize(repo, "--live", "--once")
    assert result.returncode == 0, result.stderr
    assert "1 edited" in result.stdout
    page = repo / ".code-review-graph" / "graph.html"
    assert "crg-live" in page.read_text(encoding="utf-8")
    assert not (repo / ".code-review-graph" / "live.html").exists()


def test_live_flags_that_cannot_apply_are_usage_errors(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    for flags, message in (
        (("--once",), "--once requires --live"),
        (("--session", "s1"), "--session requires --live"),
        (("--live", "--seed-file", "lib.py"), "drop the seed flags"),
        (("--live", "--format", "json"), "no effect on --format json"),
        (("--live", "--mode", "community"), "--mode community has no effect with --live"),
    ):
        result = _visualize(repo, *flags)
        assert result.returncode == 2, (flags, result.stderr)
        assert message in result.stderr, (flags, result.stderr)


def test_the_live_command_is_gone(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "code_review_graph", "live", "--once"],
        capture_output=True, text=True, cwd=str(tmp_path),
    )
    assert result.returncode != 0


def test_page_carries_d3_itself_so_it_opens_from_disk(tmp_path: Path) -> None:
    """Chrome blocks a file:// script with an integrity attribute, so the
    page must not depend on loading D3 by src."""
    from code_review_graph.visualization import D3_LOCAL_FILENAME, _d3_script_tags

    repo = _repo(tmp_path)
    out = tmp_path / "live.html"
    render_live(repo, out)
    page = out.read_text(encoding="utf-8")
    assert _d3_script_tags() not in page
    assert f'src="{D3_LOCAL_FILENAME}"' not in page
    assert "document.write(" not in page
