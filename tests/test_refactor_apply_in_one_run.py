"""``refactor rename --apply`` previews and writes the rename in one run.

A preview lives only in the process that made it, so a later ``apply`` from
another CLI run could never find it. ``--apply`` refuses the modes that make
no edits.

From PR #3.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import code_review_graph.tools  # noqa: F401 - exposes lazy patch targets
from code_review_graph import cli
from code_review_graph.toon import decode


def _repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "graph.db").touch()
    monkeypatch.setenv("CRG_DATA_DIR", str(data_dir))
    return repo


def test_apply_writes_the_preview_it_just_made(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path, monkeypatch)
    argv = [
        "code-review-graph", "refactor", "rename", "--old-name", "charge",
        "--new-name", "bill", "--repo", str(repo), "--apply",
    ]
    preview = {"status": "ok", "refactor_id": "r1", "edits": [{"file": "a.py"}]}
    applied = {
        "status": "ok", "applied": True, "files_modified": ["a.py", "b.py"], "edits_applied": 3,
    }
    with patch.object(sys, "argv", argv), \
            patch("code_review_graph.tools.refactor_func", return_value=preview), \
            patch("code_review_graph.tools.apply_refactor_func", return_value=applied) as apply:
        cli.main()

    apply.assert_called_once()
    assert apply.call_args.kwargs["refactor_id"] == "r1"
    out = decode(capsys.readouterr().out)
    assert out["summary"] == "Renamed charge -> bill: 3 edit(s) in 2 file(s)."
    assert out["files_modified"] == ["a.py", "b.py"]


def test_apply_is_refused_outside_rename(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path, monkeypatch)
    argv = ["code-review-graph", "refactor", "dead_code", "--repo", str(repo), "--apply"]
    with patch.object(sys, "argv", argv), \
            patch("code_review_graph.tools.refactor_func") as refactor:
        with pytest.raises(SystemExit) as exc:
            cli.main()

    assert exc.value.code == 2
    assert "--apply works with rename only" in capsys.readouterr().err
    refactor.assert_not_called()
