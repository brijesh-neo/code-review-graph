"""Flows of importers an update leaves as stored are traced again only if reached.

With ``CRG_FAST_ENGINE`` an update keeps the rows of importers whose edges
name no changed definition. Their flows trace as stored unless a write of the
update reaches them: an edge they hold or point at, or a node that appeared,
disappeared or changed across one of their edges. The flow scope names those
files, and the other kept importers drop out of the files whose flows are
traced again. These cases pin that the flows equal those of the legacy update
and of a full rebuild.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Callable

import pytest

from code_review_graph.tools import build as build_module
from code_review_graph.tools.build import build_or_update_graph

from .test_fast_update_matches_full_reparse import EDITS, FILES, _git

FLOW_FILES = {
    **FILES,
    "e.py": (
        "from lib import Box\n\n\n"
        "def own():\n"
        "    return 5\n\n\n"
        "def use_e():\n"
        "    own()\n"
        "    return Box\n"
    ),
}


@pytest.fixture
def flow_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "base"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    for name, text in FLOW_FILES.items():
        (repo / name).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    return repo


def _flows(db: Path) -> list[tuple]:
    conn = sqlite3.connect(db)
    try:
        qn = dict(conn.execute("SELECT id, qualified_name FROM nodes"))
        rows = []
        for flow_id, name, entry, depth, nodes, files, criticality in conn.execute(
            "SELECT id, name, entry_point_id, depth, node_count, file_count, criticality "
            "FROM flows"
        ):
            path = tuple(
                qn.get(node, node) for (node,) in conn.execute(
                    "SELECT node_id FROM flow_memberships WHERE flow_id = ? "
                    "ORDER BY position", (flow_id,),
                )
            )
            rows.append((name, qn.get(entry, entry), depth, nodes, files, criticality, path))
        return sorted(rows)
    finally:
        conn.close()


def _graph(
    flow_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    label: str, edit: Callable[[Path], None], mode: str,
) -> Path:
    repo = tmp_path / label
    shutil.copytree(flow_repo, repo)
    data = tmp_path / f"{label}-data"
    monkeypatch.setenv("CRG_DATA_DIR", str(data))
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    if mode != "rebuild":
        assert build_or_update_graph(full_rebuild=True, repo_root=str(repo))["status"] == "ok"
    edit(repo)
    if mode == "fast":
        monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    result = build_or_update_graph(full_rebuild=mode == "rebuild", repo_root=str(repo))
    assert result["status"] == "ok", result
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    return data / "graph.db"


@pytest.mark.parametrize("case", sorted(EDITS))
def test_flows_after_a_fast_update_equal_legacy_and_rebuild(
    case: str, flow_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    narrowed: list[tuple[list[str], list[str]]] = []
    original = build_module._files_flows_reach

    def spy(store, changed, kept, root):
        reached = original(store, changed, kept, root)
        narrowed.append((list(changed), reached))
        return reached

    monkeypatch.setattr(build_module, "_files_flows_reach", spy)
    legacy = _flows(_graph(flow_repo, tmp_path, monkeypatch, "off", EDITS[case], "legacy"))
    fast = _flows(_graph(flow_repo, tmp_path, monkeypatch, "on", EDITS[case], "fast"))
    rebuilt = _flows(_graph(flow_repo, tmp_path, monkeypatch, "re", EDITS[case], "rebuild"))
    assert fast == legacy
    assert fast == rebuilt
    if case == "body_changed":
        # No name changed, so every importer kept its rows, and e.py, whose
        # flow never enters lib.py, is not traced again.
        changed, reached = narrowed[-1]
        assert "e.py" in changed
        assert "e.py" not in reached


def test_a_kept_importer_an_edge_now_points_at_is_traced_again(
    flow_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def call_into_e(repo: Path) -> None:
        # lib.py now calls e.own, so own is no longer an entry point.
        lib = (repo / "lib.py").read_text()
        (repo / "lib.py").write_text(
            "from e import own\n" + lib + "\n\ndef helper2():\n    return own()\n"
        )

    fast = _flows(_graph(flow_repo, tmp_path, monkeypatch, "on", call_into_e, "fast"))
    rebuilt = _flows(_graph(flow_repo, tmp_path, monkeypatch, "re", call_into_e, "rebuild"))
    assert fast == rebuilt


def test_without_a_scope_every_file_is_traced(tmp_path: Path) -> None:
    class NoScope:
        def flow_scope_hits(self, paths):
            return None

    changed = ["lib.py", "a.py", "e.py"]
    assert build_module._files_flows_reach(NoScope(), changed, ["e.py"], str(tmp_path)) == (
        changed
    )
    assert build_module._files_flows_reach(NoScope(), changed, [], str(tmp_path)) == changed
