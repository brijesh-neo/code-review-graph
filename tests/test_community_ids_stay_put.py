"""An update that finds the same communities keeps their ids.

Storing communities used to delete the table and insert every row again, so
each update gave each community a new id and rewrote every node's
``community_id`` (and its index and search rows) even when nothing moved.
The table's ids are now reused in ascending order, by the native engine and
by the Python store alike, so both still write the same rows. Pinned for
PR #13 on the fork.
"""

from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

import pytest

from code_review_graph.communities import store_communities
from code_review_graph.graph import GraphStore
from code_review_graph.tools.build import build_or_update_graph


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@example.invalid", "-c", "user.name=T", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _snapshot(db: Path) -> tuple[list[tuple], dict[str, int | None]]:
    # Names are left out: a tie between keywords is broken by node row order,
    # which re-storing the edited file changes.
    conn = sqlite3.connect(db)
    try:
        communities = conn.execute(
            "SELECT id, size FROM communities ORDER BY id"
        ).fetchall()
        members = dict(conn.execute("SELECT qualified_name, community_id FROM nodes"))
    finally:
        conn.close()
    return communities, members


@pytest.mark.parametrize("flag", [False, True], ids=["legacy", "fast"])
def test_an_update_with_the_same_communities_keeps_every_id(
    flag: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    for package in ("alpha", "beta"):
        (repo / package).mkdir(parents=True)
        for index in range(3):
            (repo / package / f"m{index}.py").write_text(
                f"from {package}.m{(index + 1) % 3} import f{(index + 1) % 3}\n\n\n"
                f"def f{index}():\n    return f{(index + 1) % 3}\n\n\n"
                f"def g{index}():\n    return f{index}()\n"
            )
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "init")
    data = tmp_path / "data"
    monkeypatch.setenv("CRG_DATA_DIR", str(data))
    if flag:
        monkeypatch.setenv("CRG_FAST_ENGINE", "1")
    else:
        monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    assert build_or_update_graph(full_rebuild=True, repo_root=str(repo))["status"] == "ok"
    before = _snapshot(data / "graph.db")
    assert before[0], "the fixture must form at least one community"

    edited = repo / "alpha" / "m0.py"
    edited.write_text(edited.read_text() + "\n# edited\n")
    assert build_or_update_graph(repo_root=str(repo))["status"] == "ok"

    assert _snapshot(data / "graph.db") == before


def test_storing_fewer_communities_keeps_the_first_ids_and_drops_the_rest(
    tmp_path: Path,
) -> None:
    store = GraphStore(tmp_path / "graph.db")
    try:
        def community(name: str) -> dict:
            return {"name": name, "size": 0, "members": [], "detector": "file"}

        store_communities(store, [community("a"), community("b"), community("c")])
        first = store._conn.execute("SELECT id FROM communities ORDER BY id").fetchall()
        store_communities(store, [community("x"), community("y")])
        kept = [tuple(row) for row in store._conn.execute(
            "SELECT id, name FROM communities ORDER BY id"
        )]
        assert kept == [(first[0][0], "x"), (first[1][0], "y")]
        store_communities(store, [community(name) for name in "pqrs"])
        grown = [row[0] for row in store._conn.execute("SELECT id FROM communities")]
        assert grown[:2] == [first[0][0], first[1][0]]
        assert grown == sorted(grown) and len(set(grown)) == 4
        store_communities(store, [])
        assert store._conn.execute("SELECT COUNT(*) FROM communities").fetchone()[0] == 0
    finally:
        store.close()
