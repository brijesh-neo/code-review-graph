"""An update saves the edges of unchanged dependents in batched transactions.

A change to a widely imported file re-saves the edges of every dependent whose
bytes did not change. One transaction per dependent made those commits most
of the cost of writing them; they now share a transaction per batch, and the
edges stored are the same.
"""

import subprocess

from code_review_graph import incremental
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, incremental_update

_DEPENDENTS = 20


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _repo(root):
    root.mkdir()
    (root / "core.py").write_text("def helper():\n    return 1\n")
    for i in range(_DEPENDENTS):
        (root / f"use{i}.py").write_text(
            f"from core import helper\n\ndef run{i}():\n    return helper()\n"
        )
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "initial")
    return root


def _edges(store):
    return sorted(
        tuple(row) for row in store._conn.execute(
            "SELECT kind, source_qualified, target_qualified, file_path, line, extra, "
            "confidence, confidence_tier, target_resolution FROM edges",
        )
    )


def _update(tmp_path, monkeypatch, name, batch):
    monkeypatch.setenv("CRG_SERIAL_PARSE", "1")
    # CRG_FAST_ENGINE leaves these dependents' rows alone (a body change names
    # nothing they use), so the batching is pinned on the default path.
    monkeypatch.delenv("CRG_FAST_ENGINE", raising=False)
    monkeypatch.setattr(incremental, "_EDGE_WRITE_BATCH", batch)
    repo = _repo(tmp_path / name).resolve()
    store = GraphStore(repo / ".code-review-graph" / "graph.db")
    full_build(repo, store)
    (repo / "core.py").write_text("def helper():\n    return 2\n")
    calls = []
    edges_only = store.store_files_edges
    monkeypatch.setattr(
        store, "store_files_edges", lambda batch: (calls.append(len(batch)), edges_only(batch))[1],
    )
    incremental_update(repo, store, changed_files=["core.py"])
    edges = [
        tuple(str(v).replace(str(repo), "<root>") for v in row) for row in _edges(store)
    ]
    store.close()
    return calls, edges


def test_dependent_edges_share_transactions(tmp_path, monkeypatch):
    calls, _ = _update(tmp_path, monkeypatch, "repo", batch=8)
    assert sum(calls) == _DEPENDENTS
    # 20 dependents in batches of 8: three transactions, not twenty.
    assert sorted(calls, reverse=True)[:3] == [8, 8, 4]


def test_batching_stores_the_same_edges(tmp_path, monkeypatch):
    _, one_at_a_time = _update(tmp_path, monkeypatch, "single", batch=1)
    _, batched = _update(tmp_path, monkeypatch, "batched", batch=64)
    assert batched == one_at_a_time
    assert any(row[0] == "CALLS" and row[2].endswith("core.py::helper") for row in batched)
