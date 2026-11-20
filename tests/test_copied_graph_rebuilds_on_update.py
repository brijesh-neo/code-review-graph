"""A graph copied from another checkout updates incrementally there.

Relative storage lets a ``graph.db`` copied into a worktree, fork or clone
answer with that checkout's paths. ``extra`` JSON (``claimed_endpoint``,
``receiver_module``, ``ambiguous_targets``) is stored relative too since
schema v17, so the first update in the copy is incremental. A graph whose
``extra`` is still absolute (written before v17, copied before it was opened
where it was built) still names its old checkout, so that one rebuilds.
"""

import json
import shutil
import sqlite3
import subprocess

from code_review_graph.graph import GraphStore
from code_review_graph.incremental import full_build, incremental_update
from code_review_graph.relpaths import EXTRA_MODE_KEY, MOVED_FROM_KEY


def _git(repo, *args):
    subprocess.run(
        ["git", "-c", "user.email=t@example.com", "-c", "user.name=T", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _built(root, *, absolute_extra=False):
    root.mkdir(parents=True)
    (root / "m.py").write_text("def helper():\n    return 1\n")
    (root / "b.py").write_text("from m import helper\n\ndef run():\n    return helper()\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "initial")
    db = root / ".code-review-graph" / "graph.db"
    store = GraphStore(db)
    try:
        full_build(root, store)
        # What a resolver leaves behind: a path to the build's own tree in extra.
        store._conn.execute(
            "INSERT INTO edges (kind, source_qualified, target_qualified, file_path, "
            "line, extra, updated_at) VALUES (?, ?, ?, ?, ?, ?, 0)",
            ("CALLS", f"{root}/b.py::run", "helper", f"{root}/b.py", 4,
             json.dumps({"claimed_endpoint": f"{root}/m.py::helper"})),
        )
        store._conn.commit()
    finally:
        store.close()
    if absolute_extra:
        # How a v16 release left it: absolute extra, no relpath_extra mark.
        conn = sqlite3.connect(db)
        conn.execute(
            "UPDATE edges SET extra = ? WHERE extra LIKE '%claimed_endpoint%'",
            (json.dumps({"claimed_endpoint": f"{root}/m.py::helper"}),),
        )
        conn.execute("DELETE FROM metadata WHERE key = ?", (EXTRA_MODE_KEY,))
        conn.commit()
        conn.close()
    return root


def _raw_extras(db):
    conn = sqlite3.connect(db)
    try:
        return [r[0] for r in conn.execute("SELECT extra FROM edges") if r[0]]
    finally:
        conn.close()


def _copy(tmp_path, origin):
    copy = tmp_path / "b" / "repo"
    shutil.copytree(origin, copy, symlinks=True)
    return copy.resolve()


def test_the_first_update_of_a_copied_graph_is_incremental(tmp_path):
    origin = _built(tmp_path / "a" / "repo").resolve()
    copy = _copy(tmp_path, origin)
    db = copy / ".code-review-graph" / "graph.db"

    store = GraphStore(db)
    try:
        assert not store.get_metadata(MOVED_FROM_KEY)
        # The planted edge reads back against the copy.
        claimed = [
            json.loads(r[0])["claimed_endpoint"] for r in store._conn.execute(
                "SELECT extra FROM edges WHERE extra LIKE '%claimed_endpoint%'",
            )
        ]
        assert claimed == [f"{copy}/m.py::helper"]
        (copy / "b.py").write_text(
            "from m import helper\n\ndef run():\n    return helper() + 1\n",
        )
        result = incremental_update(copy, store, changed_files=["b.py"])
        assert "identity_rebuild" not in result
        assert result["files_updated"] == 1
        assert store.get_metadata("last_build_type") == "incremental"
        files = {r[0] for r in store._conn.execute(
            "SELECT file_path FROM nodes WHERE kind = 'File'",
        )}
        assert files == {f"{copy}/b.py", f"{copy}/m.py"}
    finally:
        store.close()
    raw = _raw_extras(db)
    assert raw and not any(str(origin) in extra for extra in raw)
    assert not any(str(copy) in extra for extra in raw)


def test_a_graph_updated_where_it_was_built_is_not_rebuilt(tmp_path):
    origin = _built(tmp_path / "repo").resolve()
    store = GraphStore(origin / ".code-review-graph" / "graph.db")
    try:
        assert not store.get_metadata(MOVED_FROM_KEY)
        result = incremental_update(origin, store, changed_files=[])
        assert "identity_rebuild" not in result
        # The planted edge survives: nothing was rebuilt.
        assert any(
            "claimed_endpoint" in r[0]
            for r in store._conn.execute("SELECT extra FROM edges") if r[0]
        )
    finally:
        store.close()


def test_a_copied_graph_with_absolute_extra_still_rebuilds(tmp_path):
    origin = _built(tmp_path / "a" / "repo", absolute_extra=True).resolve()
    copy = _copy(tmp_path, origin)

    store = GraphStore(copy / ".code-review-graph" / "graph.db")
    try:
        assert store.get_metadata(MOVED_FROM_KEY) == str(origin)
        assert store.get_metadata(EXTRA_MODE_KEY) != "relative"
        result = incremental_update(copy, store, changed_files=[])
        assert result.get("identity_rebuild") is True
        assert store.get_metadata("last_build_type") == "full"
        assert not store.get_metadata(MOVED_FROM_KEY)
        assert store.get_metadata(EXTRA_MODE_KEY) == "relative"

        # Once rebuilt it is this checkout's graph: updates are incremental.
        again = incremental_update(copy, store, changed_files=[])
        assert "identity_rebuild" not in again
    finally:
        store.close()
    assert not any(str(origin) in extra for extra in _raw_extras(
        copy / ".code-review-graph" / "graph.db",
    ))
