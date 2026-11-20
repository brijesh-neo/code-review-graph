"""Build speed-ups that must leave the graph exactly as it was.

- full_build writes parsed files in batches, one transaction each.
- PathCodec.strip skips values that cannot hold a root.
- _extract_from_tree reuses its per-language node-type sets.
- _extract_value_references names the caller only for the nodes it acts on.
- Python walks skip statements that cannot hold a declaration or method scope.
- full_build and incremental_update run with a larger page cache, put back afterwards.

From PR #3.
"""

import sqlite3

import pytest

from code_review_graph import incremental
from code_review_graph.graph import BUILD_CACHE_KIB, GraphStore
from code_review_graph.incremental import full_build
from code_review_graph.parser import CodeParser
from code_review_graph.relpaths import PathCodec


def _write_repo(root, count):
    for i in range(count):
        (root / f"m{i}.py").write_text(
            f"from m{(i + 1) % count} import f{(i + 1) % count}\n"
            f"def f{i}():\n    return f{(i + 1) % count}()\n"
            f"HANDLERS = [f{i}]\n"
        )


def _graph_rows(db):
    conn = sqlite3.connect(db)
    try:
        nodes = sorted(conn.execute(
            "SELECT kind, name, qualified_name, file_path, line_start FROM nodes",
        ).fetchall())
        edges = sorted(conn.execute(
            "SELECT kind, source_qualified, target_qualified, file_path, line FROM edges",
        ).fetchall())
    finally:
        conn.close()
    return nodes, edges


def _build(root, data):
    data.mkdir()
    with GraphStore(data / "graph.db") as store:
        result = full_build(root, store)
    return result, _graph_rows(data / "graph.db")


@pytest.mark.parametrize("serial", ["1", "0"])
def test_batched_writes_give_the_same_graph_as_one_file_per_commit(
    tmp_path, monkeypatch, serial,
):
    monkeypatch.setenv("CRG_SERIAL_PARSE", serial)
    monkeypatch.setenv("CRG_PARSE_EXECUTOR", "thread")
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo(repo, 12)

    batches = []
    real = GraphStore.store_file_batch

    def recording(self, batch):
        batches.append(len(batch))
        return real(self, batch)

    monkeypatch.setattr(GraphStore, "store_file_batch", recording)
    monkeypatch.setattr(incremental, "STORE_BATCH_FILES", 5)
    result, batched = _build(repo, tmp_path / "batched")
    assert result["errors"] == []
    assert batches == [5, 5, 2]

    monkeypatch.setattr(incremental, "STORE_BATCH_FILES", 1)
    _, single = _build(repo, tmp_path / "single")
    assert batched == single
    assert len(batched[0]) > 12 and len(batched[1]) > 12


def test_a_failed_batch_write_stops_the_build_before_it_is_marked_done(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("CRG_SERIAL_PARSE", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo(repo, 3)

    def locked(self, batch):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(GraphStore, "store_file_batch", locked)
    data = tmp_path / "data"
    data.mkdir()
    with GraphStore(data / "graph.db") as store:
        with pytest.raises(sqlite3.OperationalError):
            full_build(repo, store)
        assert store.get_metadata("last_updated") is None


def _slow_strip(codec, value):
    """PathCodec.strip without the first-character shortcut."""
    stripped = codec.strip_root(value)
    if stripped is value:
        body = value.lstrip("!")
        if body == "." or body.startswith(("./", "../")):
            return "!" + value
    return stripped


def test_strip_shortcut_matches_the_full_check(tmp_path):
    root = tmp_path.as_posix()
    codec = PathCodec(root)
    values = [
        "", ".", "..", "./x", "../x", "!./x", "!!./x", "!", "!x",
        root, root + "/", root + "/a.py", root + "/a.py::f", root + "x",
        "/elsewhere/a.py", "CALLS", "helper", "Redis::get", '{"k": "v"}',
        "C:/repo/a.py", "~/x", " ./x",
    ]
    for value in values:
        assert codec.strip(value) == _slow_strip(codec, value), value


def test_node_type_sets_are_built_once_per_language(tmp_path):
    parser = CodeParser()
    first = parser._node_type_sets("python")
    assert parser._node_type_sets("python") is first
    assert "class_definition" in first[0]
    assert "function_definition" in first[1]
    assert parser._node_type_sets("no-such-language") == (
        frozenset(), frozenset(), frozenset(), frozenset(),
    )


def test_value_references_name_the_caller_only_for_nodes_they_act_on(
    tmp_path, monkeypatch,
):
    source = tmp_path / "app.py"
    source.write_text(
        "def target():\n    return 1\n"
        "def run(x):\n"
        "    total = x + 1\n"
        "    handlers = [target]\n"
        "    return total, handlers\n"
    )
    parser = CodeParser()
    seen = []
    real = CodeParser._extract_value_references

    def recording(self, child, node_type, *args, **kwargs):
        before = len(seen)
        real(self, child, node_type, *args, **kwargs)
        if len(seen) > before:
            assert node_type in CodeParser._VALUE_REFERENCE_TYPES, node_type

    real_qualify = CodeParser._qualify

    def qualify(self, *args):
        seen.append(args)
        return real_qualify(self, *args)

    monkeypatch.setattr(CodeParser, "_extract_value_references", recording)
    monkeypatch.setattr(CodeParser, "_qualify", qualify)
    _, edges = parser.parse_file(source)
    assert any(
        edge.kind == "REFERENCES"
        and edge.source == f"{source.as_posix()}::run"
        and edge.target.endswith("::target")
        for edge in edges
    )


def test_builds_raise_the_page_cache_and_put_it_back(tmp_path, monkeypatch):
    monkeypatch.setenv("CRG_SERIAL_PARSE", "1")
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_repo(repo, 3)
    seen = []
    with GraphStore(tmp_path / "graph.db") as store:
        before = store._conn.execute("PRAGMA cache_size").fetchone()[0]
        real_commit = store.commit

        def spy_commit():
            seen.append(store._conn.execute("PRAGMA cache_size").fetchone()[0])
            real_commit()

        monkeypatch.setattr(store, "commit", spy_commit)
        full_build(repo, store)
        incremental.incremental_update(repo, store, changed_files=["m0.py"])
        assert store._conn.execute("PRAGMA cache_size").fetchone()[0] == before
    assert seen and all(size == -BUILD_CACHE_KIB for size in seen)
