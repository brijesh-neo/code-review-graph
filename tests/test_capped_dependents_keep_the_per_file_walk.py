"""A capped dependent walk returns the files the one-file-at-a-time walk does.

``find_dependents`` asks the store once per hop for every file in the
frontier. That gives the same set as one query per file, but the set holds its
members in another order, and a walk over ``_MAX_DEPENDENT_FILES`` returns the
first files of that order. A walk that reaches the cap is therefore run again
one file at a time. These cases pin that the capped list, order included,
equals that walk's, and that an uncapped walk still uses the batched query.
The fastapi ``delete_file`` case of the incremental fidelity eval reaches the
cap (582 dependents); it came up in PR #13.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_review_graph import incremental
from code_review_graph.graph import GraphStore
from code_review_graph.incremental import find_dependents
from code_review_graph.parser import EdgeInfo, NodeInfo


def _store(tmp_path: Path, leaves_per_middle: int) -> tuple[GraphStore, str]:
    """A hub imported by three files, each imported by *leaves_per_middle*."""
    root = tmp_path.resolve()
    store = GraphStore(root / "graph.db")

    def add_file(path: str, name: str) -> None:
        store.upsert_node(NodeInfo(
            kind="File", name=path, file_path=path,
            line_start=1, line_end=10, language="python",
        ))
        store.upsert_node(NodeInfo(
            kind="Function", name=name, file_path=path,
            line_start=2, line_end=8, language="python",
        ))

    hub = (root / "hub.py").as_posix()
    add_file(hub, "hub_func")
    for m in range(3):
        middle = (root / f"mid{m}.py").as_posix()
        add_file(middle, f"mid_{m}")
        store.upsert_edge(EdgeInfo(
            kind="IMPORTS_FROM", source=f"{middle}::mid_{m}",
            target=f"{hub}::hub_func", file_path=middle, line=1,
        ))
        for leaf_index in range(leaves_per_middle):
            leaf = (root / f"leaf{m}_{leaf_index}.py").as_posix()
            add_file(leaf, f"leaf_{m}_{leaf_index}")
            store.upsert_edge(EdgeInfo(
                kind="IMPORTS_FROM", source=f"{leaf}::leaf_{m}_{leaf_index}",
                target=f"{middle}::mid_{m}", file_path=leaf, line=1,
            ))
    store.commit()
    return store, hub


def _batched_calls(store: GraphStore, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = store.get_direct_dependent_files_of_all

    def spy(file_paths):
        calls.append(len(file_paths))
        return original(file_paths)

    monkeypatch.setattr(store, "get_direct_dependent_files_of_all", spy)
    return calls


def test_a_capped_walk_returns_the_per_file_walks_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, hub = _store(tmp_path, leaves_per_middle=200)
    try:
        calls = _batched_calls(store, monkeypatch)
        capped = find_dependents(store, hub)
        per_file = find_dependents(store, hub, _batched=False)
        assert capped.truncated and per_file.truncated
        assert len(capped) == incremental._MAX_DEPENDENT_FILES
        assert list(capped) == list(per_file)
        # The batched hop ran, reached the cap, and the walk was redone.
        assert calls == [3]
    finally:
        store.close()


def test_an_uncapped_walk_keeps_the_batched_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, hub = _store(tmp_path, leaves_per_middle=20)
    try:
        calls = _batched_calls(store, monkeypatch)
        batched = find_dependents(store, hub)
        per_file = find_dependents(store, hub, _batched=False)
        assert not batched.truncated
        assert sorted(batched) == sorted(per_file)
        assert len(batched) == 3 + 60
        assert calls == [3]
    finally:
        store.close()
