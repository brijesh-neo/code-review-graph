"""CLI wrappers for graph tools reconciled from PR #95."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import code_review_graph.tools  # noqa: F401 - exposes lazy patch targets
from code_review_graph import cli
from code_review_graph.graph import GraphStore
from code_review_graph.parser import NodeInfo
from code_review_graph.toon import decode

_LEGACY_GRAPH_COMMANDS = [
    ["query", "file_summary", "app.py"],
    ["search", "handler"],
    ["flows"],
    ["architecture"],
]


@pytest.mark.parametrize(
    ("arguments", "tool_name", "expected"),
    [
        (
            ["query", "callers_of", "target"],
            "query_graph",
            {"pattern": "callers_of", "target": "target", "max_results": 100},
        ),
        (
            ["query", "tests_for", "target", "--max-results", "500"],
            "query_graph",
            {"pattern": "tests_for", "target": "target", "max_results": 500},
        ),
        (
            ["impact", "--files", "a.py", "b.py", "--depth", "3", "--max-results", "20"],
            "get_impact_radius",
            {
                "changed_files": ["a.py", "b.py"],
                "max_depth": 3,
                "max_results": 20,
                "base": "HEAD~1",
                "detail_level": "standard",
                "max_files": 25,
            },
        ),
        (
            ["search", "login", "--kind", "Function", "--limit", "7"],
            "semantic_search_nodes",
            {"query": "login", "kind": "Function", "limit": 7},
        ),
        (
            ["flows", "--sort", "depth", "--limit", "9", "--kind", "Function"],
            "list_flows",
            {"sort_by": "depth", "limit": 9, "kind": "Function"},
        ),
        (
            ["flow", "--id", "7", "--source"],
            "get_flow",
            {"flow_id": 7, "flow_name": None, "include_source": True},
        ),
        (
            ["communities", "--sort", "cohesion", "--min-size", "3"],
            "list_communities_func",
            {"sort_by": "cohesion", "min_size": 3},
        ),
        (
            ["community", "--name", "parser", "--members"],
            "get_community_func",
            {
                "community_name": "parser",
                "community_id": None,
                "include_members": True,
            },
        ),
        (
            ["architecture", "--detail-level", "standard"],
            "get_architecture_overview_func",
            {"detail_level": "standard"},
        ),
        (
            ["large-functions", "--min-lines", "80", "--kind", "Class", "--limit", "4"],
            "find_large_functions",
            {
                "min_lines": 80,
                "kind": "Class",
                "file_path_pattern": None,
                "limit": 4,
            },
        ),
        (
            ["refactor", "dead_code", "--kind", "Function", "--path", "src/"],
            "refactor_func",
            {
                "mode": "dead_code",
                "old_name": None,
                "new_name": None,
                "kind": "Function",
                "file_pattern": "src/",
            },
        ),
        (
            ["context", "review auth", "--files", "a.py", "--base", "main"],
            "get_minimal_context",
            {"task": "review auth", "changed_files": ["a.py"], "base": "main"},
        ),
        (
            ["review-context", "--no-source", "--max-lines", "9", "--detail-level", "minimal"],
            "get_review_context",
            {
                "changed_files": None,
                "max_depth": 2,
                "include_source": False,
                "max_lines_per_file": 9,
                "base": "HEAD~1",
                "detail_level": "minimal",
                "max_results": 50,
                "max_files": 25,
            },
        ),
        (
            ["affected-flows", "--files", "a.py", "--limit", "3"],
            "get_affected_flows_func",
            {
                "changed_files": ["a.py"],
                "base": "HEAD~1",
                "detail_level": "standard",
                "max_flows": 3,
            },
        ),
        (
            ["traverse", "login", "--mode", "dfs", "--depth", "2", "--token-budget", "500"],
            "traverse_graph_func",
            {"query": "login", "mode": "dfs", "depth": 2, "token_budget": 500},
        ),
        (
            ["hubs", "--limit", "4"],
            "get_hub_nodes_func",
            {"top_n": 4, "detail_level": "standard"},
        ),
        (
            ["bridges", "--detail-level", "minimal"],
            "get_bridge_nodes_func",
            {"top_n": 10, "detail_level": "minimal"},
        ),
        (
            ["gaps", "--limit", "5"],
            "get_knowledge_gaps_func",
            {"max_per_category": 5, "detail_level": "standard"},
        ),
        (
            ["surprises"],
            "get_surprising_connections_func",
            {"top_n": 15, "detail_level": "standard"},
        ),
        (["questions"], "get_suggested_questions_func", {}),
        (
            ["wiki-page", "parser", "--max-chars", "100"],
            "get_wiki_page_func",
            {"community_name": "parser", "max_chars": 100},
        ),
    ],
)
def test_tool_command_forwards_typed_arguments(
    arguments, tool_name, expected, tmp_path, monkeypatch, capsys,
):
    repo = tmp_path / "repo"
    nested = repo / "src" / "nested"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "graph.db").touch()
    monkeypatch.setenv("CRG_DATA_DIR", str(data_dir))
    argv = ["code-review-graph", *arguments, "--repo", str(nested)]
    result = {"status": "ok", "tool": tool_name}

    with patch.object(sys, "argv", argv):
        with patch(f"code_review_graph.tools.{tool_name}", return_value=result) as tool:
            cli.main()

    assert decode(capsys.readouterr().out) == result
    tool.assert_called_once_with(repo_root=str(repo), **expected)


@pytest.mark.parametrize(
    "arguments",
    [
        ["flow"],
        ["flow", "--id", "1", "--name", "duplicate"],
        ["community"],
        ["community", "--id", "1", "--name", "duplicate"],
        ["refactor", "rename", "--old-name", "only-old"],
        ["impact", "--depth", "-1"],
        ["search", "query", "--limit", "0"],
    ],
)
def test_tool_commands_reject_invalid_or_ambiguous_arguments(arguments):
    with patch.object(sys, "argv", ["code-review-graph", *arguments]):
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
    assert exc_info.value.code == 2


def test_tool_command_missing_graph_exits_nonzero(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    monkeypatch.setenv("CRG_DATA_DIR", str(tmp_path / "missing"))

    with patch.object(
        sys,
        "argv",
        ["code-review-graph", "query", "callers_of", "target", "--repo", str(repo)],
    ):
        with pytest.raises(SystemExit) as exc_info:
            cli.main()

    assert exc_info.value.code == 1
    assert "No graph found" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", _LEGACY_GRAPH_COMMANDS)
def test_tool_commands_migrate_and_read_legacy_graph(
    arguments, tmp_path, monkeypatch, capsys,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    legacy_db = repo / ".code-review-graph.db"
    with GraphStore(legacy_db) as store:
        store.upsert_node(NodeInfo(
            kind="Function", name="handler", file_path="app.py",
            line_start=1, line_end=2, language="python",
        ), file_hash="fixture")
        store.commit()
    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["code-review-graph", *arguments, "--repo", str(repo)])

    cli.main()

    result = decode(capsys.readouterr().out)
    assert result["status"] == "ok", result
    assert not legacy_db.exists()
    with GraphStore(repo / ".code-review-graph" / "graph.db") as store:
        assert store.get_node("app.py::handler") is not None


@pytest.mark.parametrize("arguments", _LEGACY_GRAPH_COMMANDS)
@pytest.mark.parametrize("external_source", [None, "env", "registry"])
def test_tool_commands_missing_graph_preserve_state(
    arguments, external_source, tmp_path, monkeypatch, capsys,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    data_dir = repo / ".code-review-graph"
    legacy_db = repo / ".code-review-graph.db"
    monkeypatch.delenv("CRG_DATA_DIR", raising=False)
    if external_source:
        # An external location must not silently consume an unrelated legacy graph.
        with GraphStore(legacy_db):
            pass
        legacy_bytes = legacy_db.read_bytes()
        data_dir = tmp_path / "external-data"
        if external_source == "env":
            monkeypatch.setenv("CRG_DATA_DIR", str(data_dir))
        else:
            from code_review_graph.registry import Registry

            Registry().register(str(repo), data_dir=str(data_dir))
    monkeypatch.setattr(sys, "argv", ["code-review-graph", *arguments, "--repo", str(repo)])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 1
    assert "No graph found" in capsys.readouterr().err
    assert not data_dir.exists()
    assert not (repo / ".code-review-graph").exists()
    if external_source:
        assert legacy_db.read_bytes() == legacy_bytes
