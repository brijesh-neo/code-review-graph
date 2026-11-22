"""Graph answers are printed and served as TOON.

- ``toon.encode`` writes lists of records as tables and ``toon.decode`` reads
  them back.
- CLI graph commands print TOON with repository-relative paths; ``--json``
  prints the full JSON with absolute paths, as before.
- ``cross-search`` and ``docs`` answer without a graph of the current
  repository.
- The MCP server returns every tool result as TOON text, with no output schema.
- A ``tests_for`` summary counts only the guesses among the rows it shows.

From PR #3.
"""

from __future__ import annotations

import asyncio
import json
import sys
from unittest.mock import patch

import pytest

import code_review_graph.tools  # noqa: F401 - exposes lazy patch targets
from code_review_graph import cli
from code_review_graph.toon import decode, encode

_RESULT = {
    "status": "ok",
    "summary": "Found 2 result(s): a, b",
    "results": [
        {"name": "run", "line": 12, "call_site": {"file": "app/main.py", "line": 3}},
        {"name": "-x", "line": None, "score": 0.25},
    ],
    "files": ["a.py", "b c.py", ""],
    "nested": {"empty": [], "obj": {}, "flag": True, "text": "true", "num": "12"},
    "mixed": [1, {"k": [1, 2]}, [3, 4], {}],
}


def test_records_become_a_table_that_reads_back():
    text = encode(_RESULT)

    assert "results[2]{name,line,call_site.file,call_site.line,score}:" in text
    assert "  run,12,app/main.py,3,null" in text
    assert '  "-x",null,null,null,0.25' in text
    assert 'files[3]: a.py,b c.py,""' in text
    decoded = decode(text)
    assert decoded["results"][0] == {**_RESULT["results"][0], "score": None}
    assert decoded["results"][1] == {**_RESULT["results"][1], "call_site": None}
    assert {k: v for k, v in decoded.items() if k != "results"} == {
        k: v for k, v in _RESULT.items() if k != "results"
    }


@pytest.mark.parametrize(
    "value",
    ["", " pad", "null", "1.5", "007", "a:b", 'say "hi"', "line\nbreak", "tab\t", "x,y"],
)
def test_strings_that_would_read_as_something_else_are_quoted(value):
    assert encode({"v": value}).startswith('v: "')
    assert decode(encode({"v": value})) == {"v": value}


def _graph_repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / ".git").mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "graph.db").touch()
    monkeypatch.setenv("CRG_DATA_DIR", str(data_dir))
    return repo.resolve()


def _run(argv, tool_name, result):
    with patch.object(sys, "argv", ["code-review-graph", *argv]):
        with patch(f"code_review_graph.tools.{tool_name}", return_value=result) as tool:
            cli.main()
    return tool


def test_toon_shows_repo_relative_paths_and_json_keeps_them_absolute(
    tmp_path, monkeypatch, capsys,
):
    repo = _graph_repo(tmp_path, monkeypatch)
    result = {"status": "ok", "hub_nodes": [{"file": f"{repo}/src/a.py", "total_degree": 3}]}
    monkeypatch.chdir(repo)

    _run(["hubs", "--repo", str(repo)], "get_hub_nodes_func", result)
    toon_out = capsys.readouterr().out
    _run(["hubs", "--repo", str(repo), "--json"], "get_hub_nodes_func", result)
    json_out = capsys.readouterr().out

    assert "hub_nodes[1]{file,total_degree}:\n  src/a.py,3" in toon_out
    assert json.loads(json_out) == result


def test_cross_search_and_docs_need_no_graph_here(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    tool = _run(
        ["cross-search", "login", "--kind", "Function", "--repos", "api"],
        "cross_repo_search_func",
        {"status": "ok", "total": 0},
    )
    tool.assert_called_once_with(
        query="login", kind="Function", limit=20, max_results=50, repos=["api"],
    )
    assert decode(capsys.readouterr().out) == {"status": "ok", "total": 0}

    tool = _run(["docs", "commands"], "get_docs_section", {"status": "ok", "content": "x"})
    tool.assert_called_once_with(section_name="commands", repo_root=None)
    assert decode(capsys.readouterr().out) == {"status": "ok", "content": "x"}


def test_new_commands_are_listed_in_the_banner(capsys):
    with patch.object(sys, "argv", ["code-review-graph"]):
        cli.main()
    out = capsys.readouterr().out
    for name in (
        "context", "review-context", "affected-flows", "traverse", "hubs", "bridges",
        "gaps", "surprises", "questions", "cross-search", "wiki-page", "docs",
    ):
        assert f"  {name} " in out


def test_mcp_tools_answer_in_toon_without_an_output_schema(tmp_path):
    from fastmcp import Client

    from code_review_graph.main import mcp

    async def go():
        async with Client(mcp) as client:
            tools = await client.list_tools()
            result = await client.call_tool(
                "get_docs_section_tool", {"section_name": "commands"}, raise_on_error=False,
            )
            return tools, result

    tools, result = asyncio.run(go())

    assert tools and all(tool.output_schema is None for tool in tools)
    assert result.structured_content is None
    text = result.content[0].text
    assert text.startswith("status: ")
    assert decode(text)["status"] in ("ok", "not_found")


def test_tests_for_summary_counts_only_the_guesses_it_shows(tmp_path):
    from code_review_graph.graph import GraphStore
    from code_review_graph.parser import NodeInfo
    from code_review_graph.tools import query_graph

    (tmp_path / ".git").mkdir()
    db = tmp_path / ".code-review-graph" / "graph.db"
    db.parent.mkdir()
    with GraphStore(db) as store:
        store.upsert_node(NodeInfo(
            kind="Function", name="charge", file_path=str(tmp_path / "pay.py"),
            line_start=1, line_end=2, language="python",
        ))
        for i in range(3):
            store.upsert_node(NodeInfo(
                kind="Test", name=f"test_charge_{i}", file_path=str(tmp_path / f"t{i}.py"),
                line_start=1, line_end=2, language="python", is_test=True,
            ))
        store.commit()

    full = query_graph("tests_for", "charge", repo_root=str(tmp_path))
    shown = query_graph("tests_for", "charge", repo_root=str(tmp_path), max_results=1)

    assert full["result_count"] == 3
    assert "3 of them guesses (see inferred_by)" in full["summary"]
    assert "more guesses" not in full["summary"]
    assert "1 of them guesses (see inferred_by)" in shown["summary"]
    assert "2 more guesses among the omitted results" in shown["summary"]


def _detect_changes(repo, extra, capsys):
    argv = ["code-review-graph", "detect-changes", "--repo", str(repo), *extra]
    with patch.object(sys, "argv", argv), \
            patch("code_review_graph.graph.GraphStore"), \
            patch("code_review_graph.incremental.get_db_path"), \
            patch("code_review_graph.incremental.get_changed_files", return_value=["a.py"]), \
            patch(
                "code_review_graph.changes.analyze_changes",
                return_value={"summary": "1 change", "changed_functions": []},
            ):
        cli.main()
    return capsys.readouterr().out


def test_detect_changes_prints_toon_unless_asked_for_json(tmp_path, capsys):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "a.py").write_text("x = 1\n", encoding="utf-8")

    toon_out = _detect_changes(repo, [], capsys)
    json_out = _detect_changes(repo, ["--json"], capsys)

    assert not toon_out.lstrip().startswith("{")
    assert decode(toon_out)["summary"] == "1 change"
    assert json.loads(json_out)["summary"] == "1 change"
