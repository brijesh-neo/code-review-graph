"""A build or update refreshes target resolution once, after every resolver.

``build_or_update_graph`` used to refresh after the build's resolvers and
again in post-processing, after the bare passes. The first refresh only
repeated work the second one redoes, so the build now leaves it to
post-processing. Every post-processing level must still run it, and the
internal flag must not reach the result.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from code_review_graph.graph import GraphStore
from code_review_graph.tools.build import build_or_update_graph


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com",
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        },
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "lib.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "from lib import helper\n\n\ndef run():\n    return helper()\n", encoding="utf-8",
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def _counted(call):
    original = GraphStore.refresh_target_resolution
    calls: list[int] = []

    def counting(self):
        calls.append(1)
        return original(self)

    with patch.object(GraphStore, "refresh_target_resolution", counting):
        result = call()
    return result, len(calls)


@pytest.mark.parametrize("postprocess", ["full", "minimal", "none"])
def test_build_and_update_refresh_once(repo: Path, postprocess: str) -> None:
    result, calls = _counted(lambda: build_or_update_graph(
        full_rebuild=True, repo_root=str(repo), postprocess=postprocess,
    ))
    assert result["status"] == "ok"
    assert "target_refresh_owed" not in result
    assert calls == 1

    (repo / "lib.py").write_text(
        "def helper():\n    return 2\n\n\ndef extra():\n    return 3\n", encoding="utf-8",
    )
    result, calls = _counted(lambda: build_or_update_graph(
        repo_root=str(repo), postprocess=postprocess,
    ))
    assert result["status"] == "ok"
    assert result["files_updated"] >= 1
    assert "target_refresh_owed" not in result
    assert calls == 1
