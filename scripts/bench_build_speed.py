"""Time ``build`` and ``update`` on a frozen copy of this repository.

The copy is taken once from a Git revision (default ``HEAD``) into
``tmp/bench_build/repo`` so later edits to the parser do not change the input.
Each run uses its own ``CRG_DATA_DIR`` under ``tmp/bench_build/``.

    uv run python scripts/bench_build_speed.py [--runs 3] [--rev staging] [--dump NAME]

``--dump NAME`` writes the built graph's nodes and edges, sorted and without
row ids or timestamps, to ``tmp/bench_build/NAME.nodes`` and ``NAME.edges`` so
two code versions can be compared with ``cmp``; ``NAME_update.*`` hold the
graph after the one-file ``update``. ``--snapshot-code`` runs the snapshot's
own copy of the package, which is the baseline when ``--rev`` is the base
branch.

``--repo PATH`` benchmarks an existing Git repository in place instead of a
snapshot (it is not modified: the edited file is restored), and ``--edit FILE``
names the file the ``update`` run appends to; a JS/TS file gets a JS function.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp" / "bench_build"
REPO = WORK / "repo"
# The console script, not ``-m``: ``-m`` would import the snapshot's own copy from the cwd.
CLI = [str(Path(sys.executable).parent / "code-review-graph")]
# A hub file: many nodes and many dependents, so ``update`` does real work.
EDITED = "code_review_graph/graph.py"


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def make_repo(rev: str) -> None:
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.mkdir(parents=True)
    archive = WORK / "src.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(archive), rev], cwd=ROOT, check=True
    )
    with tarfile.open(archive) as tar:
        tar.extractall(REPO, filter="data")
    archive.unlink()
    _git("init", "-q", cwd=REPO)
    _git("add", "-A", cwd=REPO)
    _git(
        "-c", "user.name=bench", "-c", "user.email=bench@example.invalid",
        "commit", "-q", "-m", "snapshot", cwd=REPO,
    )


def run(args: list[str], data_dir: Path, cli: list[str]) -> float:
    env = {**os.environ, "CRG_DATA_DIR": str(data_dir)}
    t = time.perf_counter()
    subprocess.run(cli + args, cwd=REPO, env=env, check=True, capture_output=True)
    return time.perf_counter() - t


def _community_names(conn: sqlite3.Connection, cols: list[str], rows: list) -> list:
    """*rows* of nodes with ``community_id`` read as its community's name.

    The id is a rowid, like ``nodes.id``: which number a community gets depends
    on what the table held before, not on the graph.
    """
    if "community_id" not in cols:
        return rows
    at = cols.index("community_id")
    names = dict(conn.execute("SELECT id, name FROM communities"))
    return [
        (*row[:at], names.get(row[at], row[at]), *row[at + 1:]) for row in rows
    ]


def dump(data_dir: Path, name: str) -> None:
    conn = sqlite3.connect(data_dir / "graph.db")
    node_cols = [
        r[1] for r in conn.execute("PRAGMA table_info(nodes)") if r[1] not in ("id", "updated_at")
    ]
    edge_cols = [
        r[1] for r in conn.execute("PRAGMA table_info(edges)") if r[1] not in ("id", "updated_at")
    ]
    for table, cols in (("nodes", node_cols), ("edges", edge_cols)):
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()  # nosec B608
        if table == "nodes":
            rows = _community_names(conn, cols, rows)
        lines = sorted("\t".join(repr(v) for v in row) for row in rows)
        (WORK / f"{name}.{table}").write_text("\n".join(lines) + "\n")
        print(f"{table}: {len(lines)} rows -> {WORK / f'{name}.{table}'}")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--keep-repo", action="store_true", help="reuse tmp/bench_build/repo")
    ap.add_argument("--dump")
    ap.add_argument("--repo", type=Path, help="an existing Git repository to build in place")
    ap.add_argument("--edit", default=EDITED, help="file the update run appends to")
    ap.add_argument(
        "--snapshot-code", action="store_true",
        help="run the snapshot's own code (the baseline) instead of this checkout's",
    )
    a = ap.parse_args()
    global REPO
    if a.repo is not None:
        REPO = a.repo.resolve()
    cli = [sys.executable, "-m", "code_review_graph"] if a.snapshot_code else CLI

    WORK.mkdir(parents=True, exist_ok=True)
    if a.repo is None and not (a.keep_repo and REPO.exists()):
        make_repo(a.rev)
    files = len(_git("ls-files", cwd=REPO).split())
    print(f"{files} files in {REPO if a.repo is not None else f'snapshot of {a.rev}'}")
    added = (
        b"\n\ndef _bench_added():\n    return 1\n" if a.edit.endswith(".py")
        else b"\n\nexport function benchAdded() {\n  return 1;\n}\n"
    )

    builds, updates = [], []
    for i in range(a.runs):
        data_dir = WORK / f"data{i}"
        shutil.rmtree(data_dir, ignore_errors=True)
        builds.append(run(["build"], data_dir, cli))
        if i == 0 and a.dump:
            dump(data_dir, a.dump)
        edited = REPO / a.edit
        original = edited.read_bytes()
        try:
            edited.write_bytes(original + added)
            updates.append(run(["update"], data_dir, cli))
            if i == 0 and a.dump:
                dump(data_dir, f"{a.dump}_update")
        finally:
            edited.write_bytes(original)
        print(f"run {i + 1}/{a.runs}: build {builds[-1]:.2f} s, update {updates[-1]:.2f} s")
        shutil.rmtree(data_dir, ignore_errors=True)
    print(
        f"median build {statistics.median(builds):.2f} s, "
        f"median update {statistics.median(updates):.2f} s"
    )


if __name__ == "__main__":
    main()
