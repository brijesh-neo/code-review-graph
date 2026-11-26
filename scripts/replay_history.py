"""Replay this repository's history through ``update`` with two code versions.

A clone of this repository under ``tmp/replay/repo`` is checked out at each
first-parent commit in turn. Two graphs follow it: one kept by the baseline
code (``--base``, default ``staging``, extracted to ``tmp/replay/basecode``)
and one by this checkout's code. The first commit is a ``build``; every later
one is an ``update``. After each step both graphs are dumped, sorted and
without row ids or timestamps, and compared byte for byte.

    uv run python scripts/replay_history.py [--commits 30] [--rev staging] [--base staging]
        [--new-code checkout|base] [--work tmp/replay]

Prints ``N/TOTAL`` progress, both timings per step and every step whose graphs
differ; the differing dumps stay in ``tmp/replay/`` for ``diff``. Exits 1 when
any step differs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp" / "replay"
REPO = WORK / "repo"
BASE_CODE = WORK / "basecode"


def _set_work(work: Path) -> None:
    global WORK, REPO, BASE_CODE
    WORK, REPO, BASE_CODE = work, work / "repo", work / "basecode"


def _git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def extract_code(rev: str) -> None:
    if BASE_CODE.exists():
        shutil.rmtree(BASE_CODE)
    BASE_CODE.mkdir(parents=True)
    archive = WORK / "base.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(archive), rev, "code_review_graph"],
        cwd=ROOT, check=True,
    )
    with tarfile.open(archive) as tar:
        tar.extractall(BASE_CODE, filter="data")
    archive.unlink()


def run(args: list[str], data_dir: Path, code: Path) -> float:
    # ``-P`` keeps the checked-out commit's own package (the cwd) off sys.path.
    # Set iteration order follows string hashes, which Python varies per process;
    # update visits some files in set order, which decides row ids and so the
    # community ids. One fixed seed for both sides leaves only the code to differ.
    env = {
        **os.environ, "CRG_DATA_DIR": str(data_dir), "PYTHONPATH": str(code),
        "PYTHONHASHSEED": "0",
    }
    t = time.perf_counter()
    done = subprocess.run(
        [sys.executable, "-P", "-m", "code_review_graph", *args],
        cwd=REPO, env=env, capture_output=True, text=True,
    )
    if done.returncode:
        raise SystemExit(f"{' '.join(args)} failed with {code}:\n{done.stderr[-2000:]}")
    return time.perf_counter() - t


def dump(data_dir: Path, name: str) -> list[Path]:
    conn = sqlite3.connect(data_dir / "graph.db")
    out = []
    for table in ("nodes", "edges"):
        cols = [
            r[1] for r in conn.execute(f"PRAGMA table_info({table})")  # nosec B608
            if r[1] not in ("id", "updated_at")
        ]
        rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table}").fetchall()  # nosec B608
        path = WORK / f"{name}.{table}"
        path.write_text("\n".join(sorted("\t".join(repr(v) for v in row) for row in rows)) + "\n")
        out.append(path)
    qualified = dict(conn.execute("SELECT id, qualified_name FROM nodes"))
    rows = []
    for flow, entry, depth, nodes, files, criticality, path_json in conn.execute(
        "SELECT name, entry_point_id, depth, node_count, file_count, criticality, path_json "
        "FROM flows"
    ):
        steps = [qualified.get(node_id, node_id) for node_id in json.loads(path_json)]
        rows.append(repr((flow, qualified.get(entry, entry), depth, nodes, files,
                          criticality, steps)))
    names = dict(conn.execute("SELECT id, name FROM communities"))
    for row in conn.execute(
        "SELECT name, level, parent_id, cohesion, size, dominant_language, description "
        "FROM communities"
    ):
        rows.append(repr(("community", *row[:2], names.get(row[2], row[2]), *row[3:])))
    path = WORK / f"{name}.derived"
    path.write_text("\n".join(sorted(rows)) + "\n")
    out.append(path)
    conn.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commits", type=int, default=30)
    ap.add_argument("--rev", default="staging", help="history to replay")
    ap.add_argument("--base", default="staging", help="baseline code revision")
    ap.add_argument(
        "--new-code", choices=("checkout", "base"), default="checkout",
        help="'base' runs the baseline code twice, to see what varies between runs",
    )
    ap.add_argument("--work", type=Path, default=WORK, help="scratch directory")
    a = ap.parse_args()
    _set_work(a.work.resolve())

    WORK.mkdir(parents=True, exist_ok=True)
    extract_code(a.base)
    commits = _git(
        "rev-list", "--first-parent", "--reverse", f"-n{a.commits}", a.rev,
    ).split()
    if REPO.exists():
        shutil.rmtree(REPO)
    _git("clone", "-q", "--no-checkout", "--shared", str(ROOT), str(REPO))
    data = {"base": WORK / "data_base", "new": WORK / "data_new"}
    code = {"base": BASE_CODE, "new": BASE_CODE if a.new_code == "base" else ROOT}
    for d in data.values():
        shutil.rmtree(d, ignore_errors=True)

    print(f"{len(commits)} examples (commits of {a.rev}), baseline code {a.base}")
    differing = 0
    for i, commit in enumerate(commits, 1):
        _git("checkout", "-q", "--detach", "--force", commit, cwd=REPO)
        command = ["build"] if i == 1 else ["update"]
        times = {k: run(command, data[k], code[k]) for k in ("base", "new")}
        base_dumps = dump(data["base"], "base")
        new_dumps = dump(data["new"], "new")
        same = all(b.read_bytes() == n.read_bytes() for b, n in zip(base_dumps, new_dumps))
        print(
            f"{i}/{len(commits)} {commit[:9]} {command[0]}: "
            f"base {times['base']:.2f} s, new {times['new']:.2f} s, "
            f"{'same' if same else 'DIFFERENT'}",
            flush=True,
        )
        if not same:
            differing += 1
            for path in base_dumps + new_dumps:
                shutil.copy(path, path.with_name(f"{commit[:9]}_{path.name}"))
    print(f"{differing} of {len(commits)} steps differ")
    sys.exit(1 if differing else 0)


if __name__ == "__main__":
    main()
