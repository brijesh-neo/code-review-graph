"""Parse every file of a repository with both engines and compare the output.

    .venv/bin/python scripts/check_fast_engine.py REPO [REPO ...] [--limit N]

For each Python, JavaScript or TypeScript file Git tracks under REPO, a
legacy ``CodeParser`` and a ``CRG_FAST_ENGINE=1`` one parse the same
bytes; their nodes and edges must be equal, in order. Prints each file that
differs (its first differing item), then the per-engine parse time. Needs the
``crg_fast`` module (``fast_engine/``) installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from code_review_graph import parser as parser_module  # noqa: E402
from code_review_graph.parser import CodeParser  # noqa: E402

# Module-level caches keyed by source bytes (``ast.parse`` and the like): the
# engine that ran second would get them for free, so each parse starts cold.
_SOURCE_CACHES = [
    value for value in vars(parser_module).values()
    if callable(getattr(value, "cache_clear", None))
]

SUFFIXES = {".py", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".mts", ".cts"}


def tracked_files(repo: Path) -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True,
    ).stdout.decode()
    return [repo / name for name in out.split("\0") if Path(name).suffix in SUFFIXES]


def parse(parser: CodeParser, engine: str, path: Path, source: bytes):
    os.environ["CRG_FAST_ENGINE"] = "1" if engine == "fast" else "0"
    for cached in _SOURCE_CACHES:
        cached.cache_clear()
    start = time.perf_counter()
    nodes, edges = parser.parse_bytes(path, source)
    return time.perf_counter() - start, [repr(n) for n in nodes], [repr(e) for e in edges]


def first_difference(left: list[str], right: list[str]) -> str:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return f"item {index}:\n    legacy {a}\n    fast   {b}"
    return f"lengths {len(left)} vs {len(right)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="stop after N files per repo")
    a = ap.parse_args()
    import crg_fast  # noqa: F401  (fail early when it is not installed)

    differing = total = 0
    times = {"legacy": 0.0, "fast": 0.0}
    for repo in a.repos:
        repo = repo.resolve()
        legacy, fast = CodeParser(repo), CodeParser(repo)
        files = tracked_files(repo)
        if a.limit:
            files = files[: a.limit]
        print(f"{repo}: {len(files)} examples")
        for index, path in enumerate(files, 1):
            source = path.read_bytes()
            t_legacy, nodes_legacy, edges_legacy = parse(legacy, "legacy", path, source)
            t_fast, nodes_fast, edges_fast = parse(fast, "fast", path, source)
            times["legacy"] += t_legacy
            times["fast"] += t_fast
            total += 1
            for what, left, right in (
                ("nodes", nodes_legacy, nodes_fast), ("edges", edges_legacy, edges_fast),
            ):
                if left != right:
                    differing += 1
                    print(f"DIFF {path.relative_to(repo)} {what} {first_difference(left, right)}")
                    break
            if index % 200 == 0:
                print(f"{index}/{len(files)}", flush=True)
    print(
        f"{differing} of {total} files differ; parse time legacy {times['legacy']:.2f} s, "
        f"fast {times['fast']:.2f} s"
    )
    return 1 if differing else 0


if __name__ == "__main__":
    sys.exit(main())
