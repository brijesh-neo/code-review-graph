"""Parse workers share the Python module summaries they compute (CRG_FAST_ENGINE).

Each process worker read and summarised the same imported modules again (star
exports, return annotations, ``sys.modules`` swaps): eight workers did most of
that work eight times. Under the fast engine a worker's new cache entries come
back with its results and go out to the other workers with later chunks.

Only an answer that is a function of the module file alone may travel. Star
exports computed across an import cycle depend on which module of the cycle
was read first, so they stay in the worker that computed them.
"""

import concurrent.futures

import pytest

from code_review_graph import incremental, parser


@pytest.fixture(autouse=True)
def _empty_module_caches():
    def clear():
        for _name, cache, lock in parser._python_module_caches():
            with lock:
                cache.clear()
        parser._PYTHON_STAR_EXPORT_CYCLE_KEYS.clear()

    clear()
    yield
    clear()


def _star_keys(entries):
    return {key[0].rsplit("/", 1)[-1] for name, key, _ in entries if name == "star"}


def test_star_exports_across_a_cycle_are_not_shared(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / "a.py").write_text("from b import *\n\ndef from_a():\n    pass\n")
    (tmp_path / "b.py").write_text("from a import *\n\ndef from_b():\n    pass\n")
    (tmp_path / "leaf.py").write_text("def from_leaf():\n    pass\n")
    (tmp_path / "mid.py").write_text("from leaf import *\n")
    (tmp_path / "via_cycle.py").write_text("from b import *\n")
    (tmp_path / "user.py").write_text(
        "from a import *\nfrom mid import *\nfrom via_cycle import *\n"
    )
    code_parser = parser.CodeParser(tmp_path)
    for name in ("user.py", "a.py", "b.py"):
        path = tmp_path / name
        code_parser.parse_bytes(path, path.read_bytes())

    shared = _star_keys(parser.python_module_cache_entries(set()))

    assert {"leaf.py", "mid.py"} <= shared
    # In the cycle, and reaching it through another module.
    assert not shared & {"a.py", "b.py", "via_cycle.py"}


def test_entries_are_sent_once_and_seeding_keeps_local_values():
    key = ("/repo/m.py", 1, 2)
    parser._PYTHON_RETURN_ANNOTATION_CACHE[key] = {"f": "int"}
    sent: set = set()
    assert parser.python_module_cache_entries(sent) == [("returns", key, {"f": "int"})]
    assert parser.python_module_cache_entries(sent) == []

    other = ("/repo/n.py", 1, 2)
    parser.seed_python_module_caches(
        [("returns", key, {"f": "str"}), ("returns", other, {"g": "bool"})], sent,
    )
    assert parser._PYTHON_RETURN_ANNOTATION_CACHE[key] == {"f": "int"}
    assert parser._PYTHON_RETURN_ANNOTATION_CACHE[other] == {"g": "bool"}
    # A seeded entry is not sent back.
    assert parser.python_module_cache_entries(sent) == []


def test_large_files_get_their_own_chunk_and_go_first(tmp_path):
    for index in range(10):
        (tmp_path / f"f{index}.py").write_text("x = 1\n")
    (tmp_path / "f4.py").write_text("x = 1\n" * 2000)
    args = [(f"f{index}.py", str(tmp_path)) for index in range(10)]

    chunks, heavy = incremental._shared_parse_chunks(args, 3, 2)

    assert [arg for chunk in chunks for arg in chunk] == args
    assert [chunks[index] for index in heavy] == [[("f4.py", str(tmp_path))]]


def test_shared_parse_yields_the_same_results_in_order(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "base.py").write_text(
        "class Base:\n    def run(self) -> int:\n        return 1\n"
    )
    (tmp_path / "api.py").write_text("from base import *\n")
    files = []
    for index in range(24):
        name = f"user{index}.py"
        (tmp_path / name).write_text(
            "from api import *\n\n"
            f"def call{index}():\n    return Base().run()\n" * (40 if index == 5 else 1)
        )
        files.append(name)
    args = [(name, str(tmp_path)) for name in ["api.py", "base.py", *files]]
    monkeypatch.setattr(incremental, "_select_executor_kind", lambda: "process")

    expected = [incremental._parse_single_file(arg) for arg in args]
    with incremental._make_executor(2) as executor:
        assert isinstance(executor, concurrent.futures.ProcessPoolExecutor)
        shared = list(incremental._parse_sharing_caches(executor, args, 4, 2))

    assert [result[0] for result in shared] == [arg[0] for arg in args]
    assert shared == expected
