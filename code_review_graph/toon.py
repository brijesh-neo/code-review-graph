"""TOON (Token-Oriented Object Notation) for tool results.

The CLI prints graph answers as TOON and the MCP server returns them as TOON
text. TOON keeps JSON's data model but writes a list of records as a table:
the field names once in a header, then one comma-separated row per record::

    results[2]{file_path,line,name}:
      app/main.py,12,run
      app/util.py,3,helper

Everything else follows the TOON spec (https://github.com/toon-format/spec):
two-space indentation, ``key: value`` fields, ``key[N]: a,b`` for lists of
plain values, ``- item`` for mixed lists, and double quotes only where a string
would otherwise read as something else.

Two choices make more lists fit the table form, so a reader may see them:

- A nested record whose values are all plain is written as dotted columns
  (``call_site.line``), as the spec's key folding does. ``decode`` expands them.
- Records with different fields share one header; a field a record lacks is
  written as ``null``.

The CLI and the MCP server also pass results through ``drop_repeated_fields``
first, which leaves out the fields the rest give back exactly;
``restore_repeated_fields`` puts them back, so nothing is lost.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Any

_INDENT = "  "
_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*\Z")
_SEGMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")
_LEADING_ZERO_RE = re.compile(r"-?0\d")
_NEEDS_QUOTES_RE = re.compile(r'[:"\\\[\]{},\x00-\x1f\x7f]')
_ESCAPES = {"\\": "\\\\", '"': '\\"', "\n": "\\n", "\r": "\\r", "\t": "\\t"}
_UNESCAPES = {"\\": "\\", '"': '"', "n": "\n", "r": "\r", "t": "\t"}


def _plain(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _normalize(value: Any) -> Any:
    """Turn *value* into JSON's data model, the way ``json.dumps(default=str)`` would."""
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if _plain(value):
        return value
    return str(value)


def _quote(text: str) -> str:
    return '"' + "".join(_ESCAPES.get(ch, ch) for ch in text) + '"'


def _string(text: str) -> str:
    if (
        text == ""
        or text != text.strip()
        or text in ("true", "false", "null")
        or _NUMBER_RE.match(text)
        or _LEADING_ZERO_RE.match(text)
        or _NEEDS_QUOTES_RE.search(text)
        or text.startswith("-")
    ):
        return _quote(text)
    return text


def _number(value: float) -> str:
    if not math.isfinite(value):
        return "null"
    if value == 0:
        return "0"
    text = repr(value)
    if "e" in text or "E" in text:
        text = format(Decimal(text), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _scalar(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _number(value)
    return _string(value)


def _key(key: str) -> str:
    return key if _KEY_RE.match(key) else _quote(key)


def _table(items: list) -> tuple[list[str], list[list[Any]]] | None:
    """Header names and rows for *items* in table form, or None when they do not fit one."""
    if not all(isinstance(item, dict) and item for item in items):
        return None
    columns: dict[tuple[str, ...], None] = {}
    flat = []
    for item in items:
        row: dict[tuple[str, ...], Any] = {}
        for key, value in item.items():
            if _plain(value):
                row[(key,)] = value
            elif (
                isinstance(value, dict) and value
                and _SEGMENT_RE.match(key)
                and all(_plain(v) and _SEGMENT_RE.match(k) for k, v in value.items())
            ):
                for sub, sub_value in value.items():
                    row[(key, sub)] = sub_value
            else:
                return None
        flat.append(row)
        columns.update(dict.fromkeys(row))
    # A plain column named like a folded record would decode ambiguously.
    heads = {path[0] for path in columns if len(path) == 2}
    if any(len(path) == 1 and path[0] in heads for path in columns):
        return None
    names = [
        ".".join(path) if len(path) == 2
        else path[0] if _SEGMENT_RE.match(path[0]) else _quote(path[0])
        for path in columns
    ]
    rows = [[row.get(path) for path in columns] for row in flat]
    return names, rows


def _array(key: str, items: list, depth: int, child: int, lines: list[str], lead: str) -> None:
    head = f"{_INDENT * depth}{lead}{key}[{len(items)}]"
    if not items:
        lines.append(head + ":")
        return
    if all(_plain(item) for item in items):
        lines.append(head + ": " + ",".join(_scalar(item) for item in items))
        return
    table = _table(items)
    if table is not None:
        columns, rows = table
        lines.append(head + "{" + ",".join(columns) + "}:")
        pad = _INDENT * child
        for row in rows:
            lines.append(pad + ",".join(_scalar(cell) for cell in row))
        return
    lines.append(head + ":")
    for item in items:
        _item(item, child, lines)


def _field(key: str, value: Any, depth: int, child: int, lines: list[str], lead: str = "") -> None:
    if isinstance(value, dict):
        lines.append(f"{_INDENT * depth}{lead}{key}:")
        _object(value, child, lines)
    elif isinstance(value, list):
        _array(key, value, depth, child, lines, lead)
    else:
        lines.append(f"{_INDENT * depth}{lead}{key}: {_scalar(value)}")


def _object(obj: dict, depth: int, lines: list[str]) -> None:
    for key, value in obj.items():
        _field(_key(key), value, depth, depth + 1, lines)


def _item(item: Any, depth: int, lines: list[str]) -> None:
    if isinstance(item, dict):
        if not item:
            lines.append(f"{_INDENT * depth}-")
            return
        (first, value), *rest = item.items()
        # The first field shares the hyphen line, so what it holds goes one
        # level deeper than the item's other fields.
        _field(_key(first), value, depth, depth + 2, lines, lead="- ")
        for key, other in rest:
            _field(_key(key), other, depth + 1, depth + 2, lines)
    elif isinstance(item, list):
        _array("", item, depth, depth + 1, lines, lead="- ")
    else:
        lines.append(f"{_INDENT * depth}- {_scalar(item)}")


#: The fields ``node_to_dict`` writes for every node; a record with all of
#: them is a node.
_NODE_KEYS = frozenset({
    "id", "kind", "name", "qualified_name", "file_path", "line_start", "line_end",
    "language", "parent_name", "is_test",
})
#: The node fields that can be left out because the rest give them back.
_DERIVED_NODE_KEYS = ("qualified_name", "language", "is_test")
_KEPT_NODE_KEYS = _NODE_KEYS - set(_DERIVED_NODE_KEYS)
#: The fields ``edge_to_dict`` writes for every edge.
_EDGE_KEYS = ("id", "kind", "source", "target", "file_path", "line", "confidence",
              "confidence_tier")
#: Says which edges were folded into rows: the kind they share, which end is
#: the row (``source`` or ``target``) and the node at the other end.
_FOLDED = "edges_in_rows"
_TEST_SUFFIX = " (test)"


def _language_of(path: Any) -> str | None:
    if not isinstance(path, str):
        return None
    from .parser import EXTENSION_TO_LANGUAGE

    dot = path.rfind(".")
    if dot <= path.replace("\\", "/").rfind("/"):
        return None
    return EXTENSION_TO_LANGUAGE.get(path[dot:].lower())


def _qualified_name_of(node: dict) -> str:
    path, name, parent = node["file_path"], node["name"], node["parent_name"]
    if node["kind"] == "File":
        return str(path)
    return f"{path}::{parent}.{name}" if parent else f"{path}::{name}"


def _is_test_of(kind: Any) -> bool:
    return kind == "Test" or (isinstance(kind, str) and kind.endswith(_TEST_SUFFIX))


def _derives(field: str, node: dict) -> bool:
    """Whether *node*'s *field* is exactly what ``_restore_node`` would rebuild."""
    value = node[field]
    if field == "qualified_name":
        return isinstance(node["file_path"], str) and value == _qualified_name_of(node)
    if field == "language":
        return value is not None and value == _language_of(node["file_path"])
    kind = node["kind"]
    if not isinstance(kind, str) or kind.endswith(_TEST_SUFFIX) or type(value) is not bool:
        return False
    return value or kind != "Test"


def _is_node(item: Any) -> bool:
    return isinstance(item, dict) and _NODE_KEYS <= item.keys() and item["id"] is not None


def _drop_node_fields(items: list) -> list:
    """*items* with the node fields the rest of each node gives back left out.

    TOON writes a field one row lacks as ``null``, so a ``null`` must still
    read back right. ``qualified_name`` and ``is_test`` are never null on a
    node, so a null one means "rebuild it" and each node drops them on its
    own. ``language`` can be null, so it is left out of every node or none,
    and only when no other record in the list has one. Any other record in
    the list must carry no ``id``, which is how ``restore_repeated_fields``
    tells nodes apart.
    """
    nodes = [item for item in items if _is_node(item)]
    others = [item for item in items if isinstance(item, dict) and not _is_node(item)]
    if not nodes or any(item.get("id") is not None for item in others):
        return items
    per_row = [f for f in ("qualified_name", "is_test") if all(n[f] is not None for n in nodes)]
    whole = (
        not any("language" in item for item in others)
        and all(_derives("language", n) for n in nodes)
    )
    out = []
    for item in items:
        if not _is_node(item):
            out.append(item)
            continue
        drop = {f for f in per_row if _derives(f, item)} | ({"language"} if whole else set())
        row = {key: value for key, value in item.items() if key not in drop}
        if "is_test" in drop and item["is_test"] and item["kind"] != "Test":
            row["kind"] = f"{item['kind']}{_TEST_SUFFIX}"
        out.append(row)
    return out


def _restore_node(record: dict) -> dict:
    if not (_KEPT_NODE_KEYS <= record.keys()) or record["id"] is None:
        return record
    # A null qualified_name or is_test is one a table row left out; a null
    # language is a real value, as language is only ever left out of a whole list.
    missing = [
        f for f in _DERIVED_NODE_KEYS
        if f not in record or (f != "language" and record[f] is None)
    ]
    if not missing:
        return record
    node = dict(record)
    if "is_test" in missing:
        kind = node["kind"]
        node["is_test"] = _is_test_of(kind)
        if isinstance(kind, str) and kind.endswith(_TEST_SUFFIX):
            node["kind"] = kind[: -len(_TEST_SUFFIX)]
    if "language" in missing:
        node["language"] = _language_of(node["file_path"])
    if "qualified_name" in missing:
        node["qualified_name"] = _qualified_name_of(node)
    return {key: node[key] for key in dict.fromkeys((*record.keys(), *missing))}


def _call_site(row: dict) -> dict | None:
    call_site = row.get("call_site")
    return call_site if isinstance(call_site, dict) else None


def _call_file(row: dict, call_site: dict) -> Any:
    # A call site names its file only when it is not the row's own.
    file = call_site.get("file")
    return row.get("file_path") if file is None else file


def _folded_edge(edge: dict, row: dict) -> dict | None:
    """What *row* must carry for *edge* to be rebuilt from it, or None."""
    allowed = set(_EDGE_KEYS) | {"target_resolution"}
    if not (set(_EDGE_KEYS) <= edge.keys() <= allowed):
        return None
    if edge.get("target_resolution") != row.get("target_resolution"):
        return None
    tier, confidence = edge["confidence_tier"], edge["confidence"]
    if not isinstance(tier, str) or isinstance(confidence, bool) or not isinstance(
        confidence, (int, float)
    ):
        return None
    folded: dict[str, Any] = {"id": edge["id"]}
    if tier != "EXTRACTED" or confidence != 1:
        folded["tier"], folded["confidence"] = tier, confidence
    call_site = _call_site(row)
    if call_site is not None:
        if (_call_file(row, call_site), call_site.get("line")) != (
            edge["file_path"], edge["line"]
        ):
            return None
    else:
        if edge["file_path"] != row.get("file_path") or edge["line"] is None:
            return None
        folded["line"] = edge["line"]
    return folded


def _fold_edges(response: dict) -> dict:
    """*response* with each edge that repeats a result row folded into that row.

    A query pairs each result row with the edge that found it, and the edge
    repeats the row's name, file and call line. Such an edge becomes the row's
    ``edge`` field: its id, plus its tier and confidence when it is not a plain
    extracted one, plus its line when the row has no call site.
    ``edges_in_rows`` gives the kind, which end the row is and the node at the
    other end, once for all of them. Every other edge stays in ``edges``.
    Nothing is folded unless ``restore_repeated_fields`` rebuilds the response
    exactly, edge order included.
    """
    rows, edges = response["results"], response["edges"]
    if _FOLDED in response or not all(isinstance(r, dict) and "edge" not in r for r in rows):
        return response
    for end, other in (("source", "target"), ("target", "source")):
        first = next(
            (e for e in edges if isinstance(e, dict) and e.get(end) is not None), None,
        )
        if first is None or not rows:
            continue
        meta = {"kind": first.get("kind"), "row_end": end, "other": first.get(other)}
        out = [dict(row) for row in rows]
        kept: list = []
        folded_count = 0
        next_row = 0
        for edge in edges:
            # Folded edges come back first, in row order, so folding stops at
            # the first edge that stays.
            if not kept and isinstance(edge, dict) and (
                (edge.get("kind"), edge.get(other)) == (meta["kind"], meta["other"])
            ):
                match = next(
                    (
                        (i, carried)
                        for i in range(next_row, len(out))
                        if edge.get(end) == out[i].get("qualified_name")
                        and (carried := _folded_edge(edge, out[i])) is not None
                    ),
                    None,
                )
                if match is not None:
                    out[match[0]]["edge"] = match[1]
                    next_row = match[0] + 1
                    folded_count += 1
                    continue
            kept.append(edge)
        if folded_count:
            return {**response, "results": out, "edges": kept, _FOLDED: meta}
    return response


def _unfold_edges(response: dict) -> dict:
    meta = response[_FOLDED]
    end = meta["row_end"]
    other = "target" if end == "source" else "source"
    rows, folded = [], []
    for row in response["results"]:
        carried = row.get("edge") if isinstance(row, dict) else None
        if isinstance(row, dict) and "edge" in row:
            row = {key: value for key, value in row.items() if key != "edge"}
        rows.append(row)
        if not isinstance(carried, dict) or carried.get("id") is None:
            continue
        call_site = _call_site(row)
        edge = {
            "id": carried["id"], "kind": meta["kind"], end: row["qualified_name"],
            other: meta["other"],
            "file_path": _call_file(row, call_site) if call_site else row.get("file_path"),
            "line": call_site.get("line") if call_site else carried.get("line"),
            "confidence": 1.0 if carried.get("tier") is None else carried["confidence"],
            "confidence_tier": carried.get("tier") or "EXTRACTED",
        }
        if "target_resolution" in row and row["target_resolution"] is not None:
            edge["target_resolution"] = row["target_resolution"]
        folded.append({key: edge[key] for key in (*_EDGE_KEYS, *edge.keys()) if key in edge})
    out = {key: value for key, value in response.items() if key != _FOLDED}
    out["results"] = rows
    out["edges"] = folded + list(response.get("edges") or [])
    return out


def _drop(value: Any) -> Any:
    if isinstance(value, list):
        return _drop_node_fields([_drop(item) for item in value])
    if not isinstance(value, dict):
        return value
    if isinstance(value.get("results"), list) and isinstance(value.get("edges"), list):
        value = _fold_edges(value)
    return {
        key: _drop_node_fields([_drop(item)])[0] if isinstance(item, dict) else _drop(item)
        for key, item in value.items()
    }


def drop_repeated_fields(value: Any) -> Any:
    """*value* without the fields the rest of it gives back exactly.

    For TOON output only; ``--json`` keeps every field, and
    ``restore_repeated_fields`` rebuilds them. Leaves out:

    - each query edge that repeats its result row (see ``_fold_edges``);
    - a node's ``qualified_name`` when it is ``<file>`` (a File),
      ``<file>::<parent>.<name>`` or ``<file>::<name>``;
    - a node's ``is_test``, which ``kind`` then carries (``Function (test)``);
    - ``language`` when the file extension of every node in the list maps to
      it (see ``_drop_node_fields``).
    """
    return _drop_node_fields([_drop(value)])[0] if isinstance(value, dict) else _drop(value)


def restore_repeated_fields(value: Any) -> Any:
    """Undo ``drop_repeated_fields``: the same value with every field back."""
    if isinstance(value, list):
        return [restore_repeated_fields(item) for item in value]
    if not isinstance(value, dict):
        return value
    value = _restore_node({key: restore_repeated_fields(item) for key, item in value.items()})
    if isinstance(value.get(_FOLDED), dict) and isinstance(value.get("results"), list):
        value = _unfold_edges(value)
    return value


def encode(value: Any) -> str:
    """Return *value* (a JSON-like structure) as TOON text."""
    value = _normalize(value)
    lines: list[str] = []
    if isinstance(value, dict):
        _object(value, 0, lines)
    elif isinstance(value, list):
        _array("", value, 0, 1, lines, "")
    else:
        lines.append(_scalar(value))
    return "\n".join(lines)


# Decoding ------------------------------------------------------------------


def _unquote(text: str, start: int) -> tuple[str, int]:
    """The string quoted at *start* and the index just past its closing quote."""
    out = []
    i = start + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            if i + 1 >= len(text) or text[i + 1] not in _UNESCAPES:
                raise ValueError(f"bad escape in {text!r}")
            out.append(_UNESCAPES[text[i + 1]])
            i += 2
        elif ch == '"':
            return "".join(out), i + 1
        else:
            out.append(ch)
            i += 1
    raise ValueError(f"unterminated string in {text!r}")


def _parse_scalar(token: str) -> Any:
    token = token.strip()
    if token.startswith('"'):
        text, end = _unquote(token, 0)
        if token[end:].strip():
            raise ValueError(f"text after string in {token!r}")
        return text
    if token in ("true", "false"):
        return token == "true"
    if token == "null":
        return None
    if _NUMBER_RE.match(token) and not _LEADING_ZERO_RE.match(token):
        number = float(token)
        if number.is_integer() and not any(c in token for c in ".eE"):
            return int(token)
        return number
    return token


def _cells(text: str) -> list[Any]:
    cells, current, i = [], [], 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            _, end = _unquote(text, i)
            current.append(text[i:end])
            i = end
        elif ch == ",":
            cells.append(_parse_scalar("".join(current)))
            current = []
            i += 1
        else:
            current.append(ch)
            i += 1
    cells.append(_parse_scalar("".join(current)))
    return cells


_HEADER_RE = re.compile(r"\[(\d+)\](?:\{(.*)\})?:(?: (.*))?\Z")


def _split_field(content: str) -> tuple[str, str] | None:
    """Split ``key<rest>`` where rest starts with ``:`` or ``[``; None for a bare value."""
    if content.startswith('"'):
        key, end = _unquote(content, 0)
        rest = content[end:]
        return (key, rest) if rest[:1] in (":", "[") else None
    for i, ch in enumerate(content):
        if ch in ":[":
            return content[:i], content[i:]
        if ch in '"{}],':
            return None
    return None


def _header(text: str) -> list[tuple[str, ...]]:
    """Column paths in a table header; an unquoted dotted name is a folded record."""
    paths: list[tuple[str, ...]] = []
    for name in text.split(",") if '"' not in text else _raw_cells(text):
        if name.startswith('"'):
            paths.append((_unquote(name, 0)[0],))
        else:
            paths.append(tuple(name.split(".")))
    return paths


def _raw_cells(text: str) -> list[str]:
    cells, start, i = [], 0, 0
    while i < len(text):
        if text[i] == '"':
            i = _unquote(text, i)[1]
        elif text[i] == ",":
            cells.append(text[start:i])
            start = i = i + 1
        else:
            i += 1
    cells.append(text[start:])
    return cells


def _expand(paths: list[tuple[str, ...]], cells: list[Any]) -> dict:
    out: dict[str, Any] = {}
    for path, value in zip(paths, cells):
        if len(path) == 2:
            out.setdefault(path[0], {})[path[1]] = value
        else:
            out[path[0]] = value
    # A folded record that is null in every column was missing from this row.
    for key in [k for k, v in out.items() if isinstance(v, dict)]:
        if all(v is None for v in out[key].values()):
            out[key] = None
    return out


class _Reader:
    def __init__(self, text: str) -> None:
        self.lines = []
        for raw in text.split("\n"):
            if not raw.strip():
                continue
            spaces = len(raw) - len(raw.lstrip(" "))
            self.lines.append((spaces // 2, raw.strip()))
        self.pos = 0

    def peek(self) -> tuple[int, str] | None:
        return self.lines[self.pos] if self.pos < len(self.lines) else None

    def array(self, rest: str, child: int) -> list:
        match = _HEADER_RE.match(rest)
        if not match:
            raise ValueError(f"bad array header {rest!r}")
        count, columns, inline = int(match.group(1)), match.group(2), match.group(3)
        if inline is not None:
            values = _cells(inline)
        elif columns is not None:
            paths = _header(columns)
            values = []
            for _ in range(count):
                _, line = self.lines[self.pos]
                self.pos += 1
                values.append(_expand(paths, _cells(line)))
        else:
            values = [self.item(child) for _ in range(count)]
        if len(values) != count:
            raise ValueError(f"expected {count} values, found {len(values)}")
        return values

    def value(self, rest: str, child: int) -> Any:
        if rest.startswith("["):
            return self.array(rest, child)
        if rest == ":":
            nxt = self.peek()
            return self.object(child) if nxt and nxt[0] >= child else {}
        if not rest.startswith(": "):
            raise ValueError(f"expected ': ' in {rest!r}")
        return _parse_scalar(rest[2:])

    def object(self, depth: int) -> dict:
        obj: dict[str, Any] = {}
        while (nxt := self.peek()) and nxt[0] == depth and not nxt[1].startswith("-"):
            self.pos += 1
            split = _split_field(nxt[1])
            if split is None:
                raise ValueError(f"expected a field, found {nxt[1]!r}")
            key, rest = split
            obj[key] = self.value(rest, depth + 1)
        return obj

    def item(self, depth: int) -> Any:
        _, line = self.lines[self.pos]
        self.pos += 1
        if line == "-":
            return {}
        content = line[2:]
        if content.startswith("["):
            return self.array(content, depth + 1)
        split = _split_field(content)
        if split is None:
            return _parse_scalar(content)
        key, rest = split
        obj = {key: self.value(rest, depth + 2)}
        obj.update(self.object(depth + 1))
        return obj


def decode(text: str) -> Any:
    """Read TOON text written by :func:`encode` back into Python values."""
    reader = _Reader(text)
    first = reader.peek()
    if first is None:
        return {}
    if first[1].startswith("["):
        reader.pos += 1
        return reader.array(first[1], 1)
    if len(reader.lines) == 1 and _split_field(first[1]) is None:
        return _parse_scalar(first[1])
    return reader.object(0)
