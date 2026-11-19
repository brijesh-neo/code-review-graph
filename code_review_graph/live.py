"""A page that follows the session: what it edited, read and has not checked.

``live`` redraws a k-hop neighbourhood of the files changed since ``HEAD``
(plus the files the activity log says were edited) whenever the graph or the
log changes. The page reloads itself, so leaving it open in a browser tab is
enough. Edited files are drawn red, files read green, and callers never read
or tests never run amber.
"""
from __future__ import annotations

import html
import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL = 3.0
COLOURS = {"edited": "#e5484d", "read": "#30a46c", "unchecked": "#f5a623"}


def live_state(repo_root: Path, *, base: str = "HEAD",
               session: Optional[str] = None) -> dict:
    """Repository-relative files: edited, read, and unchecked (callers/tests)."""
    from .activity import read_activity, unchecked
    from .incremental import get_changed_files

    activity = read_activity(repo_root, session)
    try:
        changed = set(get_changed_files(repo_root, base))
    except (RuntimeError, OSError, ValueError) as exc:
        logger.warning("Could not list the changed files: %s", exc)
        changed = set()
    report = unchecked(repo_root, base=base, session=activity["session"])
    pending: set[str] = set()
    for s in report.get("symbols", []):
        pending.update(s["unread_caller_files"])
        pending.update(s["unrun_tests"])
    edited = sorted(changed | activity["edited"])
    return {
        "session": activity["session"],
        "edited": edited,
        "read": sorted(activity["read"] - set(edited)),
        "unchecked": sorted(pending - set(edited)),
        "report": report,
    }


def _overlay(state: dict, interval: float) -> tuple[str, str]:
    """The ``<head>`` refresh tag and the ``<body>`` legend and colouring script."""
    sets = json.dumps({k: state[k] for k in COLOURS}).replace("</", "<\\/")
    colours = json.dumps(COLOURS)
    rows = "".join(
        f'<div><span style="color:{c}">&#9679;</span> {k}: {len(state[k])}</div>'
        for k, c in COLOURS.items()
    )
    head = f'<meta http-equiv="refresh" content="{max(1, int(interval))}">'
    body = f'''<div id="crg-live" style="position:fixed;right:12px;bottom:12px;z-index:9;
background:rgba(20,20,24,.85);color:#ddd;font:12px sans-serif;padding:8px 10px;
border-radius:6px">code-review-graph live{rows}</div>
<script>
(function () {{
  var sets = {sets}, colours = {colours};
  function kind(fp) {{
    if (!fp) return null;
    fp = String(fp).replace(/\\\\/g, "/");
    for (var k in sets) {{
      for (var i = 0; i < sets[k].length; i++) {{
        var rel = sets[k][i];
        if (fp === rel || fp.slice(-rel.length - 1) === "/" + rel) return k;
      }}
    }}
    return null;
  }}
  function paint() {{
    if (typeof d3 === "undefined") return;
    d3.selectAll(".node-g").each(function (d) {{
      var k = kind(d && d.file_path);
      if (k) d3.select(this).select(".node-shape")
        .style("stroke", colours[k]).style("stroke-width", "3px");
    }});
  }}
  setInterval(paint, 1000);
  paint();
}})();
</script>'''
    return head, body


def _empty_page(state: dict, interval: float) -> str:
    head, _ = _overlay(state, interval)
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'>{head}"
            "<title>code-review-graph live</title></head><body style='font:14px sans-serif'>"
            f"<p>Nothing changed since the base yet (session "
            f"{html.escape(str(state['session'] or 'none'))}). This page reloads by itself."
            "</p></body></html>")


def render_live(repo_root: Path, out: Path, *, base: str = "HEAD",
                session: Optional[str] = None, depth: int = 1,
                interval: float = DEFAULT_INTERVAL) -> dict:
    """Write the page once; returns the state it drew."""
    from .graph import GraphStore
    from .incremental import get_db_path
    from .neighbourhood import SeedResolutionError
    from .visualization import generate_html

    repo_root = repo_root.resolve()
    state = live_state(repo_root, base=base, session=session)
    out.parent.mkdir(parents=True, exist_ok=True)
    seeds = [str(repo_root / rel) for rel in state["edited"]]
    if not seeds:
        out.write_text(_empty_page(state, interval), encoding="utf-8")
        return state
    store = GraphStore(get_db_path(repo_root, read_only=True), read_only=True)
    try:
        generate_html(store, out, seed_files=seeds, depth=depth)
    except SeedResolutionError:
        out.write_text(_empty_page(state, interval), encoding="utf-8")
        return state
    finally:
        store.close()
    page = out.read_text(encoding="utf-8")
    head, body = _overlay(state, interval)
    page = page.replace("</head>", head + "</head>", 1)
    cut = page.rfind("</body>")
    page = page[:cut] + body + page[cut:] if cut >= 0 else page + body
    out.write_text(page, encoding="utf-8")
    return state


def _stamp(repo_root: Path) -> tuple:
    from .activity import activity_path
    from .incremental import get_db_path

    paths = [activity_path(repo_root), get_db_path(repo_root, read_only=True)]
    paths.append(paths[1].with_name(paths[1].name + "-wal"))
    stamp = []
    for p in paths:
        try:
            stamp.append(p.stat().st_mtime_ns)
        except OSError:
            stamp.append(0)
    return tuple(stamp)


def run_live(repo_root: Path, out: Path, *, base: str = "HEAD",
             session: Optional[str] = None, depth: int = 1,
             interval: float = DEFAULT_INTERVAL, once: bool = False) -> None:
    """Redraw *out* whenever the graph or the activity log changes."""
    last: tuple = ()
    while True:
        stamp = _stamp(repo_root)
        if stamp != last:
            state = render_live(repo_root, out, base=base, session=session,
                                depth=depth, interval=interval)
            print(f"live: {len(state['edited'])} edited, {len(state['read'])} read, "
                  f"{len(state['unchecked'])} unchecked -> {out}", flush=True)
            last = stamp
        if once:
            return
        time.sleep(interval)
