"""Export a guild knowledge graph as a self-contained HTML explorer.

Read-only: workers off, no LLM. Typed relations and identity links only —
incidence (``dm_links``) is an index and is not drawn.

Serve the file over HTTP (``--serve``) so ForceAtlas2's layout worker can run;
opening as ``file://`` still works with a one-shot layout.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from icelake.api.client import DiscordMemory
from icelake.config import MemoryConfig
from icelake.visualizer.html import write_html
from icelake.visualizer.snapshot import CenterAmbiguousError, CenterError, build_snapshot


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m icelake.visualizer",
        description=("Explore users, entities, relations, and server facts in one HTML canvas."),
    )
    parser.add_argument(
        "--storage",
        required=True,
        help="Storage URL (mongodb://... or sqlite:///...)",
    )
    parser.add_argument("--guild", help="Guild snowflake")
    parser.add_argument("--center", help="User name/id, entity name, or 'server'")
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Ego-neighborhood hops when --center is set",
    )
    parser.add_argument("--out", type=Path, default=Path("icelake-graph.html"))
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the file over HTTP (enables live layout)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the result in a browser")
    parser.add_argument(
        "--list-guilds",
        action="store_true",
        help="Print guilds in storage and exit",
    )
    return parser.parse_args(argv)


def open_memory(storage: str) -> DiscordMemory:
    config = MemoryConfig.model_validate({"storage": storage, "workers": {"enabled": False}})
    return DiscordMemory(config, llm=None, embedder=None)


async def run(args: argparse.Namespace) -> int:
    memory = open_memory(args.storage)
    async with memory:
        if args.list_guilds:
            for guild_id in sorted(memory.active_guilds):
                print(guild_id)
            return 0
        if not args.guild:
            print("error: --guild is required (or pass --list-guilds)", file=sys.stderr)
            return 1
        try:
            snapshot = await build_snapshot(
                memory, args.guild, center=args.center, depth=args.depth
            )
        except CenterAmbiguousError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except CenterError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        write_html(args.out, snapshot)
        print(
            f"wrote {args.out.resolve()} "
            f"({len(snapshot.nodes)} nodes, {len(snapshot.edges)} edges, "
            f"{len(snapshot.facts)} facts)"
        )

    if args.serve:
        return _serve(args.out, args.port, args.open)
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())
    return 0


def _serve(path: Path, port: int, open_browser: bool) -> int:
    directory = path.parent.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):  # type: ignore[no-untyped-def]
            super().__init__(*a, directory=str(directory), **kw)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/{path.name}"
    print(f"serving {url}")
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(run(parse_args(argv))))


if __name__ == "__main__":
    main()
