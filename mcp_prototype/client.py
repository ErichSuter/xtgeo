#!/usr/bin/env python
"""Interactive MCP command-line client for the xtgeo MCP server.

Usage:
    python -m mcp_prototype.client                       # default: auto-finds mcp_server.py
    python -m mcp_prototype.client --server ./mcp_server.py
    python -m mcp_prototype.client --python /path/to/.venv/bin/python --server /path/to/mcp_server.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import readline  # noqa: F401  — enables line-editing in input()
import shlex
import sys
import textwrap
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


# ── Helpers ─────────────────────────────────────────────────────────────────

def _json_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)


def _parse_value(raw: str) -> Any:
    """Best-effort coerce a CLI string to a Python value."""
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.lower() in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    # Try JSON (for lists/dicts)
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


def _parse_kwargs(tokens: list[str]) -> dict[str, Any]:
    """Parse 'key=value' tokens into a dict."""
    result: dict[str, Any] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ValueError(f"Expected key=value, got: {tok!r}")
        key, _, val = tok.partition("=")
        result[key] = _parse_value(val)
    return result


# ── Client class ────────────────────────────────────────────────────────────

class MCPClient:
    """Thin wrapper around an MCP ClientSession for interactive use."""

    def __init__(self, session: ClientSession):
        self.session = session
        self.tools: dict[str, dict] = {}

    async def initialize(self) -> None:
        await self.session.initialize()
        await self.refresh_tools()

    async def refresh_tools(self) -> None:
        result = await self.session.list_tools()
        self.tools = {}
        for tool in result.tools:
            schema = tool.inputSchema or {}
            self.tools[tool.name] = {
                "description": tool.description or "",
                "parameters": schema.get("properties", {}),
                "required": schema.get("required", []),
            }

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        result = await self.session.call_tool(name, arguments or {})
        # Extract text content from the result
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        combined = "\n".join(parts)
        # Try parsing as JSON for pretty display
        try:
            return json.loads(combined)
        except (json.JSONDecodeError, ValueError):
            return combined


# ── REPL commands ───────────────────────────────────────────────────────────

HELP_TEXT = textwrap.dedent("""\
    Commands:
      help                  Show this help
      tools                 List available tools
      describe <tool>       Show tool description and parameters
      call <tool> [k=v ...] Call a tool with keyword arguments
      raw <tool> <json>     Call a tool with a raw JSON argument object
      quit / exit           Exit the client
""")


async def repl(client: MCPClient) -> None:
    print("xtgeo MCP client — type 'help' for commands, 'quit' to exit.\n")

    while True:
        try:
            line = input("mcp> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        tokens = shlex.split(line)
        cmd, args = tokens[0].lower(), tokens[1:]

        if cmd in ("quit", "exit"):
            break

        elif cmd == "help":
            print(HELP_TEXT)

        elif cmd == "tools":
            if not client.tools:
                print("(no tools registered)")
            else:
                for name in sorted(client.tools):
                    desc = client.tools[name]["description"]
                    short = (desc[:70] + "…") if len(desc) > 70 else desc
                    print(f"  {name:40s} {short}")
                print(f"\n  ({len(client.tools)} tools)")

        elif cmd == "describe":
            if not args:
                print("Usage: describe <tool_name>")
                continue
            name = args[0]
            if name not in client.tools:
                print(f"Unknown tool: {name}")
                continue
            info = client.tools[name]
            print(f"\n  {name}")
            print(f"  {info['description']}\n")
            params = info["parameters"]
            required = set(info["required"])
            if params:
                print("  Parameters:")
                for pname, pinfo in params.items():
                    req = " (required)" if pname in required else ""
                    ptype = pinfo.get("type", "any")
                    pdesc = pinfo.get("description", "")
                    default = pinfo.get("default")
                    default_str = f" [default: {default}]" if default is not None else ""
                    print(f"    {pname}: {ptype}{req}{default_str}")
                    if pdesc:
                        print(f"      {pdesc}")
            else:
                print("  (no parameters)")
            print()

        elif cmd == "call":
            if not args:
                print("Usage: call <tool_name> [key=value ...]")
                continue
            name = args[0]
            if name not in client.tools:
                print(f"Unknown tool: {name}")
                continue
            try:
                kwargs = _parse_kwargs(args[1:])
            except ValueError as exc:
                print(f"Error: {exc}")
                continue
            try:
                result = await client.call_tool(name, kwargs)
                print(_json_pretty(result))
            except Exception as exc:
                print(f"Error: {exc}")

        elif cmd == "raw":
            if len(args) < 2:
                print("Usage: raw <tool_name> <json_object>")
                continue
            name = args[0]
            raw_json = " ".join(args[1:])
            try:
                kwargs = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                print(f"Invalid JSON: {exc}")
                continue
            try:
                result = await client.call_tool(name, kwargs)
                print(_json_pretty(result))
            except Exception as exc:
                print(f"Error: {exc}")

        else:
            # Treat unknown commands as tool calls: <tool_name> [k=v ...]
            name = cmd
            if name in client.tools:
                try:
                    kwargs = _parse_kwargs(args)
                except ValueError as exc:
                    print(f"Error: {exc}")
                    continue
                try:
                    result = await client.call_tool(name, kwargs)
                    print(_json_pretty(result))
                except Exception as exc:
                    print(f"Error: {exc}")
            else:
                print(f"Unknown command: {cmd}  (type 'help' for usage)")


# ── Main ────────────────────────────────────────────────────────────────────

def _resolve_server_path(given: str | None) -> str:
    """Find the mcp_server.py script."""
    if given:
        p = Path(given).resolve()
        if not p.exists():
            sys.exit(f"Server script not found: {p}")
        return str(p)
    # Auto-detect: look relative to this file's repo root
    candidates = [
        Path(__file__).resolve().parent.parent / "mcp_server.py",
        Path.cwd() / "mcp_server.py",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    sys.exit(
        "Could not find mcp_server.py. "
        "Use --server /path/to/mcp_server.py."
    )


def _resolve_python(given: str | None) -> str:
    if given:
        return given
    # Try the venv next to mcp_server.py
    repo = Path(__file__).resolve().parent.parent
    venv_python = repo / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


async def async_main(python: str, server: str) -> None:
    params = StdioServerParameters(
        command=python,
        args=[server],
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            client = MCPClient(session)
            await client.initialize()
            print(f"Connected to server: {server}")
            print(f"Python: {python}")
            print(f"Tools available: {len(client.tools)}\n")
            await repl(client)

    print("Disconnected.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive CLI client for the xtgeo MCP server.",
    )
    parser.add_argument(
        "--server",
        default=None,
        help="Path to the MCP server script (default: auto-detect mcp_server.py)",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Path to the Python interpreter to run the server (default: .venv/bin/python)",
    )
    args = parser.parse_args()

    server = _resolve_server_path(args.server)
    python = _resolve_python(args.python)

    asyncio.run(async_main(python, server))


if __name__ == "__main__":
    main()
