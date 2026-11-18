"""
MCP client: turns the tools of any stdio MCP server into LangChain tools.

The LangGraph agent in this app is synchronous, while MCP sessions are
asyncio-only. The bridge is one background thread that owns an event loop
and the live session; every tool call is scheduled onto that loop and the
calling thread waits for the result. One server process per configured
server, started lazily on first use and restarted if it dies.

The Salesforce Cases server (salesforce_case_server.py) logs in with this
app's Connected App (SF_LOGIN_URL / SF_CLIENT_ID / SF_CLIENT_SECRET) when
those are set, otherwise with the Salesforce CLI login for SF_ORG_ALIAS.
"""

import asyncio
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.tools import StructuredTool

load_dotenv()

STARTUP_TIMEOUT_S = 120   # first npx run may download the package
CALL_TIMEOUT_S = 90


@dataclass
class ServerSpec:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    # Only expose these tools to the LLM (None = all). Keeps the prompt small
    # and hides destructive tools we do not want a chat user to reach.
    allow: set[str] | None = None
    prefix: str = ""


def _npx() -> str:
    return shutil.which("npx.cmd" if sys.platform == "win32" else "npx") or "npx"


def salesforce_spec() -> ServerSpec | None:
    alias = os.getenv("SF_ORG_ALIAS", "").strip()
    if not alias or os.getenv("SF_MCP_ENABLED", "1").lower() in {"0", "false", "no"}:
        return None
    # Opt-in: the official server takes about a minute to start and, for data,
    # only adds run_soql_query, which salesforce_case_server.py also provides.
    if os.getenv("SF_OFFICIAL_MCP", "0").lower() not in {"1", "true", "yes"}:
        return None
    toolsets = os.getenv("SF_MCP_TOOLSETS", "data").strip()
    # Default to the one read tool; the org-management tools only confuse a chat agent.
    allow_env = os.getenv("SF_MCP_TOOLS", "run_soql_query").strip()
    allow = {t.strip() for t in allow_env.split(",") if t.strip()} or None
    if allow_env == "*":
        allow = None
    return ServerSpec(
        name="salesforce",
        command=_npx(),
        args=["-y", "@salesforce/mcp", "--orgs", alias, "--toolsets", toolsets, "--no-telemetry"],
        allow=allow,
        prefix="sf_",
    )


def connected_app_configured() -> bool:
    return all(os.getenv(k, "").strip() for k in ("SF_LOGIN_URL", "SF_CLIENT_ID", "SF_CLIENT_SECRET"))


def salesforce_cases_spec() -> ServerSpec | None:
    """Our own small server (salesforce_case_server.py) for creating/updating Cases.
    Enabled by either a Connected App (SF_LOGIN_URL + SF_CLIENT_ID + SF_CLIENT_SECRET)
    or a Salesforce CLI login alias (SF_ORG_ALIAS)."""
    alias = os.getenv("SF_ORG_ALIAS", "").strip()
    if os.getenv("SF_MCP_ENABLED", "1").lower() in {"0", "false", "no"}:
        return None
    if not alias and not connected_app_configured():
        return None
    here = os.path.dirname(os.path.abspath(__file__))
    return ServerSpec(
        name="salesforce-cases",
        command=sys.executable,
        args=[os.path.join(here, "salesforce_case_server.py")],
        prefix="sf_",
    )


class _Bridge:
    """Owns one MCP server process and the event-loop thread that talks to it."""

    def __init__(self, spec: ServerSpec):
        self.spec = spec
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name=f"mcp-{spec.name}", daemon=True)
        self.thread.start()
        self.session = None
        self._stack = None
        self._lock = threading.Lock()
        self.tool_specs = []

    # ---- lifecycle -------------------------------------------------------

    async def _open(self):
        from contextlib import AsyncExitStack
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self.spec.command,
            args=self.spec.args,
            env={**os.environ, **self.spec.env},
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        self._stack, self.session = stack, session
        self.tool_specs = [
            t for t in listed.tools if self.spec.allow is None or t.name in self.spec.allow
        ]

    async def _close(self):
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except Exception:
                pass
        self._stack, self.session = None, None

    def _run(self, coro, timeout):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout=timeout)

    def connect(self):
        with self._lock:
            if self.session is None:
                t0 = time.perf_counter()
                self._run(self._open(), STARTUP_TIMEOUT_S)
                print(f"[mcp:{self.spec.name}] connected in {time.perf_counter() - t0:.1f}s, "
                      f"{len(self.tool_specs)} tools: {', '.join(t.name for t in self.tool_specs)}")

    def reconnect(self):
        with self._lock:
            self._run(self._close(), 30)
        self.connect()

    # ---- calls -----------------------------------------------------------

    async def _call(self, name: str, arguments: dict):
        result = await self.session.call_tool(name, arguments)
        parts = []
        for item in result.content:
            text = getattr(item, "text", None)
            parts.append(text if text is not None else json.dumps(item.model_dump(), default=str))
        text = "\n".join(parts).strip() or "(no output)"
        if getattr(result, "isError", False):
            return f"Tool error: {text}"
        return text

    def call(self, name: str, arguments: dict) -> str:
        try:
            self.connect()
            return self._run(self._call(name, arguments), CALL_TIMEOUT_S)
        except Exception as first:
            # Server process may have died (e.g. CLI token refresh crash). One retry on a fresh process.
            try:
                self.reconnect()
                return self._run(self._call(name, arguments), CALL_TIMEOUT_S)
            except Exception as second:
                return f"Tool error ({self.spec.name}): {type(second).__name__}: {second} (first failure: {first})"


def _make_tool(bridge: _Bridge, spec) -> StructuredTool:
    schema = dict(getattr(spec, "input_schema", None) or getattr(spec, "inputSchema", None) or {"type": "object", "properties": {}})
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    name = spec.name

    def _func(**kwargs):
        return bridge.call(name, kwargs)

    return StructuredTool.from_function(
        func=_func,
        name=f"{bridge.spec.prefix}{name}",
        description=(spec.description or name)[:1024],
        args_schema=schema,
    )


_BRIDGES: dict[str, _Bridge] = {}


def _shutdown():
    for bridge in _BRIDGES.values():
        try:
            bridge._run(bridge._close(), 10)
        except Exception:
            pass


import atexit
atexit.register(_shutdown)


def load_mcp_tools() -> list[StructuredTool]:
    """Connect to every configured MCP server and return its tools. Never raises:
    a server that fails to start just contributes no tools, and the app runs on."""
    specs = [s for s in (salesforce_cases_spec(), salesforce_spec()) if s is not None]
    tools: list[StructuredTool] = []
    for spec in specs:
        try:
            bridge = _BRIDGES.get(spec.name) or _Bridge(spec)
            _BRIDGES[spec.name] = bridge
            bridge.connect()
            tools.extend(_make_tool(bridge, t) for t in bridge.tool_specs)
        except Exception as exc:
            print(f"[mcp:{spec.name}] disabled: {type(exc).__name__}: {exc}")
    return tools
