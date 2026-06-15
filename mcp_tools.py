"""
mcp_tools.py

Connects to the AlphaXiv MCP server (https://www.alphaxiv.org/docs/mcp)
and exposes its tools (discover_papers, get_paper_content,
answer_pdf_queries, read_files_from_github_repository) in the same
"schema + dispatch" shape as tools.py, so agent.py can treat web tools
and MCP tools uniformly.

The real AlphaXiv MCP endpoint (https://api.alphaxiv.org/mcp/v1) requires
OAuth 2.0 and explicitly does not support being hit directly from a plain
HTTP client in a script. The documented workaround is to run it through
the `mcp-remote` bridge (a local stdio<->HTTP proxy that handles the OAuth
flow, opening a browser the first time):

    npx mcp-remote https://api.alphaxiv.org/mcp/v1

So this module spawns that bridge as a subprocess and talks to it over
stdio using the official `mcp` Python SDK.

The MCP ClientSession must be entered/exited from the SAME asyncio task
(mixing tasks causes "cancel scope in a different task" errors). To avoid
that, we run a single long-lived background task that owns the session
for the whole program, and other coroutines talk to it via queues.

If `mcp`/`npx`/the server are unavailable, this module degrades
gracefully: schema fetching falls back to static schemas, and tool calls
return an error string the model can see and work around.
"""

import os
import json
import asyncio
import shutil
from dotenv import load_dotenv

load_dotenv()

ALPHAXIV_MCP_URL = os.environ.get("ALPHAXIV_MCP_URL", "https://api.alphaxiv.org/mcp/v1")

ALPHAXIV_TOOL_NAMES = {
    "discover_papers",
    "get_paper_content",
    "answer_pdf_queries",
    "read_files_from_github_repository",
}

# Fallback schemas, used if we can't reach the live MCP server to list
# tools. These mirror AlphaXiv's documented tool signatures (see
# https://www.alphaxiv.org/docs/mcp) so the agent loop still works
# (offering these tools) even if listing live tools fails.
_FALLBACK_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "discover_papers",
            "description": (
                "Discover and rank candidate academic papers for a research "
                "topic (runs keyword + embedding search, with optional "
                "multi-round follow-up searches). Returns up to 15 papers "
                "with title, date, organizations, abstract preview, and "
                "arXiv ID. Use this to find papers before reading any in full."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "3-4 concise keyword terms for exact-name, acronym, "
                            "method, benchmark, author, or title matching."
                        ),
                    },
                    "question": {
                        "type": "string",
                        "description": (
                            "A detailed semantic description of the papers that "
                            "would best answer the request, including key "
                            "concepts, methods, applications, and related terms."
                        ),
                    },
                    "difficulty": {
                        "type": "number",
                        "description": (
                            "1-10 estimate of retrieval effort warranted. Higher "
                            "values take longer but trigger multi-round searches."
                        ),
                    },
                },
                "required": ["keywords", "question", "difficulty"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_content",
            "description": (
                "Get the content of an arXiv/alphaXiv paper as text. By default "
                "returns a structured AI-generated report optimized for LLM "
                "consumption; falls back to full extracted text if no report "
                "is available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "An arXiv or alphaXiv URL, e.g. "
                            "'https://arxiv.org/abs/2307.12307'."
                        ),
                    },
                    "fullText": {
                        "type": "boolean",
                        "description": (
                            "If true, return the raw full extracted text instead "
                            "of the AI-generated report. Default false."
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer_pdf_queries",
            "description": (
                "Return the page-level content of a single PDF that is relevant "
                "to one or more queries, as XML "
                "(<paper id=...><page num=...>...</page></paper>) so citations "
                "can be built directly from the page text. Batch all questions "
                "about one paper into a single call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "A PDF/arXiv/alphaXiv URL.",
                    },
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "One or more brief descriptions of the information "
                            "you're looking for in the paper."
                        ),
                    },
                },
                "required": ["url", "queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_files_from_github_repository",
            "description": (
                "Read files or directories from a paper's GitHub repository. "
                "Reading '/' returns the complete file tree and top-level "
                "files; reading a directory fetches all files in it; reading "
                "a file returns its contents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "githubUrl": {
                        "type": "string",
                        "description": "URL of the paper's codebase repository.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Path to the file or directory, or '/' for an overview.",
                    },
                },
                "required": ["githubUrl", "path"],
            },
        },
    },
]


class _AlphaXivClient:
    """Owns a single long-lived MCP ClientSession, run inside its own
    background asyncio task so enter/exit always happens in that task."""

    def __init__(self, url: str):
        self.url = url
        self._task = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._session = None
        self._init_error = None
        # request/response plumbing for cross-task calls
        self._lock = asyncio.Lock()

    async def _run(self):
        try:
            if shutil.which("npx") is None:
                raise RuntimeError(
                    "npx not found. Install Node.js (https://nodejs.org) so "
                    "the AlphaXiv MCP bridge (`npx mcp-remote ...`) can run."
                )

            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "mcp-remote", self.url],
            )

            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()
        except Exception as e:
            self._init_error = e
            self._ready.set()

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        await self._ready.wait()
        if self._init_error is not None:
            raise self._init_error

    async def list_tools(self):
        await self.start()
        async with self._lock:
            if self._session is None:
                raise RuntimeError("AlphaXiv MCP session is not available")
            return await self._session.list_tools()

    async def call_tool(self, name, args):
        await self.start()
        async with self._lock:
            if self._session is None:
                raise RuntimeError("AlphaXiv MCP session is not available")
            return await self._session.call_tool(name, args)

    async def stop(self):
        if self._task is not None:
            self._stop.set()
            await self._task


_client = None
_live_schemas = None


async def _get_client() -> _AlphaXivClient:
    global _client
    if _client is None or _client._session is None:
        _client = _AlphaXivClient(ALPHAXIV_MCP_URL)
        await _client.start()
    return _client


async def ensure_alphaxiv_schemas():
    """Return OpenAI-style function-tool schemas for the AlphaXiv MCP
    tools we care about.

    Tries to fetch live schemas from the MCP server (so descriptions /
    parameters always match the real server); falls back to the static
    schemas above if that fails. Cached after first call.
    """
    global _live_schemas

    if _live_schemas is not None:
        return _live_schemas

    try:
        client = await _get_client()
        tools_result = await client.list_tools()

        schemas = []
        for tool in tools_result.tools:
            if tool.name not in ALPHAXIV_TOOL_NAMES:
                continue
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": tool.inputSchema
                        or {"type": "object", "properties": {}},
                    },
                }
            )

        if schemas:
            _live_schemas = schemas
            return _live_schemas
    except Exception:
        # MCP bridge unreachable, npx missing, OAuth not completed, etc.
        pass

    _live_schemas = _FALLBACK_SCHEMAS
    return _live_schemas


async def call_alphaxiv_tool(name: str, args: dict) -> str:
    """Call a tool on the AlphaXiv MCP server and return its result as a
    string suitable for feeding back to the model."""
    try:
        client = await _get_client()
        result = await client.call_tool(name, args)
    except Exception as e:
        return f"Error calling AlphaXiv MCP tool '{name}': {e}"

    if getattr(result, "isError", False):
        return f"AlphaXiv tool '{name}' returned an error: {_stringify_content(result.content)}"

    return _stringify_content(result.content)


def _stringify_content(content) -> str:
    """MCP tool results are a list of content blocks (text, image, etc.).
    Flatten text blocks into a single string; describe non-text blocks."""
    if not content:
        return ""

    parts = []
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            parts.append(block.text)
        else:
            try:
                parts.append(json.dumps(block.model_dump(), default=str))
            except Exception:
                parts.append(str(block))

    return "\n".join(parts)