"""
agent.py

The core "Perplexity-style" research agent.

Step 1: a single API call (base case, used as a sanity check) -- see
        _step1_single_call() and `python agent.py --step1 "..."`.
Step 2/3: a full agent loop wired up with web_search and web_fetch,
          runnable from the command line -- see ResearchAgent and the
          CLI entry point below.
Step 4: extended with AlphaXiv MCP tools (discover_papers,
        get_paper_content), added lazily in ResearchAgent.ask().

The ResearchAgent class is UI-agnostic so the same object can be driven
from the command line or from the Textual TUI in app.py.
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

from tools import WEB_TOOL_SCHEMAS, WEB_TOOL_FUNCTIONS
from mcp_tools import ensure_alphaxiv_schemas, call_alphaxiv_tool, ALPHAXIV_TOOL_NAMES

load_dotenv()

MODEL = os.environ.get("AGENT_MODEL", "openrouter/free")
BASE_URL = os.environ.get("OPENAI_BASE_URL")  # set to OpenRouter's URL to use OpenRouter
MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "4096"))


def _make_client() -> OpenAI:
    if BASE_URL:
        return OpenAI(base_url=BASE_URL)
    return OpenAI()

SYSTEM_PROMPT = """\
You are a meticulous research assistant, similar to Perplexity AI.

Given a user's research question, your job is to:
1. Search the web (web_search) for relevant, up-to-date sources.
2. Read the most promising pages in full (web_fetch) before relying on them.
3. If the question is technical, scientific, or about AI/ML research, also
   search academic papers (discover_papers) and read relevant ones in full
   (get_paper_content).
4. Synthesise everything into a clear, well-organised answer.

Rules:
- Always ground factual claims in the sources you actually fetched/read.
  Do not rely on snippets alone for important claims -- fetch the page.
- Cite your sources inline using bracketed numbers like [1], [2], and end
  your answer with a "Sources" section listing each numbered source as a
  URL (and paper title/ID for academic papers).
- Prefer 2-5 high-quality, diverse sources over many shallow ones.
- If sources disagree, point that out explicitly.
- If you cannot find a good answer, say so plainly rather than guessing.
- Be concise but complete. Use short paragraphs or bullet points.
"""


def _step1_single_call(question: str) -> str:
    """Step 1: the simplest possible version -- a single API call,
    no tools, no loop. Useful as a smoke test that the API key/model work."""
    client = _make_client()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        max_tokens=MAX_TOKENS,
    )
    return resp.choices[0].message.content


class ResearchAgent:
    """An agent loop: send messages + tool schemas to the model, execute
    any tool calls it requests, feed results back, and repeat until the
    model produces a final answer (no more tool calls)."""

    def __init__(self, on_event=None, use_alphaxiv: bool = True, max_turns: int = 12):
        """
        on_event: optional callback(event_type: str, data: dict) called for
                  each notable step (e.g. "tool_call", "tool_result",
                  "assistant_message"). Used by the TUI to show progress.
        use_alphaxiv: whether to include the AlphaXiv MCP tools.
        max_turns: safety cap on the number of tool-call round trips.
        """
        self.client = _make_client()
        self.on_event = on_event or (lambda event_type, data: None)
        self.max_turns = max_turns
        self.use_alphaxiv = use_alphaxiv

        self.tool_schemas = list(WEB_TOOL_SCHEMAS)
        self.tool_functions = dict(WEB_TOOL_FUNCTIONS)
        self._alphaxiv_loaded = False

        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _emit(self, event_type: str, data: dict):
        self.on_event(event_type, data)

    async def _ensure_alphaxiv(self):
        """Fetch and append the AlphaXiv MCP tool schemas (once)."""
        if self._alphaxiv_loaded or not self.use_alphaxiv:
            return
        try:
            schemas = await ensure_alphaxiv_schemas()
            self.tool_schemas += schemas
        except Exception as e:
            self._emit("warning", {"message": f"AlphaXiv MCP unavailable: {e}"})
        self._alphaxiv_loaded = True

    async def _execute_tool(self, name: str, args: dict) -> str:
        """Run a single tool call (web tool or AlphaXiv MCP tool)."""
        self._emit("tool_call", {"name": name, "args": args})

        try:
            if name in self.tool_functions:
                fn = self.tool_functions[name]
                # Web tools are plain sync functions; run off the event loop
                result = await asyncio.to_thread(fn, **args)
            elif name in ALPHAXIV_TOOL_NAMES:
                result = await call_alphaxiv_tool(name, args)
            else:
                result = f"Error: unknown tool '{name}'"
        except Exception as e:  # noqa: BLE001 -- surface tool errors to the model
            result = f"Error running tool '{name}': {e}"

        self._emit("tool_result", {"name": name, "args": args, "result": result})
        return result

    async def _create_completion(self, max_retries: int = 3):
        """Call the chat completions API, retrying on transient 429
        rate-limit errors from OpenRouter free-tier providers."""
        for attempt in range(max_retries):
            try:
                return await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=MODEL,
                    messages=self.messages,
                    tools=self.tool_schemas,
                    max_tokens=MAX_TOKENS,
                )
            except RateLimitError as e:
                wait = 5
                # Try to read OpenRouter's suggested retry delay
                try:
                    wait = e.response.json()["error"]["metadata"].get(
                        "retry_after_seconds", wait
                    )
                except Exception:
                    pass
                wait = float(wait) + 1
                if attempt == max_retries - 1:
                    raise
                self._emit(
                    "warning",
                    {
                        "message": (
                            f"Rate limited by provider, retrying in "
                            f"{wait:.0f}s (attempt {attempt + 1}/{max_retries})..."
                        )
                    },
                )
                await asyncio.sleep(wait)

    async def ask(self, question: str) -> str:
        """Run the full agent loop for a single user question and return
        the final synthesised answer."""
        await self._ensure_alphaxiv()

        self.messages.append({"role": "user", "content": question})
        self._emit("user_message", {"content": question})

        for _ in range(self.max_turns):
            response = await self._create_completion()

            choice = response.choices[0]
            message = choice.message

            # Append the assistant message (with any tool_calls) to history.
            self.messages.append(message.model_dump(exclude_none=True))

            tool_calls = message.tool_calls or []

            if not tool_calls:
                content = message.content or ""
                self._emit("assistant_message", {"content": content})
                return content

            # Execute every requested tool call, then loop again.
            for tool_call in tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                result = await self._execute_tool(name, args)

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

        return (
            "Sorry, I ran out of reasoning steps before reaching a final "
            "answer. Try narrowing your question."
        )


async def _cli_main():
    print("Build-your-own-Perplexity agent (CLI mode). Ctrl-C to exit.\n")

    def on_event(event_type, data):
        if event_type == "tool_call":
            print(f"\n[tool call] {data['name']}({data['args']})")
        elif event_type == "tool_result":
            preview = str(data["result"])[:300].replace("\n", " ")
            print(f"[tool result] {preview}...\n")
        elif event_type == "warning":
            print(f"\n[warning] {data['message']}")

    agent = ResearchAgent(on_event=on_event)

    while True:
        try:
            question = input("\nResearch question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        answer = await agent.ask(question)
        print("\n=== ANSWER ===\n")
        print(answer)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--step1":
        # python agent.py --step1 "your question"
        q = " ".join(sys.argv[2:]) or "What is the capital of France?"
        print(_step1_single_call(q))
    else:
        asyncio.run(_cli_main())