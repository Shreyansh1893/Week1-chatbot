# Build your own Perplexity

A terminal research agent: ask a question, it searches the web (Serper),
reads pages in full (requests + BeautifulSoup), optionally searches
academic papers via the AlphaXiv MCP server, and synthesises a cited
answer -- all inside a Textual TUI.

## Files

- `tools.py` -- `web_search` (Serper.dev) and `web_fetch` (requests +
  BeautifulSoup), plus their OpenAI tool schemas.
- `mcp_tools.py` -- connects to the AlphaXiv MCP server
  (https://www.alphaxiv.org/docs/mcp) over streamable HTTP and exposes
  `discover_papers` / `get_paper_content`.
- `agent.py` -- `ResearchAgent`, the tool-calling loop. Also contains a
  bare single-call function (`--step1`) and a CLI chat loop.
- `app.py` -- the Textual TUI built around `ResearchAgent`.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env: add OPENAI_API_KEY, SERPER_API_KEY, ALPHAXIV_MCP_URL
```

## Usage

```bash
# Step 1: sanity-check the base API call
python agent.py --step1 "What is the capital of France?"

# Step 2/3: agent loop with web_search + web_fetch, from the terminal
python agent.py

# Step 5: full Textual TUI (web + AlphaXiv papers)
python app.py
```

## Notes

- AlphaXiv tools are loaded lazily on first question; if the MCP server
  is unreachable, the agent falls back to static schemas for those tools
  and still works fine with web_search/web_fetch.
- `MAX_PAGE_CHARS` in `tools.py` truncates fetched pages to keep the
  context window manageable -- tune as needed.
- `AGENT_MODEL` env var controls which model is used (default
  `gpt-4o-mini`); must support tool calling.
