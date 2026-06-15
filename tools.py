"""
Standalone tool functions for the research agent.

Each tool is a plain Python function with a corresponding JSON schema
(used to describe it to the OpenAI-compatible Chat Completions API).
"""

import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
}

MAX_PAGE_CHARS = 8000  # truncate fetched page text so we don't blow the context window


def web_search(query: str, num_results: int = 5) -> str:
    """Search the web using the Serper.dev API and return a formatted list
    of results (title, link, snippet)."""
    if not SERPER_API_KEY:
        return "Error: SERPER_API_KEY is not set."

    try:
        resp = requests.post(
            SERPER_URL,
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": num_results},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return f"Error performing web search: {e}"

    results = []

    # "Answer box" / featured snippet, if present
    if "answerBox" in data:
        ab = data["answerBox"]
        snippet = ab.get("answer") or ab.get("snippet") or ""
        if snippet:
            results.append(f"[Answer box] {snippet}")

    for item in data.get("organic", [])[:num_results]:
        title = item.get("title", "")
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        results.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}")

    if not results:
        return f"No search results found for query: {query!r}"

    return "\n\n".join(results)


def web_fetch(url: str) -> str:
    """Fetch a web page and return its main text content, with HTML
    stripped out and whitespace collapsed."""
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        return f"Error fetching {url}: {e}"

    content_type = resp.headers.get("Content-Type", "")
    if "html" not in content_type and "text" not in content_type:
        return f"Error: unsupported content type '{content_type}' for {url}"

    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS] + "\n\n[... content truncated ...]"

    if not text:
        return f"Error: no readable text content found at {url}"

    return f"Content from {url}:\n\n{text}"


# JSON schemas describing these tools to the model. Matches the OpenAI
# "function" tool format used by the Chat Completions API.
WEB_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for a query and return a list of relevant results "
                "(title, URL, and snippet). Use this to discover sources before "
                "reading them in full with web_fetch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max ~10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch the full text content of a web page given its URL. "
                "Use this to read a page found via web_search in full so you "
                "can extract details and quotes for your answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the page to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

# Dispatch table: tool name -> python callable
WEB_TOOL_FUNCTIONS = {
    "web_search": web_search,
    "web_fetch": web_fetch,
}


if __name__ == "__main__":
    # Quick manual smoke test
    print(web_search("OpenAI GPT-5 release date")[:1000])
    print("\n---\n")
    print(web_fetch("https://example.com")[:1000])