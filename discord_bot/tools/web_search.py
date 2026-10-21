import os
import re
import html
import requests


def _clean_html(text: str) -> str:
    """Strips tags (e.g. <br>, <b>) and decodes entities (e.g. &amp;) that
    some SearXNG engines leave in title/content even under JSON format."""
    if not text:
        return ""
    text = re.sub(r"<[^<]+?>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()

# Matches one result block as web_search() formats it below: a bold numbered
# title line immediately followed by the URL on its own line. Keep this in sync
# with the formatting in web_search().
_RESULT_RE = re.compile(r"^\*\*\d+\. (.*)\*\*\n(https?://\S+)$", re.MULTILINE)


# NOTE: the helpers below are underscore-prefixed on purpose. tool_registry
# exposes every public function defined in a tools/ module to the LLM as a
# callable tool, and these are for bot.py's use only.

def _collect_sources(output: str, collected: list) -> None:
    """Parses the (title, url) pairs out of a web_search() result and appends
    any URL not already in `collected` (a list of (title, url) tuples), in order."""
    seen = {url for _, url in collected}
    for title, url in _RESULT_RE.findall(output or ""):
        if url not in seen:
            seen.add(url)
            collected.append((title, url))


def _format_sources(sources: list) -> str:
    """Formats collected (title, url) pairs as a Discord "Sources" list in small
    subtext (every line needs its own "-# " prefix). Each URL is wrapped in <>
    so Discord doesn't unfurl it into an embed. Returns "" if there are no
    sources."""
    if not sources:
        return ""
    lines = ["-# **Sources**"]
    for i, (title, url) in enumerate(sources, start=1):
        if len(title) > 80:
            title = title[:79].rstrip() + "…"
        title = re.sub(r"([*_~`|\\])", r"\\\1", title)  # don't let a title's markdown leak
        lines.append(f"-# {i}. {title} — <{url}>")
    return "\n".join(lines)


# Point this at your SearXNG instance's /search endpoint, e.g.
# http://localhost:8080/search or http://192.168.1.50:8888/search
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080/search")


def web_search(query: str, num_results: int = 5) -> str:
    """
    Searches the web via a local SearXNG instance and returns the top
    results (title, url, and a short snippet). Use this tool whenever the
    user asks you to look something up, search the internet, find current
    information, check facts, or answer anything that needs up-to-date
    knowledge you might not have.
    """
    params = {
        "q": query,
        "format": "json",
    }

    try:
        response = requests.get(SEARXNG_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        return (
            "⚠️ SearXNG didn't return JSON — check that `json` is listed under "
            "`search.formats` in your instance's settings.yml."
        )
    except Exception as e:
        return f"⚠️ Web search failed: {e}"

    results = data.get("results", [])
    if not results:
        return f"No results found for '{query}'."

    num_results = max(1, min(num_results, 10))
    lines = [f"🔎 **Search results for:** {query}"]
    for i, r in enumerate(results[:num_results], start=1):
        title = _clean_html(r.get("title", "Untitled"))
        url = r.get("url", "")
        content = _clean_html(r.get("content", ""))
        if len(content) > 220:
            content = content[:220].rsplit(" ", 1)[0] + "…"
        lines.append(f"\n**{i}. {title}**\n{url}\n{content}")

    return "\n".join(lines)
