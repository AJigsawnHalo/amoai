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
