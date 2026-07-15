import re
import html
import requests

# Cap how much text gets handed to the LLM per page — a full article can
# easily blow past a local model's context window otherwise.
MAX_CHARS = 6000


def fetch_page(url: str) -> str:
    """
    Fetches a specific webpage by URL and returns its readable text content,
    with HTML/scripts/styles stripped out. Use this any time you need to
    read the actual content of a page — after a web_search when a snippet
    isn't enough, when the user pastes or mentions a specific link, or when
    the user asks you to check, summarize, or read something from a URL.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AmoaiBot/1.0)"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return f"⚠️ Failed to fetch page: {e}"

    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type and "text" not in content_type:
        return f"⚠️ Unsupported content type ({content_type or 'unknown'}) — can't extract readable text."

    raw = response.text

    # Drop script/style blocks whole so their code doesn't leak into the text
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^<]+?>", " ", raw)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "⚠️ Page fetched but no readable text could be extracted (it may be JS-rendered)."

    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0] + "…"

    return f"📄 **Content from {url}:**\n{text}"
