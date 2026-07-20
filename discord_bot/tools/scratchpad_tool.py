"""
scratchpad_tool.py

A lightweight scratchpad for short, freeform things the user wants jotted
down directly in chat — grocery items, a password, a thought, a link, a
one-liner to remember. This is NOT the indexed document/knowledge base;
that's rag_knowledge.py (index_knowledge_base / search_knowledge), which
covers files the user explicitly indexed.

--- Boundary with rag_knowledge.py ---
The word "notes" belongs to rag_knowledge.py, not this module. If the user
says "based on my notes", "search my notes", "what do my notes say about
X" — that ALWAYS means their INDEXED files (search_knowledge), never
anything in here, even if something was jotted down here earlier in the
same conversation. This module is for jotting something down fresh in the
conversation itself, e.g. "jot this down", "quick scratchpad entry: call
the plumber tomorrow" — content that doesn't exist as a file anywhere.
Deliberately named without the word "note" in any function name, to avoid
being pulled in by phrases that are actually about the indexed knowledge
base.

--- Boundary with reminder_tool.py ---
This module never schedules anything and has no concept of time. If the
request includes a delay, a clock time, a date, or "when I get home/arrive
at X", that is set_reminder in reminder_tool.py, not jot_down — even if the
user phrases it as "remember to X at 5pm". A bare "remember this: <thing>"
with no time attached is the only case where "remember" means jot_down.

--- Boundary with the bot's own background memory ---
Durable facts about the user themselves (name, job, preferences, ongoing
routines) are captured automatically by a separate process after every
message. Don't call jot_down to record a fact about the user — it's for
freeform content the user explicitly wants jotted down verbatim, not
personal facts the bot infers about them.

Public tool functions (auto-discovered by bot.py's register_tools()):
    - jot_down(content, tag)
    - list_scratchpad(tag)
    - delete_scratchpad_entry(identifier)

Everything prefixed with "_" is a private helper and will NOT be
registered as an LLM-callable tool.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

# Sibling to reminders.json, same storage pattern as reminder_tool.py
SCRATCHPAD_FILE = Path(__file__).resolve().parent.parent / "scratchpad.json"


def _load_entries() -> list:
    if not SCRATCHPAD_FILE.exists():
        return []
    try:
        with open(SCRATCHPAD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        print(f"[SCRATCHPAD TOOL] scratchpad.json contained {type(data).__name__}, not a list — resetting.")
        return []
    return data


def _save_entries(entries: list):
    try:
        with open(SCRATCHPAD_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[SCRATCHPAD TOOL] Failed to save scratchpad: {e}")


def jot_down(user_id: str, content: str, tag: str = None) -> str:
    """
    Jots down a short freeform entry straight from the conversation, saved
    with no time trigger and no source file. Use this for: "jot this down:
    the wifi password changed", "add to scratchpad: call the plumber
    tomorrow", "quick note: <thing>". Also covers a bare "remember this:
    <thing>" that has no clock time, delay, or arrival event attached — if
    a time or arrival IS attached ("remind me at 5pm", "when I get home"),
    use set_reminder instead, not this tool. And if the user says "notes"
    ("based on my notes", "search my notes"), that means the indexed
    knowledge base — use search_knowledge instead, not this tool.

    :param content: The text of the entry itself.
    :param tag: Optional short label to group related entries (e.g. 'shopping', 'ideas'). Omit if the user doesn't give one.
    """
    if not content or not content.strip():
        return "❌ Error: entry content can't be empty."

    new_entry = {
        "user_id": str(user_id),
        "content": content.strip(),
        "tag": tag.strip().lower() if tag else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    entries = _load_entries()
    entries.append(new_entry)
    _save_entries(entries)

    tag_note = f" (tagged '{new_entry['tag']}')" if new_entry["tag"] else ""
    return f"✅ Jotted down{tag_note}: {new_entry['content']}"


def list_scratchpad(user_id: str, tag: str = None) -> str:
    """
    Lists this user's scratchpad entries — the quick jotted-down ones from
    jot_down. This is NOT the RAG knowledge base and does NOT cover indexed
    documents. Optionally filter to a single tag. Each entry is numbered so
    the number can be passed to delete_scratchpad_entry.

    Do NOT use this tool for "based on my notes" or any phrase containing
    the word "notes" — that always means search_knowledge in
    rag_knowledge.py instead, even if something was jotted down here
    earlier in the conversation.

    :param tag: Optional. Only show entries saved under this tag.
    """
    entries = _load_entries()
    mine = [n for n in entries if str(n.get("user_id")) == str(user_id)]
    if tag:
        tag = tag.strip().lower()
        mine = [n for n in mine if n.get("tag") == tag]

    if not mine:
        return "Your scratchpad is empty." if not tag else f"You have no scratchpad entries tagged '{tag}'."

    lines = []
    for i, n in enumerate(mine, start=1):
        tag_str = f" _[{n['tag']}]_" if n.get("tag") else ""
        lines.append(f"{i}. {n.get('content', '')}{tag_str}")

    return "Here's your scratchpad:\n" + "\n".join(lines) + "\n\nUse delete_scratchpad_entry with the number shown to remove one."


def delete_scratchpad_entry(user_id: str, identifier: str) -> str:
    """
    Deletes one of this user's scratchpad entries. Matched either by its
    1-based position in the list returned by list_scratchpad, or by a
    snippet of text from the entry's content. This is NOT the RAG knowledge
    base and has no effect on indexed documents.

    :param identifier: The number shown by list_scratchpad (e.g. '2'), or a piece of the entry's text to match against.
    """
    entries = _load_entries()
    mine_indices = [idx for idx, n in enumerate(entries) if str(n.get("user_id")) == str(user_id)]

    if not mine_indices:
        return "Your scratchpad is empty — nothing to delete."

    identifier = identifier.strip()
    target_idx = None

    if identifier.isdigit():
        pos = int(identifier) - 1
        if 0 <= pos < len(mine_indices):
            target_idx = mine_indices[pos]
    else:
        for idx in mine_indices:
            if identifier.lower() in entries[idx].get("content", "").lower():
                target_idx = idx
                break

    if target_idx is None:
        return f"❌ Couldn't find a scratchpad entry matching '{identifier}'. Try list_scratchpad to see the numbered list."

    removed = entries.pop(target_idx)
    _save_entries(entries)
    return f"🗑️ Removed from scratchpad: {removed.get('content', '')}"
