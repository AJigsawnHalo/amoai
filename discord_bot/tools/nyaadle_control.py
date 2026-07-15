import re
import subprocess
from pathlib import Path

# Adjust to wherever the nyaadle binary actually lives, e.g. via `which nyaadle`
NYAADLE_BIN = "/home/elskiee/.cargo/bin/nyaadle"
NYAADLE_LOG = Path("~/.config/nyaadle/nyaadle.log").expanduser()

# Matches lines like:
# 2026-Jul-14 Tue 04:36:26 [INFO] Downloaded [ASW] Otome Kaijuu Carameliser - 02 [1080p HEVC x265 10Bit][AAC]
DOWNLOAD_LINE = re.compile(r"^(\S+ \S+ \S+) \[INFO\] Downloaded (.+)$")


def nyaadle_status() -> str:
    """
    Lists nyaadle's configured RSS feeds. Use this tool when the user asks
    what feeds nyaadle is watching, wants to check its feed config, or asks
    "what is nyaadle tracking". Read-only — does not trigger any downloads.
    """
    try:
        result = subprocess.run(
            [NYAADLE_BIN, "feeds"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = result.stdout.strip() or result.stderr.strip()
        return output or "No feeds configured, or nyaadle returned no output."
    except subprocess.TimeoutExpired:
        return "❌ nyaadle feeds check timed out."
    except Exception as e:
        return f"❌ Failed to query nyaadle: {e}"


def nyaadle_recent_downloads(count: int = 10) -> str:
    """
    Shows the most recently downloaded items from nyaadle's log file, newest
    first. Use this tool when the user asks what nyaadle has downloaded
    recently, wants to see the latest items grabbed, or asks "what did
    nyaadle get". Read-only — just reads the log file.
    """
    if not NYAADLE_LOG.exists():
        return f"❌ Log file not found at {NYAADLE_LOG}"

    try:
        with open(NYAADLE_LOG, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except OSError as e:
        return f"❌ Failed to read nyaadle log: {e}"

    items = []
    for line in lines:
        match = DOWNLOAD_LINE.match(line.strip())
        if match:
            timestamp, item = match.groups()
            items.append((timestamp, item))

    if not items:
        return "No download entries found in the log yet."

    count = max(1, min(count, 50))  # keep it sane regardless of what the model passes
    recent = items[-count:][::-1]  # most recent N, newest first
    formatted = "\n".join(f"• `{ts}` — {item}" for ts, item in recent)
    return f"**📥 Last {len(recent)} nyaadle download(s):**\n{formatted}"


def nyaadle_check_now() -> str:
    """
    Manually triggers nyaadle's full feed check-and-download run immediately,
    instead of waiting for its scheduled cron run. This WILL download any new
    matching torrents and modify the watchlist database. Use this tool only
    when the user explicitly asks to force or trigger a nyaadle run right now.
    """
    try:
        result = subprocess.run(
            [NYAADLE_BIN],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout.strip()
        errors = result.stderr.strip()
        if result.returncode != 0:
            return f"❌ nyaadle run failed:\n```\n{errors or output}\n```"
        trimmed = output[-1500:] if output else "no output"
        return f"✅ nyaadle run complete.\n```\n{trimmed}\n```"
    except subprocess.TimeoutExpired:
        return "❌ nyaadle run timed out after 120s — it may still be running in the background."
    except Exception as e:
        return f"❌ Failed to run nyaadle: {e}"
