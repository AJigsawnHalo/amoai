import os
import sys
import re
import json
import time
import fcntl
import argparse
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from contextlib import contextmanager
import requests
from dotenv import load_dotenv, find_dotenv

# Automatically crawls up parent directories to locate your central .env file
load_dotenv(find_dotenv())

# --- CONFIGURATION ---
SCRIPT_DIR = Path(__file__).resolve().parent
STATE_FILE = Path(os.getenv("LOG_MONITOR_STATE_FILE", SCRIPT_DIR / "state.json"))
LOCK_FILE = Path(os.getenv("LOG_MONITOR_LOCK_FILE", SCRIPT_DIR / "log_monitor.lock"))

# Reuses the email webhook if a dedicated one isn't set, so this works out of
# the box in the same .env as email_sorter/monitor.py.
WEBHOOK_URL = os.getenv("WEBHOOK_LOG") or os.getenv("WEBHOOK_EMAIL")

# journalctl priority names, low number = more severe
PRIORITY_LEVELS = {"emerg": 0, "alert": 1, "crit": 2, "err": 3,
                    "warning": 4, "notice": 5, "info": 6, "debug": 7}
MIN_PRIORITY = os.getenv("CRITICAL_MIN_PRIORITY", "err")  # anything <= this is critical

# Optional LLM classification (mirrors monitor.py's classify_email) for
# plain-text log sources that have no structured priority field.
USE_LLM_CLASSIFICATION = os.getenv("USE_LLM_CLASSIFICATION", "false").strip().lower() == "true"
MODEL_NAME = os.getenv("LOG_MONITOR_MODEL", "gpt-oss:20b-cloud")
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")

# Regex net for critical events regardless of source/priority. Extend via
# CRITICAL_KEYWORDS="foo,bar" in .env (comma-separated, matched case-insensitively).
DEFAULT_KEYWORDS = [
    r"out of memory", r"oom.?killer", r"kernel panic", r"segfault",
    r"disk full", r"no space left", r"read-only file ?system",
    r"traceback \(most recent call last\)", r"connection refused",
    r"failed to start", r"\bfailed\b.*\bservice\b", r"certificate.*expir",
    r"authentication failure", r"permission denied", r"unable to connect",
    r"deadlock", r"panic:", r"fatal error",
]
_extra_keywords = [k.strip() for k in os.getenv("CRITICAL_KEYWORDS", "").split(",") if k.strip()]
_all_keywords = DEFAULT_KEYWORDS + _extra_keywords
KEYWORD_PATTERN = re.compile("|".join(_all_keywords), re.IGNORECASE) if _all_keywords else None

# Sources to monitor. Override with LOG_SOURCES in .env as a JSON array, e.g.:
#   [{"type": "journalctl", "priority": "err"},
#    {"type": "journalctl", "unit": "nginx.service", "priority": "warning"},
#    {"type": "file", "path": "/var/log/nginx/error.log"}]
DEFAULT_SOURCES = [{"type": "journalctl", "priority": "err"}]


# --- STATE (cursor / offset tracking so cron/systemd never double-report) ---

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            print(f"Warning: could not parse {STATE_FILE}, starting fresh", file=sys.stderr)
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def get_sources() -> list:
    raw = os.getenv("LOG_SOURCES")
    if not raw:
        return DEFAULT_SOURCES
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Invalid LOG_SOURCES JSON in .env, falling back to default: {e}", file=sys.stderr)
        return DEFAULT_SOURCES


# --- LOCKING (prevents cron / systemd timer / bot-triggered runs overlapping) ---

@contextmanager
def lock_or_exit():
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another log_monitor run is already in progress; exiting.", file=sys.stderr)
        fd.close()
        sys.exit(0)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


# --- READING SOURCES ---

def read_journalctl(source: dict, state: dict, since_override: "str | None"):
    """Reads new journal entries since the last saved cursor. Uses journalctl's
    own --after-cursor mechanism so restarts never duplicate or skip lines."""
    unit = source.get("unit")
    priority = source.get("priority", MIN_PRIORITY)
    cursor_key = f"journalctl:{unit or 'system'}:{priority}"
    cursor = state.get(cursor_key)

    cmd = ["journalctl", "-o", "json", "--no-pager", "-p", priority]
    if unit:
        cmd += ["-u", unit]
    if since_override:
        cmd += ["--since", since_override]
    elif cursor:
        cmd += ["--after-cursor", cursor]
    else:
        # First-ever run: don't replay the whole journal, just recent history.
        cmd += ["--since", "10 minutes ago"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        print(f"journalctl read failed for {unit or 'system'}: {e}", file=sys.stderr)
        return [], cursor

    if result.returncode != 0 and result.stderr:
        print(f"journalctl warning ({unit or 'system'}): {result.stderr.strip()}", file=sys.stderr)

    entries = []
    last_cursor = cursor
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        last_cursor = rec.get("__CURSOR", last_cursor)
        entries.append({
            "source": unit or "system",
            "unit": rec.get("_SYSTEMD_UNIT", unit or "system"),
            "timestamp": rec.get("__REALTIME_TIMESTAMP"),
            "priority": rec.get("PRIORITY"),
            "message": rec.get("MESSAGE", ""),
        })
    return entries, (last_cursor if not since_override else cursor)


def read_file_source(source: dict, state: dict, since_override: "str | None"):
    """Tails a plain-text log file from the last saved byte offset. Resets to
    0 automatically if the file shrank (log rotation/truncation)."""
    path = source["path"]
    key = f"file:{path}"
    offset = 0 if since_override else state.get(key, 0)

    try:
        size = os.path.getsize(path)
    except OSError as e:
        print(f"Cannot stat log file {path}: {e}", file=sys.stderr)
        return [], state.get(key, 0)

    if size < offset:
        offset = 0  # rotated or truncated since last run

    entries = []
    new_offset = offset
    try:
        with open(path, "r", errors="ignore") as f:
            f.seek(offset)
            for line in f:
                line = line.rstrip("\n")
                if line:
                    entries.append({
                        "source": path, "unit": path,
                        "timestamp": None, "priority": None, "message": line,
                    })
            new_offset = f.tell()
    except OSError as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return [], offset

    return entries, (new_offset if not since_override else state.get(key, 0))


# --- CLASSIFICATION ---

def classify_log_llm(entry: dict) -> str:
    prompt = f"""Classify this system log line as exactly one word: "critical" or "ignore".
Source: {entry.get('unit') or entry.get('source')}
Message: {entry.get('message', '')[:500]}
Rules:
- critical: service crashes/failures, security issues, resource exhaustion (disk/memory), data loss risk, anything needing prompt human attention
- ignore: routine informational messages, successful/expected operations, benign warnings
Respond with ONLY one word: critical or ignore"""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 512},
    }
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=30)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "").strip().lower()
    except Exception as e:
        print(f"LLM classification failed, defaulting to ignore: {e}", file=sys.stderr)
        return "ignore"
    return "critical" if "critical" in text else "ignore"


def is_critical(entry: dict) -> bool:
    priority = entry.get("priority")
    if priority is not None:
        try:
            if int(priority) <= PRIORITY_LEVELS.get(MIN_PRIORITY, 3):
                return True
        except (ValueError, TypeError):
            pass
    if KEYWORD_PATTERN and KEYWORD_PATTERN.search(entry.get("message", "")):
        return True
    if USE_LLM_CLASSIFICATION:
        return classify_log_llm(entry) == "critical"
    return False


# --- NOTIFICATIONS ---

def format_ts(raw) -> str:
    if not raw:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        # journalctl __REALTIME_TIMESTAMP is microseconds since epoch
        return datetime.fromtimestamp(int(raw) / 1_000_000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    except (ValueError, TypeError, OSError):
        return str(raw)


def send_discord_alert(entries: list) -> None:
    if not WEBHOOK_URL:
        print("Warning: WEBHOOK_LOG (or WEBHOOK_EMAIL) is not configured in .env; "
              "skipping Discord notification.", file=sys.stderr)
        return
    # Discord caps embeds at 25 fields; chunk to be safe well below that.
    for i in range(0, len(entries), 20):
        chunk = entries[i:i + 20]
        fields = [{
            "name": f"{e.get('unit') or e.get('source')} — {format_ts(e.get('timestamp'))}"[:256],
            "value": (e.get("message", "") or "(empty)")[:1000],
            "inline": False,
        } for e in chunk]
        payload = {"embeds": [{
            "title": f"🚨 {len(entries)} critical log event(s) detected",
            "color": 0xff0000,
            "fields": fields,
            "footer": {"text": "Hiryu Log Monitor"},
        }]}
        try:
            requests.post(WEBHOOK_URL, json=payload, timeout=10)
        except Exception as e:
            print(f"Failed sending Discord hook: {e}", file=sys.stderr)


def send_failure_alert(error_msg: str) -> None:
    """System-level alert if the monitor itself crashes."""
    if not WEBHOOK_URL:
        return
    try:
        requests.post(WEBHOOK_URL, json={
            "content": f"⚠️ **Log monitor crashed**\n```{error_msg[:1500]}```"
        }, timeout=10)
    except Exception as e:
        print(f"Failed to deliver crash alert: {e}", file=sys.stderr)


# --- MAIN RUN LOGIC ---

def run_check(since_override=None, dry_run=False, source_filter=None) -> list:
    state = load_state()
    sources = get_sources()
    all_critical = []
    total_scanned = 0

    for src in sources:
        label = src.get("unit") or src.get("path") or "system"
        if source_filter and source_filter not in (label, src.get("unit"), src.get("path")):
            continue
        try:
            if src.get("type") == "journalctl":
                entries, new_cursor = read_journalctl(src, state, since_override)
                if not since_override:
                    state[f"journalctl:{src.get('unit') or 'system'}:{src.get('priority', MIN_PRIORITY)}"] = new_cursor
            elif src.get("type") == "file":
                entries, new_offset = read_file_source(src, state, since_override)
                if not since_override:
                    state[f"file:{src['path']}"] = new_offset
            else:
                print(f"Unknown source type: {src.get('type')!r}", file=sys.stderr)
                continue
        except Exception as e:
            print(f"Error reading source {label}: {e}", file=sys.stderr)
            continue

        total_scanned += len(entries)
        all_critical.extend(e for e in entries if is_critical(e))

    if not since_override:
        save_state(state)

    print(f"Scanned {total_scanned} new log line(s) across {len(sources)} source(s); "
          f"{len(all_critical)} critical.")
    for e in all_critical:
        print(f"  ⚠ [{e.get('unit') or e.get('source')}] {e.get('message', '')[:150]}")

    if all_critical and not dry_run:
        send_discord_alert(all_critical)

    return all_critical


def run_watch(interval: int) -> None:
    """Continuous-loop mode for a persistent systemd service (Type=simple),
    as an alternative to a systemd timer or cron entry."""
    print(f"Starting continuous log monitor (interval={interval}s). Ctrl+C to stop.")
    while True:
        try:
            with lock_or_exit():
                run_check()
        except SystemExit:
            pass  # lock held elsewhere; just wait for the next tick
        except Exception as e:
            print(f"Unhandled error in watch loop: {e}", file=sys.stderr)
            send_failure_alert(str(e))
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Reads system logs and alerts on critical events.")
    sub = parser.add_subparsers(dest="command")

    check_p = sub.add_parser("check", help="Run a single check (default; good for cron or a systemd oneshot service)")
    check_p.add_argument("--since", help="Ad-hoc lookback, e.g. '2 hours ago'. Does NOT update the saved cursor/offset.")
    check_p.add_argument("--dry-run", action="store_true", help="Print findings but don't send a Discord alert")
    check_p.add_argument("--source", help="Only check the source whose unit name or file path matches this")

    watch_p = sub.add_parser("watch", help="Run continuously, checking every --interval seconds (for a persistent systemd service)")
    watch_p.add_argument("--interval", type=int, default=300)

    args = parser.parse_args()
    command = args.command or "check"

    if command == "check":
        with lock_or_exit():
            run_check(
                since_override=getattr(args, "since", None),
                dry_run=getattr(args, "dry_run", False),
                source_filter=getattr(args, "source", None),
            )
    elif command == "watch":
        run_watch(args.interval)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        try:
            send_failure_alert(str(e))
        except Exception:
            pass
        sys.exit(1)
