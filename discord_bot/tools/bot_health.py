"""Tools for inspecting the bot's own health via its failed_payload_*.json
error dumps (written by _dump_failed_payload in bot.py whenever a call to the
LLM backend fails).

Drop this file into your `tools/` package — it will be auto-discovered by
register_tools() on next restart, no other wiring needed.

NOTE: requires the small patch to bot.py's _dump_failed_payload (see the
accompanying bot_health_patch.py) so dumps include status/error info, not
just the raw request payload. Without that patch, check_bot_health() still
works but can only report "legacy dump, no error details saved" for old files.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

# tools/bot_health.py -> parent is tools/, parent.parent is the bot's root dir
# where bot.py lives and where failed_payload_*.json files get written.
BOT_ROOT = Path(__file__).resolve().parent.parent


def _dump_files() -> list[Path]:
    files = list(BOT_ROOT.glob("failed_payload_*.json"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def check_bot_health(limit: int = 5) -> str:
    """Checks the bot's health by summarizing its recent LLM call failures (failed_payload_*.json dump files). Reports how many failures were logged and details on the most recent ones."""
    files = _dump_files()
    if not files:
        return "✅ No failure dumps found. The bot hasn't logged any errors."

    out = [f"⚠️ Found {len(files)} failure dump(s). Showing the {min(limit, len(files))} most recent:\n"]
    for f in files[:limit]:
        mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            out.append(f"- {f.name} ({mtime}): could not read file ({e})")
            continue

        if isinstance(data, dict) and "payload" in data:
            # patched-format dump
            status = data.get("status", "unknown")
            model = data.get("model", "unknown")
            error_body = (data.get("error_body") or "").strip().replace("\n", " ")[:150]
            out.append(f"- {f.name} ({mtime}): model={model}, status={status}, error={error_body!r}")
        else:
            # legacy-format dump: just the raw payload, no error details
            model = data.get("model", "unknown") if isinstance(data, dict) else "unknown"
            out.append(f"- {f.name} ({mtime}): model={model} (legacy dump, no error details saved)")

    return "\n".join(out)


def get_failure_details(filename: str) -> str:
    """Returns the full content of one specific failure dump file (by name) for deeper debugging."""
    path = BOT_ROOT / filename
    if path.parent != BOT_ROOT or not path.name.startswith("failed_payload_") or path.suffix != ".json":
        return f"Error: '{filename}' is not a valid failure dump filename."
    if not path.exists():
        return f"Error: {filename} not found."
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as e:
        return f"Error reading {filename}: {e}"
    if len(content) > 4000:
        return content[:4000] + "\n...(truncated — file is larger, ask for a specific section if needed)"
    return content


def clear_failure_logs() -> str:
    """Deletes all failed_payload_*.json dump files, resetting the bot's failure history."""
    files = _dump_files()
    if not files:
        return "No failure dumps to clear."
    cleared = 0
    for f in files:
        try:
            f.unlink()
            cleared += 1
        except OSError:
            pass
    return f"Cleared {cleared} failure dump file(s)."
