"""
tools/job_manager.py

Generic infrastructure for background jobs amoai can run that outlast a
single tool call — e.g. "watch this container for the next hour" instead
of a single point-in-time check. A job is a coroutine scheduled onto the
bot's own event loop; this module tracks it (id, description, status,
start time) so it can be listed, checked, and stopped from Discord
without SSH.

Jobs live in memory only — they do NOT survive a bot restart. That's
deliberate: anything that needs to survive a restart belongs in a proper
recurring job (see bot.py's register_job/SCHEDULED_JOBS), not here. This
is for "run once, for a while, then report back."

--- Wiring required in bot.py (see that file for the actual lines) ---
    from tools.job_manager import set_event_loop, set_notifier, set_tool_registry
    set_tool_registry(TOOL_REGISTRY)   # right after register_tools() runs
    set_event_loop(asyncio.get_event_loop())   # in on_ready
    set_notifier(lambda text: send_chunked(bot.get_channel(ALLOWED_CHANNEL_ID), text))   # in on_ready

Without set_event_loop/set_notifier, start_job() will raise and a finished
job's completion message won't be able to reach Discord. Without
set_tool_registry, watch_tool() won't be able to look up any tools to poll.
"""

import asyncio
import itertools
import json
import os
import time
from datetime import datetime, timezone

import requests

_EVENT_LOOP = None
_NOTIFIER = None  # async callable, takes a single str
_TOOL_REGISTRY = None  # name -> callable, set from bot.py after register_tools() runs

_JOBS = {}  # job_id -> {"description", "status", "started_at", "future"}
_id_counter = itertools.count(1)

# watch_tool can only poll tools on this list — a whitelist, not a
# blocklist. Deliberately fail-closed: a new tool added to the repo later
# is NOT watchable until someone explicitly adds it here, rather than
# being watchable by default and only excluded if it happens to match a
# dangerous-looking name pattern. Every entry below was checked by hand —
# it reads state and has no side effects, so calling it repeatedly on a
# timer is safe. Anything that mutates state (restart_container,
# ha_turn_on, cancel_reminder, delete_file, nyaadle_check_now, etc.) is
# deliberately left off — watching one of those would mean re-triggering
# that action every poll, not just observing it.
_WATCHABLE_TOOLS = {
    "check_bot_health", "get_failure_details",
    "list_calendar_events", "list_calendars",
    "list_containers", "get_container_logs",
    "fetch_page",
    "list_directory", "read_file",
    "ha_get_state", "ha_list_entities",
    "check_system_logs",
    "ping", "check_port",
    "nyaadle_watchlist", "nyaadle_status",
    "get_top_processes",
    "search_knowledge", "list_knowledge_sources",
    "list_reminders",
    "list_scratchpad",
    "get_server_status",
    "check_weather",
    "web_search",
}


def set_event_loop(loop):
    global _EVENT_LOOP
    _EVENT_LOOP = loop


def set_tool_registry(registry: dict):
    """Called once from bot.py, right after register_tools() populates
    TOOL_REGISTRY, so watch_tool can look up any other tool by name."""
    global _TOOL_REGISTRY
    _TOOL_REGISTRY = registry


def set_notifier(notifier):
    """notifier: an async callable taking a single str, used to post a
    job's result back to Discord when it finishes on its own (success,
    failure, or being stopped)."""
    global _NOTIFIER
    _NOTIFIER = notifier


async def _notify(text: str):
    if _NOTIFIER is None:
        return
    try:
        await _NOTIFIER(text)
    except Exception:
        pass  # a failed notification shouldn't take the job runner down with it


def _new_job_id() -> str:
    return f"job-{next(_id_counter)}"


async def _run_job(job_id: str, coro):
    try:
        result = await coro
        _JOBS[job_id]["status"] = "completed"
        await _notify(f"✅ **{_JOBS[job_id]['description']}** finished:\n{result}")
    except asyncio.CancelledError:
        _JOBS[job_id]["status"] = "stopped"
        await _notify(f"🛑 **{_JOBS[job_id]['description']}** was stopped.")
    except Exception as e:
        _JOBS[job_id]["status"] = "failed"
        await _notify(f"⚠️ **{_JOBS[job_id]['description']}** failed: {e}")


def start_job(description: str, coro) -> str:
    """Schedules `coro` as a background job on the bot's event loop and
    registers it for tracking. Call this from other tool modules to add
    new job types (see watch_container below for an example) — it's
    infrastructure, not itself an LLM-callable tool, since it takes a
    coroutine rather than something a tool-call argument can express.
    Returns the new job's id."""
    if _EVENT_LOOP is None:
        raise RuntimeError(
            "job_manager.set_event_loop() was never called — "
            "is this running inside the bot process?"
        )
    job_id = _new_job_id()
    future = asyncio.run_coroutine_threadsafe(_run_job(job_id, coro), _EVENT_LOOP)
    _JOBS[job_id] = {
        "description": description,
        "status": "running",
        "started_at": datetime.now(timezone.utc),
        "future": future,
    }
    return job_id


# --- LLM-callable tools ---

def list_jobs() -> str:
    """Lists all background jobs amoai is tracking this session (running,
    completed, stopped, or failed), most recently started first. Use this
    whenever the user asks what's running in the background, or before
    calling job_status/stop_job if you don't already know the job id."""
    if not _JOBS:
        return "No background jobs have been started this session."
    icons = {"running": "🟢", "completed": "✅", "stopped": "🛑", "failed": "⚠️"}
    lines = []
    for job_id, info in sorted(_JOBS.items(), key=lambda kv: kv[1]["started_at"], reverse=True):
        started = info["started_at"].strftime("%Y-%m-%d %H:%M UTC")
        icon = icons.get(info["status"], "❔")
        lines.append(f"{icon} `{job_id}` — {info['description']} ({info['status']}, started {started})")
    return "\n".join(lines)


def job_status(job_id: str) -> str:
    """Checks the status of one background job by id. Use list_jobs first
    if you don't know the id.

    :param job_id: The job id, e.g. "job-3" (from list_jobs).
    """
    info = _JOBS.get(job_id)
    if not info:
        return f"No job found with id '{job_id}'. Use list_jobs to see active ones."
    started = info["started_at"].strftime("%Y-%m-%d %H:%M UTC")
    return f"`{job_id}` — {info['description']}\nStatus: {info['status']}\nStarted: {started}"


def stop_job(job_id: str) -> str:
    """Stops a running background job by id. Only affects jobs amoai
    itself started in this session — has no effect on anything else
    running on the system (containers, processes, etc. keep running;
    only the watching/monitoring job stops).

    :param job_id: The job id, e.g. "job-3" (from list_jobs).
    """
    info = _JOBS.get(job_id)
    if not info:
        return f"No job found with id '{job_id}'. Use list_jobs to see active ones."
    if info["status"] != "running":
        return f"`{job_id}` is already {info['status']} — nothing to stop."
    if _EVENT_LOOP is None:
        return "Can't stop jobs — job_manager isn't wired up to the bot's event loop."
    # Cancelling a run_coroutine_threadsafe future from another thread has
    # to happen via call_soon_threadsafe — calling .cancel() on it directly
    # from this thread isn't safe once the coroutine has already started.
    _EVENT_LOOP.call_soon_threadsafe(info["future"].cancel)
    return f"Stopping `{job_id}` ({info['description']})..."


# --- Generic job type: watch any existing tool ---
# Rather than hand-writing a bespoke polling wrapper per tool (one for
# docker, one for logs, one for nyaadle...), this looks the tool up in
# TOOL_REGISTRY and polls it directly. Covers "watch X for N minutes" for
# anything amoai can already check, without new code per tool.

async def _watch_tool_job(tool_name: str, tool_args: dict, minutes: int, interval_seconds: int) -> str:
    func = _TOOL_REGISTRY[tool_name]
    deadline = time.monotonic() + minutes * 60
    events = []
    last_output = None
    while time.monotonic() < deadline:
        output = await asyncio.to_thread(func, **tool_args)
        if output != last_output:
            events.append(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC:\n{output}")
            last_output = output
        await asyncio.sleep(interval_seconds)
    if not events:
        return f"No changes observed from `{tool_name}` over {minutes} minute(s)."
    # Cap what gets dumped back into Discord — a tool that changes on every
    # poll for an hour could otherwise produce a huge report.
    truncated = len(events) > 10
    shown = events[-10:] if truncated else events
    header = f"Changes from `{tool_name}` over {minutes} minute(s)"
    header += " (last 10 shown):\n" if truncated else ":\n"
    return header + "\n\n".join(shown)


def watch_tool(tool_name: str, minutes: int = 60, interval_seconds: int = 30, tool_args_json: str = "{}") -> str:
    """Starts a background job that repeatedly calls an existing tool on
    an interval and reports back only what changed, instead of checking
    just once. Use this for "watch X for N minutes" requests about
    anything amoai can already check — logs, docker, server status,
    nyaadle, etc. — rather than a one-time call.

    :param tool_name: The exact name of an existing tool to poll, e.g.
        "check_system_logs", "list_containers", "get_server_status".
    :param minutes: How long to watch for, in minutes. Defaults to 60.
    :param interval_seconds: How often to poll, in seconds. Defaults to 30.
    :param tool_args_json: JSON object string of arguments to pass to the
        tool on each poll, e.g. '{"hours": 0}'. Defaults to "{}" (no args).
    """
    if _TOOL_REGISTRY is None:
        return "Can't watch tools yet — job_manager isn't wired up to the bot's tool registry."
    if tool_name not in _WATCHABLE_TOOLS:
        return (
            f"'{tool_name}' isn't watchable — either it changes system state "
            f"(so polling it would repeat that action, not just observe it) "
            f"or it's not on the reviewed safe list yet."
        )
    if tool_name not in _TOOL_REGISTRY:
        return f"No tool named '{tool_name}' is registered — check the name and try again."
    try:
        tool_args = json.loads(tool_args_json)
    except Exception as e:
        return f"Couldn't parse tool_args_json: {e}"

    job_id = start_job(
        f"Watching '{tool_name}' every {interval_seconds}s for {minutes}m",
        _watch_tool_job(tool_name, tool_args, minutes, interval_seconds),
    )
    return (
        f"Started `{job_id}` — polling `{tool_name}` every {interval_seconds}s for {minutes} minute(s). "
        f"I'll post here when it's done, or check progress anytime with job_status."
    )


# --- One-shot delayed job: run a tool once, after a wait ---
# Distinct risk profile from watch_tool/watch_until: since this fires
# exactly once, mutating tools (restart_container, ha_turn_on, etc.) are
# fine here — running one of them once, later, carries the same risk as
# running it directly right now. No whitelist needed for this one.

async def _delayed_job(tool_name: str, tool_args: dict, delay_minutes: float) -> str:
    await asyncio.sleep(delay_minutes * 60)
    func = _TOOL_REGISTRY[tool_name]
    result = await asyncio.to_thread(func, **tool_args)
    return f"Ran `{tool_name}` after the {delay_minutes} minute delay:\n{result}"


def run_after_delay(tool_name: str, delay_minutes: float, tool_args_json: str = "{}") -> str:
    """Schedules an existing tool to run once, after a delay, then reports
    the result back. Use this for "do X in N minutes/hours" requests —
    e.g. "restart the transmission container in 30 minutes" or "check
    nyaadle's queue again in 2 hours." Unlike watch_tool/watch_until, this
    runs the tool exactly once after waiting, so mutating tools are fine
    to use here too — the risk is the same as calling them directly, just
    deferred to later.

    :param tool_name: The exact name of an existing tool to run.
    :param delay_minutes: How long to wait before running it, in minutes.
    :param tool_args_json: JSON object string of arguments to pass to the
        tool, e.g. '{"container_name": "transmission"}'. Defaults to "{}".
    """
    if _TOOL_REGISTRY is None:
        return "Can't schedule tools yet — job_manager isn't wired up to the bot's tool registry."
    if tool_name not in _TOOL_REGISTRY:
        return f"No tool named '{tool_name}' is registered — check the name and try again."
    try:
        tool_args = json.loads(tool_args_json)
    except Exception as e:
        return f"Couldn't parse tool_args_json: {e}"

    job_id = start_job(
        f"Running '{tool_name}' after {delay_minutes}m delay",
        _delayed_job(tool_name, tool_args, delay_minutes),
    )
    return (
        f"Started `{job_id}` — will run `{tool_name}` in {delay_minutes} minute(s) "
        f"and post the result here."
    )


# --- Condition-based watch: poll until an LLM judges a condition met ---
# Same idea as scheduled_briefing's classifier-based judgment call — a
# small/cheap model doing a narrow yes/no read on each poll's output,
# rather than raw diffing (watch_tool) or persona chat (the main model).

OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")
JOB_CLASSIFIER_MODEL = os.getenv("JOB_CLASSIFIER_MODEL", "gpt-oss:20b-cloud")


def _condition_met(tool_output: str, condition: str) -> bool:
    prompt = f"""Tool output:
{tool_output}

Condition to check: {condition}

Has this condition been met, based on the tool output above? Respond with
ONLY "YES" or "NO". No other text."""
    payload = {
        "model": JOB_CLASSIFIER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": 1024},
    }
    try:
        r = requests.post(OLLAMA_API, json=payload, timeout=30)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "").strip().lower()
        return text.startswith("yes")
    except Exception:
        return False  # classifier unreachable — treat as "not met yet", not a false trigger


async def _watch_until_job(tool_name: str, tool_args: dict, condition: str, minutes: int, interval_seconds: int) -> str:
    func = _TOOL_REGISTRY[tool_name]
    deadline = time.monotonic() + minutes * 60
    while time.monotonic() < deadline:
        output = await asyncio.to_thread(func, **tool_args)
        met = await asyncio.to_thread(_condition_met, output, condition)
        if met:
            return f'Condition met: "{condition}"\n\nLatest `{tool_name}` output:\n{output}'
        await asyncio.sleep(interval_seconds)
    return f'Watched `{tool_name}` for {minutes} minute(s) — condition "{condition}" was never met.'


def watch_until(tool_name: str, condition: str, minutes: int = 60, interval_seconds: int = 30, tool_args_json: str = "{}") -> str:
    """Starts a background job that polls an existing tool and checks its
    output against a plain-English condition, reporting back as soon as
    the condition is met instead of waiting out the full duration. Use
    this for "let me know when X happens" requests — e.g. "let me know
    when disk usage drops back under 70%" or "tell me once the jellyfin
    container is healthy again" — rather than watch_tool's raw diff dump.
    Reports "never met" if the duration runs out first.

    :param tool_name: The exact name of an existing read-only tool to
        poll — must be on the same watchable list as watch_tool.
    :param condition: Plain-English description of what to watch for,
        e.g. "disk usage is below 70%".
    :param minutes: Maximum time to watch for, in minutes. Defaults to 60.
    :param interval_seconds: How often to poll, in seconds. Defaults to 30.
    :param tool_args_json: JSON object string of arguments to pass to the
        tool on each poll. Defaults to "{}" (no args).
    """
    if _TOOL_REGISTRY is None:
        return "Can't watch tools yet — job_manager isn't wired up to the bot's tool registry."
    if tool_name not in _WATCHABLE_TOOLS:
        return (
            f"'{tool_name}' isn't watchable — either it changes system state "
            f"or it's not on the reviewed safe list yet."
        )
    if tool_name not in _TOOL_REGISTRY:
        return f"No tool named '{tool_name}' is registered — check the name and try again."
    try:
        tool_args = json.loads(tool_args_json)
    except Exception as e:
        return f"Couldn't parse tool_args_json: {e}"

    job_id = start_job(
        f'Watching \'{tool_name}\' until "{condition}" (up to {minutes}m)',
        _watch_until_job(tool_name, tool_args, condition, minutes, interval_seconds),
    )
    return (
        f"Started `{job_id}` — watching `{tool_name}` every {interval_seconds}s, up to {minutes} minute(s), "
        f'until "{condition}". I\'ll post here as soon as that happens (or if time runs out first).'
    )
