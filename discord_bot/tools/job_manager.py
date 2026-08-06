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

--- Wiring required in bot.py's on_ready (see that file for the actual
lines) ---
    from tools.job_manager import set_event_loop, set_notifier
    set_event_loop(asyncio.get_event_loop())
    set_notifier(lambda text: send_chunked(bot.get_channel(ALLOWED_CHANNEL_ID), text))

Without that wiring, start_job() will raise, and stop_job()/a finished
job's completion message won't be able to reach the event loop or post
back to Discord.
"""

import asyncio
import itertools
import subprocess
import time
from datetime import datetime, timezone

_EVENT_LOOP = None
_NOTIFIER = None  # async callable, takes a single str

_JOBS = {}  # job_id -> {"description", "status", "started_at", "future"}
_id_counter = itertools.count(1)


def set_event_loop(loop):
    global _EVENT_LOOP
    _EVENT_LOOP = loop


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


# --- Example job type: container watching ---
# This is the concrete case that motivated job_manager in the first place
# ("watch this container for the next hour"). Any other tool module can
# add its own job type the same way: an async worker function, plus a
# thin LLM-callable wrapper that calls start_job().

async def _watch_container_job(container_name: str, minutes: int) -> str:
    deadline = time.monotonic() + minutes * 60
    events = []
    last_status = None
    while time.monotonic() < deadline:
        result = await asyncio.to_thread(
            subprocess.run,
            ["docker", "inspect", "--format",
             "{{.State.Status}} (health: {{.State.Health.Status}})", container_name],
            capture_output=True, text=True,
        )
        status = result.stdout.strip() or result.stderr.strip()
        if status != last_status:
            events.append(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC — {status}")
            last_status = status
        await asyncio.sleep(30)
    if not events:
        return f"No status changes observed for `{container_name}` over {minutes} minute(s)."
    return f"Status changes for `{container_name}` over {minutes} minute(s):\n" + "\n".join(events)


def watch_container(container_name: str, minutes: int = 60) -> str:
    """Starts a background job that watches a Docker container's status
    for a set duration and reports back when done, or sooner if its
    status changes. Use this instead of list_containers/get_container_logs
    when the user wants ongoing monitoring rather than a one-time check —
    e.g. "watch the jellyfin container for the next hour and let me know
    if it goes down."

    :param container_name: The name of the Docker container to watch.
    :param minutes: How long to watch for, in minutes. Defaults to 60.
    """
    job_id = start_job(
        f"Watching container '{container_name}' for {minutes}m",
        _watch_container_job(container_name, minutes),
    )
    return (
        f"Started `{job_id}` — watching `{container_name}` for {minutes} minute(s). "
        f"I'll post here when it's done, or you can check progress anytime with job_status."
    )
