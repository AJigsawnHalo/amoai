"""
scheduler.py — the proactive job scheduler (register_job/scheduler_tick),
reminder resolution shared with webhook.py's arrival handler, and the
morning briefing gate.

MORNING_BRIEFING_ENABLED is mutable (toggled live via !briefing on/off in
bot.py's on_message) — callers must `import scheduler` and reference
`scheduler.MORNING_BRIEFING_ENABLED`, never a `from` import.
"""
import asyncio
import inspect
import os
import time
from datetime import datetime

from discord.ext import tasks

import config
from discord_client import bot
import messaging
import tool_registry
from tools.reminder_tool import _get_due_time_reminders, BOT_TIMEZONE
from scheduled_briefing import build_briefing

SCHEDULED_JOBS = []


def register_job(name: str, interval_seconds: int, func):
    SCHEDULED_JOBS.append({
        "name": name,
        "interval_seconds": interval_seconds,
        "func": func,
        "last_run": 0.0,
    })


async def _resolve_due_reminders(due: list) -> str:
    """Given a list of due reminder dicts (from reminder_tool's
    _get_due_time_reminders / _get_due_arrival_reminders — already
    deactivated and saved), runs any attached action_tool via the live
    tool_registry.TOOL_REGISTRY and builds the text to post in the channel.
    Plain reminders (no action_tool) just become a ping."""
    lines = []
    for r in due:
        uid = r.get("user_id") or config.DISCORD_USER_ID
        ping = f"<@{uid}>" if uid else "Someone"
        message = r.get("message", "")
        action_tool = r.get("action_tool")

        if not action_tool:
            lines.append(f"🔔 {ping}! Here is your reminder: **{message}**")
            continue

        if action_tool not in tool_registry.TOOL_REGISTRY:
            text = f"🔔 {ping} ⚠️ **{message}** was due, but the tool `{action_tool}` no longer exists."
            tool_registry.log_tool_call(action_tool, r.get("action_args", {}), "unknown tool", source="scheduler")
            lines.append(text)
            continue

        args = dict(r.get("action_args") or {})
        func = tool_registry.TOOL_REGISTRY[action_tool]
        if "user_id" in inspect.signature(func).parameters:
            args["user_id"] = str(r.get("user_id") or "")

        if tool_registry.needs_confirmation(action_tool, args):
            text = (
                f"🔔 {ping} ⏰ **{message}** is due and would run `{action_tool}`, "
                "but that tool needs confirmation and can't run unattended — please run it yourself."
            )
            tool_registry.log_tool_call(action_tool, args, "skipped: needs confirmation", source="scheduler")
            lines.append(text)
            continue

        try:
            output = await asyncio.to_thread(func, **args)
        except Exception as e:
            output = f"Error running tool: {e}"
        tool_registry.log_tool_call(action_tool, args, output, source="scheduler")
        lines.append(f"🔔 {ping} ⏰ **{message}** — {output}")

    return "\n".join(lines)


async def check_scheduled_reminders() -> "str | None":
    due = await asyncio.to_thread(_get_due_time_reminders)
    if not due:
        return None
    return await _resolve_due_reminders(due)


register_job("Reminder Alert", 60, check_scheduled_reminders)

# --- MORNING BRIEFING ---
# Fires once per day at/after MORNING_BRIEFING_HOUR (default 6am, in
# BOT_TIMEZONE). Gated by _last_briefing_date rather than a fixed
# interval_seconds, since register_job's interval is measured from
# whenever the bot last happened to start — that drifts over restarts and
# wouldn't reliably land at the same wall-clock hour every day.
MORNING_BRIEFING_HOUR = int(os.getenv("MORNING_BRIEFING_HOUR", 6))
_last_briefing_date = None

# Runtime on/off switch. Seeded from .env so it survives as the default
# across restarts, but toggleable live via !briefing on/off without editing
# .env or restarting the bot (see the !briefing command handler in bot.py).
MORNING_BRIEFING_ENABLED = os.getenv("MORNING_BRIEFING_ENABLED", "true").strip().lower() not in (
    "0", "false", "no", "off"
)


async def check_morning_briefing() -> "str | None":
    global _last_briefing_date
    if not MORNING_BRIEFING_ENABLED:
        return None
    now = datetime.now(BOT_TIMEZONE)
    if now.hour < MORNING_BRIEFING_HOUR:
        return None
    if _last_briefing_date == now.date():
        return None
    _last_briefing_date = now.date()
    return await asyncio.to_thread(build_briefing)


register_job("Morning Briefing", 60, check_morning_briefing)


@tasks.loop(seconds=60)
async def scheduler_tick():
    if not config.ALLOWED_CHANNEL_ID:
        return
    channel = bot.get_channel(config.ALLOWED_CHANNEL_ID)
    if channel is None:
        return

    now = time.time()
    for job in SCHEDULED_JOBS:
        if now - job["last_run"] < job["interval_seconds"]:
            continue
        job["last_run"] = now
        try:
            if asyncio.iscoroutinefunction(job["func"]):
                result = await job["func"]()
            else:
                result = await asyncio.to_thread(job["func"])
        except Exception as e:
            print(f"[SCHEDULER] Job '{job['name']}' failed: {e}")
            continue
        if result:
            await messaging.send_chunked(channel, result)
            tool_registry.log_tool_call(job["name"], {}, result, source="scheduler")
