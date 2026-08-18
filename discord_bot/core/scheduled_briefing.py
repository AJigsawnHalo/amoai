"""
discord_bot/scheduled_briefing.py

Builds the daily morning briefing text (calendar, reminders, weather, hiryu
status, log check, docker, nyaadle, bot health).

This module is intentionally NOT inside tools/, so it is never
auto-discovered as an LLM-callable tool — it exists only to be imported by
bot.py's own scheduler (see register_job("Morning Briefing", ...) in
bot.py). The bot posts the result through its own connection via
send_chunked(), so it shows up as Amoai talking, same as any other message
it sends — no webhook, no separate identity to fake.

build_briefing() does blocking I/O (subprocess calls via the tools it
wraps), so bot.py calls it with `await asyncio.to_thread(...)` rather than
awaiting it directly.
"""

import os
import json
from datetime import datetime

from tools.calendar_tool import list_calendar_events
from tools.reminder_tool import list_reminders
from tools.weather import check_weather
from tools.server_status import get_server_status
from tools.log_monitor import check_system_logs
from tools.docker_manager import list_containers
from tools.nyaadle_control import nyaadle_status
from tools.bot_health import check_bot_health

DISCORD_USER_ID = os.getenv("DISCORD_USER_ID")
WEATHER_LOCATION = os.getenv("WEATHER_LOCATION", "")


def _section(title: str, body: str) -> str:
    return f"**{title}**\n{body.strip()}"


def _safe(label: str, icon: str, fn, *args, **kwargs) -> str:
    """Runs a section function and turns any exception into a visible
    warning instead of taking down the whole briefing."""
    try:
        return _section(f"{icon} {label}", fn(*args, **kwargs))
    except Exception as e:
        return _section(f"{icon} {label}", f"⚠️ Couldn't load: {e}")


def get_calendar_section() -> str:
    return _safe("Today", "📅", list_calendar_events, days_ahead=1)


def get_reminders_section() -> str:
    if not DISCORD_USER_ID:
        return _section("⏰ Reminders", "⚠️ DISCORD_USER_ID not set in .env — skipped.")
    return _safe("Reminders", "⏰", list_reminders, DISCORD_USER_ID)


def get_weather_section() -> str:
    if not WEATHER_LOCATION:
        return _section("🌤️ Weather", "⚠️ WEATHER_LOCATION not set in .env — skipped.")
    try:
        data = json.loads(check_weather(WEATHER_LOCATION))
        if data.get("status") != "success":
            return _section("🌤️ Weather", f"⚠️ {data.get('message', 'lookup failed')}")
        line = (
            f"{data['conditions']}, {data['temperature']} "
            f"(feels like {data['feels_like_temperature']}) — "
            f"{data['chance_of_rain']} chance of rain"
        )
        return _section("🌤️ Weather", line)
    except Exception as e:
        return _section("🌤️ Weather", f"⚠️ Couldn't fetch weather: {e}")


def get_server_status_section() -> str:
    return _safe("Hiryu", "🖥️", get_server_status)


def get_log_check_section() -> str:
    return _safe("System logs", "📋", check_system_logs, hours=0)


def get_docker_section() -> str:
    return _safe("Docker", "🐳", list_containers)


def get_nyaadle_section() -> str:
    return _safe("Nyaadle", "📥", nyaadle_status, count=5)


def get_bot_health_section() -> str:
    return _safe("Bot health", "🩺", check_bot_health, limit=3)


def build_briefing() -> str:
    """Assembles the full briefing text. Blocking — call via
    asyncio.to_thread from async code."""
    today = datetime.now().strftime("%A, %B %d, %Y")
    header = f"☀️ **Good morning! Here's everything lined up for {today} — let's make it a good one.**\n"
    sections = [
        get_calendar_section(),
        get_reminders_section(),
        get_weather_section(),
        get_server_status_section(),
        get_log_check_section(),
        get_docker_section(),
        get_nyaadle_section(),
        get_bot_health_section(),
    ]
    return header + "\n\n".join(sections)

