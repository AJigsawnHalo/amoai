"""
webhook.py — small aiohttp server that listens for Home Assistant "arrived
home" events and posts a greeting / resolves any due arrival reminders.
"""
import asyncio
import os

from aiohttp import web

import config
from discord_client import bot
import messaging
import scheduler
import tool_registry
from tools.reminder_tool import _get_due_arrival_reminders

ARRIVAL_WEBHOOK_PORT = int(os.getenv("ARRIVAL_WEBHOOK_PORT", 8787))
ARRIVAL_WEBHOOK_SECRET = os.getenv("ARRIVAL_WEBHOOK_SECRET")
_webhook_runner = None


async def on_arrived_home(user_id: str, zone: str = "home"):
    if not config.ALLOWED_CHANNEL_ID:
        return
    channel = bot.get_channel(config.ALLOWED_CHANNEL_ID)
    if channel is None:
        return

    due = await asyncio.to_thread(_get_due_arrival_reminders, user_id, zone)
    if due:
        text = await scheduler._resolve_due_reminders(due)
    elif zone == "home":
        # 'home' keeps its old unconditional greeting even with no reminder set.
        text = f"🏠 Welcome home, <@{user_id}>!"
    else:
        # Other zones stay silent unless a reminder was actually set for them.
        tool_registry.log_tool_call("on_arrived_home", {"user_id": user_id, "zone": zone},
                                     "no reminder set for this zone, skipped", source="webhook")
        return

    await messaging.send_chunked(channel, text)
    tool_registry.log_tool_call("on_arrived_home", {"user_id": user_id, "zone": zone}, text, source="webhook")


async def handle_arrived_home(request: web.Request) -> web.Response:
    if not ARRIVAL_WEBHOOK_SECRET or request.headers.get("X-Webhook-Secret") != ARRIVAL_WEBHOOK_SECRET:
        return web.Response(status=401, text="unauthorized")

    try:
        body = await request.json()
    except Exception:
        body = {}
    user_id = str(body.get("user_id") or config.DISCORD_USER_ID or "")
    if not user_id:
        return web.Response(status=400, text="no user_id in request or DISCORD_USER_ID in .env")
    zone = str(body.get("zone") or "home").strip().lower()

    await on_arrived_home(user_id, zone)
    return web.Response(status=200, text="ok")


async def start_webhook_server():
    global _webhook_runner
    if _webhook_runner is not None:
        return
    if not ARRIVAL_WEBHOOK_SECRET:
        print("[WEBHOOK] ARRIVAL_WEBHOOK_SECRET not set in .env — arrival webhook disabled.")
        return
    app = web.Application()
    app.router.add_post("/webhook/arrived-home", handle_arrived_home)
    _webhook_runner = web.AppRunner(app)
    await _webhook_runner.setup()
    site = web.TCPSite(_webhook_runner, "0.0.0.0", ARRIVAL_WEBHOOK_PORT)
    await site.start()
    print(f"[WEBHOOK] Listening for arrival events on :{ARRIVAL_WEBHOOK_PORT}")
