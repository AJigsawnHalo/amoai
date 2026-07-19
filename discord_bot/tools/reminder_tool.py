import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Path to the persistent reminders storage file (located in the bot's root directory)
REMINDERS_FILE = Path(__file__).resolve().parent.parent / "reminders.json"

def _load_reminders() -> list:
    """Loads existing reminders from the local JSON file."""
    if not REMINDERS_FILE.exists():
        return []
    try:
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    # Guard against a corrupted file that parses but isn't actually a list
    # (e.g. a stray quoted string or a single object instead of an array).
    if not isinstance(data, list):
        print(f"[REMINDER TOOL] reminders.json contained {type(data).__name__}, not a list — resetting.")
        return []
    return data

def _save_reminders(reminders: list):
    """Saves active reminders back to the JSON file."""
    try:
        with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(reminders, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[REMINDER TOOL] Failed to save reminders: {e}")


def set_reminder(
    user_id: str,
    message: str,
    minutes_from_now: float = None,
    target_time_iso: str = None,
    on_arrival: bool = False,
    zone: str = "home",
    action_tool: str = None,
    action_args_json: str = None,
) -> str:
    """
    Saves a reminder for the user. Can just ping them with a message, or —
    if action_tool is given — actually RUN another one of your tools when it
    comes due and report the result. Use this for requests like "turn on the
    AC after 30 mins", "restart the server at 11pm", or "when I get home,
    turn off the porch light": pick the right tool from your own tool list
    for the action, and schedule it here instead of calling it immediately.

    Firing can be time-based (a delay or an absolute date/time) or
    arrival-based (fires next time Home Assistant reports the user entered a
    given zone).

    :param user_id: The Discord user ID of the person to be reminded / on whose behalf the action runs.
    :param message: A short human-readable label for what this reminder is about (e.g. 'Do laundry', 'Turn on the AC'). Always required, even for action reminders — it's shown to the user.
    :param minutes_from_now: Optional. The number of minutes from now to trigger.
    :param target_time_iso: Optional. An absolute future date/time in ISO format (e.g., '2026-07-20T15:00:00Z').
    :param on_arrival: Optional. If true, ignores the time params entirely and instead fires the next time the user is detected entering the given zone.
    :param zone: Optional, only used with on_arrival. The Home Assistant zone name to watch for (e.g. 'home', 'gym', 'office'). Defaults to 'home'. Must match the zone name reported by the arrival webhook.
    :param action_tool: Optional. The exact name of another tool you have access to (e.g. a Home Assistant control tool). If given, that tool is actually called when this reminder fires, instead of just sending a plain ping. Omit for a plain reminder.
    :param action_args_json: Optional, only used with action_tool. A JSON object (as a string) of the arguments to pass to that tool, using its exact parameter names — e.g. '{"entity": "ac", "state": "on"}'. Do NOT include user_id here; it's added automatically if the tool needs it. Use '{}' if the tool takes no arguments.
    """
    action_args = None
    if action_tool:
        raw_args = action_args_json if action_args_json is not None else "{}"
        try:
            action_args = json.loads(raw_args)
        except json.JSONDecodeError:
            return "❌ Error: action_args_json must be a valid JSON object string, e.g. '{\"entity\": \"ac\"}'."
        if not isinstance(action_args, dict):
            return "❌ Error: action_args_json must decode to a JSON object (key/value pairs), not a list or scalar."

    if on_arrival:
        zone = (zone or "home").strip().lower()
        new_reminder = {
            "user_id": str(user_id),
            "trigger_type": "arrival",
            "trigger_time": None,
            "trigger_zone": zone,
            "message": message,
            "action_tool": action_tool,
            "action_args": action_args or {},
            "active": True,
        }
        reminders = _load_reminders()
        reminders.append(new_reminder)
        _save_reminders(reminders)
        where = "you get home" if zone == "home" else f"you arrive at '{zone}'"
        if action_tool:
            return f"✅ Got it — I'll run **{action_tool}** ({message}) the next time {where}."
        return f"✅ Got it — I'll remind you to **{message}** the next time {where}."

    now = datetime.now(timezone.utc)
    target_dt = None

    # Determine the target datetime
    if target_time_iso:
        try:
            # Parse ISO format (handling optional 'Z' suffix as UTC timezone)
            clean_iso = target_time_iso.replace("Z", "+00:00")
            target_dt = datetime.fromisoformat(clean_iso)
        except ValueError:
            return "❌ Error: Invalid format for target_time_iso. Must be valid ISO-8601."
        if target_dt.tzinfo is None:
            # No offset was given (naive datetime) — assume it means local time
            # rather than letting it crash when compared against aware 'now'.
            target_dt = target_dt.astimezone()
    elif minutes_from_now is not None:
        if minutes_from_now <= 0:
            return "❌ Error: Time duration must be a positive number of minutes."
        from datetime import timedelta
        target_dt = now + timedelta(minutes=minutes_from_now)
    else:
        return "❌ Error: You must provide either minutes_from_now, target_time_iso, or on_arrival."

    if target_dt < now:
        return "❌ Error: The calculated reminder time is in the past!"

    new_reminder = {
        "user_id": str(user_id),
        "trigger_type": "time",
        "trigger_time": target_dt.isoformat(),
        "message": message,
        "action_tool": action_tool,
        "action_args": action_args or {},
        "active": True
    }

    reminders = _load_reminders()
    reminders.append(new_reminder)
    _save_reminders(reminders)

    # Output a friendly formatted confirmation string using local timezone
    local_target = target_dt.astimezone()
    formatted_time = local_target.strftime("%A, %B %d, %Y at %I:%M %p")

    if action_tool:
        return f"✅ Scheduled! I will run **{action_tool}** ({message}) on {formatted_time}."
    return f"✅ Reminder saved! I will ping you on {formatted_time} to: **{message}**"


def list_reminders(user_id: str) -> str:
    """
    Lists all of this user's currently active (not yet fired or cancelled)
    reminders and scheduled actions, both time-based and arrival-based.
    Each is numbered so the number can be passed to cancel_reminder.
    """
    reminders = _load_reminders()
    active = [r for r in reminders if r.get("active") and str(r.get("user_id")) == str(user_id)]

    if not active:
        return "You have no active reminders."

    lines = []
    for i, r in enumerate(active, start=1):
        action_tool = r.get("action_tool")
        action_note = f" _(runs `{action_tool}`)_" if action_tool else ""
        icon = "⚙️" if action_tool else "⏰"

        if r.get("trigger_type") == "arrival":
            zone = r.get("trigger_zone", "home")
            zone_label = "On arrival" if zone == "home" else f"On arrival at '{zone}'"
            lines.append(f"{i}. 🏠 {zone_label}: **{r.get('message', '')}**{action_note}")
        else:
            formatted = r.get("trigger_time", "unknown time")
            try:
                clean_iso = r["trigger_time"].replace("Z", "+00:00")
                trigger_dt = datetime.fromisoformat(clean_iso)
                local_dt = trigger_dt.astimezone()
                formatted = local_dt.strftime("%A, %B %d, %Y at %I:%M %p")
            except Exception:
                pass
            lines.append(f"{i}. {icon} {formatted}: **{r.get('message', '')}**{action_note}")

    return (
        "Here are your active reminders:\n"
        + "\n".join(lines)
        + "\n\nUse cancel_reminder with the number shown to cancel one."
    )


def cancel_reminder(user_id: str, identifier: str) -> str:
    """
    Cancels one of this user's active reminders or scheduled actions.
    Matched either by its 1-based position in the list returned by
    list_reminders, or by a snippet of text from the reminder's message.

    :param identifier: The number shown by list_reminders (e.g. '2'), or a piece of the reminder's message text to match against.
    """
    reminders = _load_reminders()
    active_indices = [
        idx for idx, r in enumerate(reminders)
        if r.get("active") and str(r.get("user_id")) == str(user_id)
    ]

    if not active_indices:
        return "You have no active reminders to cancel."

    identifier = identifier.strip()
    target_idx = None

    if identifier.isdigit():
        pos = int(identifier) - 1
        if 0 <= pos < len(active_indices):
            target_idx = active_indices[pos]
    else:
        for idx in active_indices:
            if identifier.lower() in reminders[idx].get("message", "").lower():
                target_idx = idx
                break

    if target_idx is None:
        return (
            f"❌ Couldn't find an active reminder matching '{identifier}'. "
            "Try list_reminders to see the numbered list."
        )

    cancelled = reminders[target_idx]
    reminders[target_idx]["active"] = False
    _save_reminders(reminders)
    return f"🗑️ Cancelled reminder: **{cancelled.get('message', '')}**"


def _get_due_arrival_reminders(user_id: str, zone: str = "home") -> list:
    """Finds active 'on arrival' reminders for this user matching the given
    zone, deactivates them, and returns the list of due reminder dicts (each
    carrying message and, if applicable, action_tool/action_args) — or [] if
    none were due. Reminders saved before zones existed have no
    'trigger_zone' key and are treated as 'home'.
    Called by the arrival webhook handler in bot.py, not by the LLM.
    bot.py is responsible for actually running any action_tool and for
    formatting/sending the resulting message, since it's the only place
    that has the tool registry."""
    zone = (zone or "home").strip().lower()
    reminders = _load_reminders()
    due = []
    updated = []
    for r in reminders:
        if (
            r.get("active")
            and r.get("trigger_type") == "arrival"
            and str(r.get("user_id")) == str(user_id)
            and r.get("trigger_zone", "home") == zone
        ):
            due.append(r)
            r["active"] = False
        updated.append(r)

    if not due:
        return []

    _save_reminders(updated)
    return due


def _get_due_time_reminders() -> list:
    """Finds all active time-based reminders whose trigger time has passed,
    deactivates them, and returns the list of due reminder dicts (each
    carrying message and, if applicable, action_tool/action_args).
    Called by the scheduler tick in bot.py, not by the LLM. bot.py is
    responsible for actually running any action_tool and for
    formatting/sending the resulting message."""
    reminders = _load_reminders()
    now = datetime.now(timezone.utc)
    due = []
    updated = []

    for r in reminders:
        if r.get("trigger_type") == "time" and r.get("active"):
            try:
                clean_iso = r["trigger_time"].replace("Z", "+00:00")
                trigger_dt = datetime.fromisoformat(clean_iso)
                if trigger_dt <= now:
                    due.append(r)
                    r["active"] = False
            except Exception as e:
                print(f"[REMINDER TOOL] Error parsing reminder time: {e}")
        updated.append(r)

    if due:
        _save_reminders(updated)
    return due
