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
) -> str:
    """
    Saves a reminder to mention (ping) a user on Discord. Can be time-based
    (fires at a specific future time) or arrival-based (fires the next time
    Home Assistant reports the user has arrived home).

    :param user_id: The Discord user ID of the person to be reminded.
    :param message: What the user wants to be reminded of (e.g., 'Do laundry', 'Check oven')
    :param minutes_from_now: Optional. The number of minutes from now to trigger the reminder.
    :param target_time_iso: Optional. An absolute future date/time in ISO format (e.g., '2026-07-20T15:00:00Z').
    :param on_arrival: Optional. If true, ignores the time params entirely and instead
        fires the next time the user is detected arriving home — e.g. "remind me to take
        out the trash when I get home".
    """
    if on_arrival:
        new_reminder = {
            "user_id": str(user_id),
            "trigger_type": "arrival",
            "trigger_time": None,
            "message": message,
            "active": True,
        }
        reminders = _load_reminders()
        reminders.append(new_reminder)
        _save_reminders(reminders)
        return f"✅ Got it — I'll remind you to **{message}** the next time you get home."

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
        # FIXED: Removed the buggy 'import timedelta' line
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
        "active": True
    }

    reminders = _load_reminders()
    reminders.append(new_reminder)
    _save_reminders(reminders)

    # Output a friendly formatted confirmation string using local timezone
    local_target = target_dt.astimezone()
    formatted_time = local_target.strftime("%A, %B %d, %Y at %I:%M %p")

    return f"✅ Reminder saved! I will ping you on {formatted_time} to: **{message}**"


def list_reminders(user_id: str) -> str:
    """
    Lists all of this user's currently active (not yet fired or cancelled)
    reminders, both time-based and arrival-based. Each is numbered so the
    number can be passed to cancel_reminder.
    """
    reminders = _load_reminders()
    active = [r for r in reminders if r.get("active") and str(r.get("user_id")) == str(user_id)]

    if not active:
        return "You have no active reminders."

    lines = []
    for i, r in enumerate(active, start=1):
        if r.get("trigger_type") == "arrival":
            lines.append(f"{i}. 🏠 On arrival: **{r.get('message', '')}**")
        else:
            formatted = r.get("trigger_time", "unknown time")
            try:
                clean_iso = r["trigger_time"].replace("Z", "+00:00")
                trigger_dt = datetime.fromisoformat(clean_iso)
                local_dt = trigger_dt.astimezone()
                formatted = local_dt.strftime("%A, %B %d, %Y at %I:%M %p")
            except Exception:
                pass
            lines.append(f"{i}. ⏰ {formatted}: **{r.get('message', '')}**")

    return (
        "Here are your active reminders:\n"
        + "\n".join(lines)
        + "\n\nUse cancel_reminder with the number shown to cancel one."
    )


def cancel_reminder(user_id: str, identifier: str) -> str:
    """
    Cancels one of this user's active reminders. Matched either by its
    1-based position in the list returned by list_reminders, or by a
    snippet of text from the reminder's message.

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


def _fire_arrival_reminders(user_id: str) -> Optional[str]:
    """Finds active 'on arrival' reminders for this user, deactivates them,
    and returns a formatted ping string — or None if there were none due.
    Called by the arrival webhook handler in bot.py, not by the LLM."""
    reminders = _load_reminders()
    due = []
    updated = []
    for r in reminders:
        if (
            r.get("active")
            and r.get("trigger_type") == "arrival"
            and str(r.get("user_id")) == str(user_id)
        ):
            due.append(r)
            r["active"] = False
        updated.append(r)

    if not due:
        return None

    _save_reminders(updated)
    lines = [f"🔔 <@{user_id}>! Here is your reminder: **{r['message']}**" for r in due]
    return "\n".join(lines)
