import json
from datetime import datetime, timezone
from pathlib import Path

# Path to the persistent reminders storage file (located in the bot's root directory)
REMINDERS_FILE = Path(__file__).resolve().parent.parent / "reminders.json"

def load_reminders() -> list:
    """Loads existing reminders from the local JSON file."""
    if not REMINDERS_FILE.exists():
        return []
    try:
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

def save_reminders(reminders: list):
    """Saves active reminders back to the JSON file."""
    try:
        with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(reminders, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[REMINDER TOOL] Failed to save reminders: {e}")

def set_reminder(user_id: str, message: str, minutes_from_now: float = None, target_time_iso: str = None) -> str:
    """
    Saves a reminder to mention (ping) a user on Discord at a specific future time.
    
    :param user_id: The Discord user ID of the person to be reminded.
    :param message: What the user wants to be reminded of (e.g., 'Do laundry', 'Check oven')
    :param minutes_from_now: Optional. The number of minutes from now to trigger the reminder.
    :param target_time_iso: Optional. An absolute future date/time in ISO format (e.g., '2026-07-20T15:00:00Z'). 
    """
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
    elif minutes_from_now is not None:
        if minutes_from_now <= 0:
            return "❌ Error: Time duration must be a positive number of minutes."
        # FIXED: Removed the buggy 'import timedelta' line
        from datetime import timedelta
        target_dt = now + timedelta(minutes=minutes_from_now)
    else:
        return "❌ Error: You must provide either minutes_from_now or target_time_iso."

    if target_dt < now:
        return "❌ Error: The calculated reminder time is in the past!"

    new_reminder = {
        "user_id": str(user_id),
        "trigger_time": target_dt.isoformat(),
        "message": message,
        "active": True
    }

    reminders = load_reminders()
    reminders.append(new_reminder)
    save_reminders(reminders)

    # Output a friendly formatted confirmation string using local timezone
    local_target = target_dt.astimezone()
    formatted_time = local_target.strftime("%A, %B %d, %Y at %I:%M %p")

    return f"✅ Reminder saved! I will ping you on {formatted_time} to: **{message}**"
