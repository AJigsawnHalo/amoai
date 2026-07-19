"""
calendar_tool.py

Google Calendar integration for the Discord bot. Lets the LLM list upcoming
events, create new ones, delete existing ones, and see which calendars are
available on the connected Google account.

--- ONE-TIME SETUP (do this before the tool will work) ---
1. In Google Cloud Console, create a project (or reuse one), enable the
   "Google Calendar API", and create an OAuth Client ID of type
   "Desktop app". Download the resulting JSON.
2. Save that file as data/google_calendar_credentials.json in the bot's
   root directory (or point GOOGLE_CALENDAR_CREDENTIALS_FILE in .env at
   wherever you put it).
3. Run `python setup_google_calendar_auth.py` ONCE, from a machine with a
   browser (this can be your laptop — it doesn't have to be the server).
   It'll open a Google consent screen and, once approved, write
   data/google_calendar_token.json. If you ran it somewhere other than the
   server, copy that token file over to the server afterward.
4. From then on the bot refreshes the token automatically — no browser
   needed again unless you revoke access.

Public tool functions (auto-discovered by bot.py's register_tools()):
    - list_calendar_events(days_ahead, calendar_id)
    - create_calendar_event(summary, start_time_iso, end_time_iso, description, location, all_day, calendar_id)
    - delete_calendar_event(identifier, days_ahead, calendar_id)
    - list_calendars()

Everything prefixed with "_" is a private helper and will NOT be
registered as an LLM-callable tool.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv(find_dotenv())

# --- CONFIGURATION ---
SCOPES = ["https://www.googleapis.com/auth/calendar"]

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CALENDAR_CREDENTIALS_FILE", _DATA_DIR / "google_calendar_credentials.json"))
TOKEN_FILE = Path(os.getenv("GOOGLE_CALENDAR_TOKEN_FILE", _DATA_DIR / "google_calendar_token.json"))
DEFAULT_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

_SETUP_HINT = (
    "Run `python setup_google_calendar_auth.py` once (see the top of "
    "calendar_tool.py for the full setup steps) to connect a Google account."
)


# --- AUTH ---

def _get_service():
    """Builds an authenticated Calendar API client from the saved token,
    refreshing it if expired. Raises RuntimeError with a user-facing
    message if setup hasn't been done yet."""
    if not TOKEN_FILE.exists():
        raise RuntimeError(f"Google Calendar isn't connected yet. {_SETUP_HINT}")

    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds.valid:
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            TOKEN_FILE.write_text(creds.to_json())
        else:
            raise RuntimeError(
                f"The saved Google Calendar token is invalid and has no refresh "
                f"token to recover with. {_SETUP_HINT}"
            )

    return build("calendar", "v3", credentials=creds, cache_discovery=False)


# --- TIME HELPERS ---

def _parse_iso(dt_str: str) -> datetime:
    """Parses ISO-8601, treating a naive (no offset) datetime as local time,
    same convention as reminder_tool.py."""
    clean = dt_str.replace("Z", "+00:00")
    dt = datetime.fromisoformat(clean)
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt


def _format_dt(dt_str: str, all_day: bool) -> str:
    if all_day:
        try:
            return datetime.fromisoformat(dt_str).strftime("%A, %B %d, %Y")
        except ValueError:
            return dt_str
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.astimezone().strftime("%A, %B %d, %Y at %I:%M %p")
    except ValueError:
        return dt_str


# ============================================================
# PUBLIC TOOLS (discovered + exposed to the LLM by bot.py)
# ============================================================

def list_calendar_events(days_ahead: int = 7, calendar_id: str = None) -> str:
    """
    Lists upcoming events on the connected Google Calendar over the next N
    days, soonest first. Use this whenever the user asks what's on their
    calendar/schedule, whether they're free, or what's coming up.

    :param days_ahead: How many days from now to look ahead. Defaults to 7.
    :param calendar_id: Optional. Which calendar to check — defaults to the primary calendar. Use list_calendars if the user names a specific calendar you're unsure of the ID for.
    """
    calendar_id = calendar_id or DEFAULT_CALENDAR_ID
    try:
        service = _get_service()
    except RuntimeError as e:
        return f"❌ {e}"

    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=max(1, days_ahead))).isoformat()

    try:
        result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=50,
        ).execute()
    except HttpError as e:
        return f"❌ Google Calendar error: {e}"

    events = result.get("items", [])
    if not events:
        return f"No events in the next {days_ahead} day(s)."

    lines = [f"📅 Upcoming events (next {days_ahead} day(s)):\n"]
    for ev in events:
        start = ev.get("start", {})
        all_day = "date" in start
        raw = start.get("date") or start.get("dateTime", "")
        when = _format_dt(raw, all_day)
        title = ev.get("summary", "(no title)")
        loc = f" — {ev['location']}" if ev.get("location") else ""
        lines.append(f"- **{title}** — {when}{loc}")

    return "\n".join(lines)


def create_calendar_event(
    summary: str,
    start_time_iso: str,
    end_time_iso: str = None,
    description: str = None,
    location: str = None,
    all_day: bool = False,
    calendar_id: str = None,
) -> str:
    """
    Creates a new event on the connected Google Calendar. Use this whenever
    the user asks to schedule, book, or add something to their calendar
    (distinct from set_reminder, which pings in Discord rather than putting
    something on the actual calendar — use this tool when the user
    specifically says "calendar" or clearly wants it to show up there).

    :param summary: The event title.
    :param start_time_iso: Start date/time in ISO-8601 (e.g. '2026-07-25T14:00:00' or, for all_day, just '2026-07-25').
    :param end_time_iso: Optional. End date/time in ISO-8601. Defaults to 1 hour after start for timed events, or the same day for all-day events.
    :param description: Optional. Event notes/description.
    :param location: Optional. Event location.
    :param all_day: Optional. If true, treats start/end as whole dates rather than specific times.
    :param calendar_id: Optional. Which calendar to add it to — defaults to the primary calendar.
    """
    calendar_id = calendar_id or DEFAULT_CALENDAR_ID
    try:
        service = _get_service()
    except RuntimeError as e:
        return f"❌ {e}"

    body = {"summary": summary}
    if description:
        body["description"] = description
    if location:
        body["location"] = location

    try:
        if all_day:
            start_date = start_time_iso[:10]
            end_date = end_time_iso[:10] if end_time_iso else (
                datetime.fromisoformat(start_date) + timedelta(days=1)
            ).strftime("%Y-%m-%d")
            body["start"] = {"date": start_date}
            body["end"] = {"date": end_date}
        else:
            start_dt = _parse_iso(start_time_iso)
            end_dt = _parse_iso(end_time_iso) if end_time_iso else start_dt + timedelta(hours=1)
            if end_dt <= start_dt:
                return "❌ Error: end time must be after the start time."
            body["start"] = {"dateTime": start_dt.isoformat()}
            body["end"] = {"dateTime": end_dt.isoformat()}
    except ValueError:
        return "❌ Error: start_time_iso/end_time_iso must be valid ISO-8601."

    try:
        created = service.events().insert(calendarId=calendar_id, body=body).execute()
    except HttpError as e:
        return f"❌ Google Calendar error: {e}"

    when = _format_dt(
        body["start"].get("date") or body["start"].get("dateTime"), all_day
    )
    return f"✅ Added **{summary}** to your calendar — {when}."


def delete_calendar_event(identifier: str, days_ahead: int = 30, calendar_id: str = None) -> str:
    """
    Deletes an event from the connected Google Calendar. Matched by exact
    Google event ID, or — more commonly — by a snippet of the event's
    title, searched across the next `days_ahead` days. Use this when the
    user asks to cancel, remove, or delete something from their calendar.
    If multiple upcoming events match the title snippet, none are deleted
    and the matches are listed instead so the user can be more specific.

    :param identifier: A Google Calendar event ID, or a piece of the event's title to search for.
    :param days_ahead: How many days ahead to search when matching by title. Defaults to 30.
    :param calendar_id: Optional. Which calendar to delete from — defaults to the primary calendar.
    """
    calendar_id = calendar_id or DEFAULT_CALENDAR_ID
    try:
        service = _get_service()
    except RuntimeError as e:
        return f"❌ {e}"

    # Try as a direct event ID first.
    try:
        service.events().delete(calendarId=calendar_id, eventId=identifier).execute()
        return f"🗑️ Deleted event {identifier}."
    except HttpError as e:
        if e.resp.status not in (404, 410):
            return f"❌ Google Calendar error: {e}"
        # Not a valid event ID (or already gone) — fall through to title search.
        pass

    now = datetime.now(timezone.utc)
    time_min = now.isoformat()
    time_max = (now + timedelta(days=max(1, days_ahead))).isoformat()

    try:
        result = service.events().list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            q=identifier,
            maxResults=25,
        ).execute()
    except HttpError as e:
        return f"❌ Google Calendar error: {e}"

    matches = result.get("items", [])
    if not matches:
        return f"❌ Couldn't find any upcoming event matching '{identifier}' in the next {days_ahead} day(s)."

    if len(matches) > 1:
        lines = [f"⚠️ Found {len(matches)} events matching '{identifier}' — be more specific, or ask me to list your calendar for the exact one:\n"]
        for ev in matches:
            start = ev.get("start", {})
            all_day = "date" in start
            raw = start.get("date") or start.get("dateTime", "")
            when = _format_dt(raw, all_day)
            lines.append(f"- **{ev.get('summary', '(no title)')}** — {when}")
        return "\n".join(lines)

    ev = matches[0]
    try:
        service.events().delete(calendarId=calendar_id, eventId=ev["id"]).execute()
    except HttpError as e:
        return f"❌ Google Calendar error: {e}"

    return f"🗑️ Deleted event: {ev.get('summary', '(no title)')}"


def list_calendars() -> str:
    """
    Lists every calendar available on the connected Google account, along
    with the calendar_id needed to target it in the other calendar tools.
    Use this when the user asks what calendars exist, or wants to act on a
    calendar other than their primary one and you don't know its ID.
    """
    try:
        service = _get_service()
    except RuntimeError as e:
        return f"❌ {e}"

    try:
        result = service.calendarList().list().execute()
    except HttpError as e:
        return f"❌ Google Calendar error: {e}"

    items = result.get("items", [])
    if not items:
        return "No calendars found on this account."

    lines = ["📚 Calendars on this account:\n"]
    for cal in items:
        primary = " (primary)" if cal.get("primary") else ""
        lines.append(f"- {cal.get('summary', '(unnamed)')}{primary} — id: `{cal['id']}`")
    return "\n".join(lines)
