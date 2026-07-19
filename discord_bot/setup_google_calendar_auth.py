"""
setup_google_calendar_auth.py

Run this ONCE, manually, from a machine with a browser, to connect a
Google account to the calendar tool. It opens a Google consent screen,
and on approval saves a refresh token that the bot then uses unattended.

Usage:
    python setup_google_calendar_auth.py

Prerequisites (see the top of tools/calendar_tool.py for full detail):
    1. A Google Cloud project with the "Google Calendar API" enabled.
    2. An OAuth Client ID of type "Desktop app", downloaded as JSON and
       saved to data/google_calendar_credentials.json (or wherever
       GOOGLE_CALENDAR_CREDENTIALS_FILE in .env points).

If you run this on a machine other than your bot's server (e.g. your
laptop, because the server has no browser), copy the resulting
data/google_calendar_token.json over to the server afterward, to the same
path GOOGLE_CALENDAR_TOKEN_FILE points to there.
"""

import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

load_dotenv(find_dotenv())

SCOPES = ["https://www.googleapis.com/auth/calendar"]

_DATA_DIR = Path(__file__).resolve().parent / "data"
CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CALENDAR_CREDENTIALS_FILE", _DATA_DIR / "google_calendar_credentials.json"))
TOKEN_FILE = Path(os.getenv("GOOGLE_CALENDAR_TOKEN_FILE", _DATA_DIR / "google_calendar_token.json"))


def main():
    if not CREDENTIALS_FILE.exists():
        print(f"❌ Couldn't find {CREDENTIALS_FILE}")
        print("   Download an OAuth Client ID (type: Desktop app) from Google Cloud")
        print("   Console and save it at that path first.")
        return

    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)

    flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
    # Fixed port so it can be reached through an SSH tunnel on a headless
    # server (e.g. `ssh -L 2658:localhost:2658 user@server`), rather than a
    # random port that would change every run.
    # open_browser=False: on a headless server, webbrowser.open() may find
    # something like w3m on PATH and try to launch that instead of doing
    # nothing — but a text browser can't complete Google's login (no JS/
    # cookies support), so it'd just hang. This skips that entirely and
    # prints the URL for you to paste into your own local browser instead.
    creds = flow.run_local_server(port=2658, open_browser=False)

    TOKEN_FILE.write_text(creds.to_json())
    print(f"✅ Connected! Token saved to {TOKEN_FILE}")
    print("   If your bot runs on a different machine, copy this file over")
    print("   to the same GOOGLE_CALENDAR_TOKEN_FILE path there.")


if __name__ == "__main__":
    main()
