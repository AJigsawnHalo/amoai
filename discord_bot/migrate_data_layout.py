#!/usr/bin/env python3
"""
migrate_data_layout.py — One-time move of runtime data files into the new
discord_bot/data/ and discord_bot/debug/ layout, ahead of the bot.py ->
core/ module split.

WHY THIS EXISTS
----------------
The refactored code (core/config.py, core/embeddings.py, core/memory_store.py,
core/tool_registry.py, core/llm.py, plus tools/reminder_tool.py and
tools/scratchpad_tool.py) will look for its files in their NEW locations.
This script physically moves the OLD files there first, so the bot doesn't
start up against an empty data/ dir and silently begin writing fresh state.

Run this BEFORE deploying the refactored code, while the old bot.py layout
is still what's on disk. Safe to re-run: anything already moved, or not
found, is skipped rather than treated as an error.

USAGE
-----
    # stop the service first so nothing is writing to these files mid-move
    sudo systemctl stop discord-bot.service

    cd ~/.amoai/amoai/discord_bot   # or wherever this repo lives on hiryu
    python3 migrate_data_layout.py --dry-run   # see what it would do
    python3 migrate_data_layout.py             # actually move things

    sudo systemctl start discord-bot.service   # only after deploying the
                                                # refactored code too
"""

import argparse
import shutil
import sys
from pathlib import Path

# This script lives at discord_bot/migrate_data_layout.py, so its own
# parent IS the discord_bot/ base directory — everything below is relative
# to that, regardless of where the repo is checked out.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DEBUG_DIR = BASE_DIR / "debug"

# (source relative to BASE_DIR, destination dir) — files that currently
# sit directly in discord_bot/ and are moving into data/.
DATA_MOVES = [
    ("reminders.json", DATA_DIR),
    ("scratchpad.json", DATA_DIR),
    ("tool_call_log.jsonl", DATA_DIR),
    ("tool_embedding_cache.json", DATA_DIR),
    ("tool_embedding_cache_local.json", DATA_DIR),
]

# google_calendar_credentials.json, google_calendar_token.json, and
# memory_store.sqlite3 are already in data/ today (calendar_tool.py and
# bot.py already point there) — nothing to do for those.

# memory_store.json (the legacy pre-SQLite fact file) is deliberately NOT
# moved. It's only ever read once, by the one-time migration-into-SQLite
# function, and has no place in the new layout. Leave it where it is.


def find_failed_payloads():
    """failed_payload_*.json dumps are timestamped and open-ended, so they
    need a glob rather than a fixed name."""
    return sorted(BASE_DIR.glob("failed_payload_*.json"))


def move_one(src: Path, dest_dir: Path, dry_run: bool) -> str:
    if not src.exists():
        return f"skip (not found): {src.relative_to(BASE_DIR)}"

    dest = dest_dir / src.name
    if dest.exists():
        return (
            f"skip (already present at destination): {src.relative_to(BASE_DIR)} "
            f"-> {dest.relative_to(BASE_DIR)}"
        )

    if dry_run:
        return f"would move: {src.relative_to(BASE_DIR)} -> {dest.relative_to(BASE_DIR)}"

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return f"moved: {src.relative_to(BASE_DIR)} -> {dest.relative_to(BASE_DIR)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would move without touching anything.",
    )
    args = parser.parse_args()

    if not args.dry_run:
        print(
            "This will move live bot state files. Make sure discord-bot.service "
            "is stopped first.\n"
        )
        confirm = input("Type 'yes' to continue: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(1)

    results = []

    for name, dest_dir in DATA_MOVES:
        results.append(move_one(BASE_DIR / name, dest_dir, args.dry_run))

    failed_payloads = find_failed_payloads()
    if not failed_payloads:
        results.append("skip (none found): failed_payload_*.json")
    else:
        for payload in failed_payloads:
            results.append(move_one(payload, DEBUG_DIR, args.dry_run))

    print("\n".join(results))
    print()
    if args.dry_run:
        print("Dry run only — nothing was moved. Re-run without --dry-run to apply.")
    else:
        print("Done. Verify data/ and debug/ contents before deploying the refactored code.")


if __name__ == "__main__":
    main()
