#!/usr/bin/env python3
"""
merge_env.py — Merge values from an existing .env into the structure/comments
of a new .env.example, so you get the new template's layout with your old
secrets carried over automatically.

100% local. No network calls. No printing of secret values to stdout/stderr
(only key names are logged, to stderr).

Usage:
    python3 merge_env.py OLD_ENV NEW_TEMPLATE > MERGED_ENV
    # e.g.
    python3 merge_env.py .env .env.example > .env.merged

Then review .env.merged yourself and rename it to .env when you're happy.
"""
import re
import sys

KEY_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)=', re.MULTILINE)


def find_value_span(text, value_start):
    """
    Given the index right after 'KEY=', return the end index of the value:
    - If the value starts with a quote char (' or "), scan for the matching
      closing quote (supports multi-line values like EMAIL_ACCOUNTS' JSON).
    - Otherwise, the value ends at the next newline (single-line value).
    Returns index just past the end of the value (not including trailing \\n).
    """
    if value_start < len(text) and text[value_start] in ("'", '"'):
        quote = text[value_start]
        i = value_start + 1
        while i < len(text):
            if text[i] == quote and text[i - 1] != "\\":
                return i + 1
            i += 1
        return len(text)  # unterminated quote; take rest of file
    else:
        nl = text.find("\n", value_start)
        return nl if nl != -1 else len(text)


def parse_env_file(path):
    """Parse a .env-style file into an ordered dict of KEY -> raw value text
    (value text includes surrounding quotes exactly as written)."""
    with open(path, "r") as f:
        text = f.read()

    values = {}
    for m in KEY_RE.finditer(text):
        key = m.group(1)
        value_start = m.end()
        value_end = find_value_span(text, value_start)
        values[key] = text[value_start:value_end]
    return values


def merge(old_env_path, template_path, out=sys.stdout):
    old_values = parse_env_file(old_env_path)
    used_keys = set()

    with open(template_path, "r") as f:
        template_text = f.read()

    matches = list(KEY_RE.finditer(template_text))
    cursor = 0
    report = []

    for m in matches:
        key = m.group(1)
        # Preserve everything since the last value (comments, headers,
        # blank lines) exactly as-is.
        out.write(template_text[cursor:m.start()])
        out.write(f"{key}=")

        value_start = m.end()
        value_end = find_value_span(template_text, value_start)

        if key in old_values:
            out.write(old_values[key])
            used_keys.add(key)
            report.append((key, "carried over from old .env"))
        else:
            out.write(template_text[value_start:value_end])
            report.append((key, "kept new template default (not in old .env)"))

        cursor = value_end

    out.write(template_text[cursor:])

    print("\n--- merge report (key names only, no values shown) ---", file=sys.stderr)
    for key, note in report:
        print(f"  {key}: {note}", file=sys.stderr)

    unused = sorted(set(old_values) - used_keys)
    if unused:
        print("\nKeys in your OLD .env that do NOT exist in the new template", file=sys.stderr)
        print("(not carried over — add manually if still needed):", file=sys.stderr)
        for key in unused:
            print(f"  {key}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} OLD_ENV NEW_TEMPLATE > MERGED_OUTPUT", file=sys.stderr)
        sys.exit(1)
    merge(sys.argv[1], sys.argv[2])
