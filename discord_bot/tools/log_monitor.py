import subprocess

def check_system_logs(hours: int = 0) -> str:
    """
    Checks system logs for critical events (crashes, service failures, disk/memory
    exhaustion, security issues) and returns a summary. Automatically sends a
    Discord alert if anything critical is found. Use this tool whenever the user
    asks to check logs, check system/server health, or see if anything is wrong.

    Args:
        hours: optional ad-hoc lookback window in hours (e.g. 2 for "check the
               last 2 hours"). Leave as 0 to check only what's new since the
               last automated run (the normal cron/systemd behavior).
    """
    venv_python = "/home/elskiee/.amoai/.venv/bin/python"
    script_path = "/home/elskiee/.amoai/log-monitor/monitor.py"

    # --dry-run: still scans and updates the cursor/offset normally, it just
    # skips send_discord_alert(). We report results back through the bot's own
    # message instead — the webhook stays reserved for the cron/systemd run.
    cmd = [venv_python, script_path, "check", "--dry-run"]
    if hours and hours > 0:
        cmd += ["--since", f"{hours} hours ago"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        output = result.stdout.strip()
        errors = result.stderr.strip()

        final_message = ""
        if output:
            final_message += f"**Output:**\n```\n{output}\n```\n"
        if errors:
            final_message += f"**Errors:**\n```\n{errors}\n```\n"

        if not output and not errors:
            return "✅ Log monitor executed successfully (no output generated)."

        return final_message

    except Exception as e:
        return f"❌ Failed to run log monitor: {str(e)}"
