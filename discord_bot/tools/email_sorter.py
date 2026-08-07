import os
import subprocess

# Configurable like the rest of the project's paths (see .env.example) —
# defaults match the original hardcoded values so this is a no-op unless
# you override it.
_VENV_PYTHON = os.getenv("EMAIL_SORT_VENV_PYTHON", "/home/elskiee/.amoai/.venv/bin/python")
_SCRIPT_PATH = os.getenv("EMAIL_SORT_SCRIPT_PATH", "/home/elskiee/.amoai/email-monitor/monitor.py")

def sort_emails() -> str:
    """
    Triggers the email sorting process and returns the logs to the user.
    Use this tool whenever the user asks to sort, check, or clean their emails.
    """
    try:
        result = subprocess.run(
            [_VENV_PYTHON, _SCRIPT_PATH, "sort_emails"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        # Capture the output
        output = result.stdout.strip()
        errors = result.stderr.strip()
        
        # Build a meaningful message
        final_message = ""
        if output:
            final_message += f"**Output:**\n```\n{output}\n```\n"
        if errors:
            final_message += f"**Errors:**\n```\n{errors}\n```\n"
            
        if not output and not errors:
            return "✅ Email monitor executed successfully (no output generated)."
            
        return final_message

    except subprocess.TimeoutExpired:
        return "❌ Email sort timed out after 120s."
    except Exception as e:
        return f"❌ Failed to run email monitor: {str(e)}"
