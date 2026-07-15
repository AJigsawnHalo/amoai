import subprocess

def sort_emails() -> str:
    """
    Triggers the email sorting process and returns the logs to the user.
    Use this tool whenever the user asks to sort, check, or clean their emails.
    """
    venv_python = "/home/elskiee/.amoai/.venv/bin/python"
    # Assuming you want to trigger the logic-based script, not the infinite loop monitor
    script_path = "/home/elskiee/.amoai/email-monitor/monitor.py"

    try:
        result = subprocess.run(
            [venv_python, script_path, "sort_emails"], 
            capture_output=True, 
            text=True
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

    except Exception as e:
        return f"❌ Failed to run email monitor: {str(e)}"
