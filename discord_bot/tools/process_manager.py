import subprocess

def get_top_processes(n=5) -> str:
    """Lists the top N processes consuming CPU/RAM."""
    try:
        # Avoid concatenation error by casting to int
        limit = int(n)
    except (ValueError, TypeError):
        return "Error: 'n' must be an integer or a numeric string."

    # Standard ps command to grab processes
    cmd = ["ps", "aux", "--sort=-%cpu"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return "Error accessing processes."
    
    # Safely slice the lines in Python instead of relying on a non-existent '--head' flag
    lines = result.stdout.splitlines()
    # lines[0] is the header, plus the top 'n' processes
    top_processes = "\n".join(lines[:limit + 1])
    return top_processes

def restart_service(service_name: str) -> str:
    """Restarts a systemd service (Requires configured sudo privilege)."""
    try:
        # This will fail unless the user running the script has passwordless sudo setup
        subprocess.run(["sudo", "systemctl", "restart", service_name], check=True, capture_output=True, text=True)
        return f"Successfully restarted {service_name}."
    except subprocess.CalledProcessError as e:
        # Capture stderr to explain exactly why it failed (e.g., "interactive password required")
        error_msg = e.stderr.strip() if e.stderr else str(e)
        return f"Failed to restart {service_name}: {error_msg}"
    except Exception as e:
        return f"Failed to restart {service_name}: {str(e)}"
