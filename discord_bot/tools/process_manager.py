import os
import subprocess

# Fail-closed: nothing is restartable until explicitly allowed. Set
# ALLOWED_RESTART_SERVICES in .env as a comma-separated list of exact
# systemd unit names, e.g. "jellyfin,transmission". Empty/unset means
# restart_service refuses everything — this runs with sudo, so a
# hallucinated or manipulated service_name must not be able to hit
# something like networking/ssh/systemd-resolved.
_ALLOWED_SERVICES = {
    s.strip() for s in os.getenv("ALLOWED_RESTART_SERVICES", "").split(",") if s.strip()
}

def get_top_processes(n=5) -> str:
    """Lists the top N processes consuming CPU/RAM."""
    try:
        # Avoid concatenation error by casting to int
        limit = int(n)
    except (ValueError, TypeError):
        return "Error: 'n' must be an integer or a numeric string."

    # Standard ps command to grab processes
    cmd = ["ps", "aux", "--sort=-%cpu"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return "❌ Timed out reading the process list."

    if result.returncode != 0:
        return "Error accessing processes."
    
    # Safely slice the lines in Python instead of relying on a non-existent '--head' flag
    lines = result.stdout.splitlines()
    # lines[0] is the header, plus the top 'n' processes
    top_processes = "\n".join(lines[:limit + 1])
    return top_processes

def restart_service(service_name: str) -> str:
    """Restarts a systemd service (Requires configured sudo privilege).
    Only services listed in the ALLOWED_RESTART_SERVICES env var can be
    restarted — see the comment at the top of this file for why."""
    if service_name not in _ALLOWED_SERVICES:
        return (
            f"❌ '{service_name}' isn't on the allowed restart list. Add it to "
            f"ALLOWED_RESTART_SERVICES in .env (comma-separated exact unit names) "
            f"if it should be restartable."
        )
    try:
        # This will fail unless the user running the script has passwordless sudo setup
        subprocess.run(
            ["sudo", "systemctl", "restart", service_name],
            check=True, capture_output=True, text=True, timeout=30,
        )
        return f"Successfully restarted {service_name}."
    except subprocess.TimeoutExpired:
        return f"❌ Restarting {service_name} timed out after 30s."
    except subprocess.CalledProcessError as e:
        # Capture stderr to explain exactly why it failed (e.g., "interactive password required")
        error_msg = e.stderr.strip() if e.stderr else str(e)
        return f"Failed to restart {service_name}: {error_msg}"
    except Exception as e:
        return f"Failed to restart {service_name}: {str(e)}"
