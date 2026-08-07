import os
import subprocess

# Fail-closed, same reasoning as process_manager.py's ALLOWED_RESTART_SERVICES:
# restart_container takes a raw string from the LLM, so without a list it
# could restart any container on the host. list_containers/get_container_logs
# stay unrestricted since they're read-only.
_ALLOWED_CONTAINERS = {
    s.strip() for s in os.getenv("CONTAINER_RESTART_ALLOWLIST", "").split(",") if s.strip()
}

def list_containers() -> str:
    """Lists status of all Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "❌ Timed out listing containers."
    return result.stdout

def restart_container(container_name: str) -> str:
    """Restarts a specific Docker container. Only containers listed in the
    CONTAINER_RESTART_ALLOWLIST env var can be restarted."""
    if container_name not in _ALLOWED_CONTAINERS:
        return (
            f"❌ '{container_name}' isn't on the allowed restart list. Add it to "
            f"CONTAINER_RESTART_ALLOWLIST in .env (comma-separated exact container "
            f"names) if it should be restartable."
        )
    try:
        subprocess.run(["docker", "restart", container_name], check=True, timeout=30)
        return f"Container {container_name} is restarting."
    except subprocess.TimeoutExpired:
        return f"❌ Restarting {container_name} timed out after 30s."
    except Exception as e:
        return f"Error: {str(e)}"

def get_container_logs(container_name: str, lines: int = 20) -> str:
    """Views the last N lines of logs for a container."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container_name],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return f"❌ Timed out reading logs for {container_name}."
    return result.stdout

