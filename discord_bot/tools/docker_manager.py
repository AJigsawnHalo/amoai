import subprocess

def list_containers() -> str:
    """Lists status of all Docker containers."""
    result = subprocess.run(["docker", "ps", "-a", "--format", "table {{.Names}}\t{{.Status}}"], capture_output=True, text=True)
    return result.stdout

def restart_container(container_name: str) -> str:
    """Restarts a specific Docker container."""
    try:
        subprocess.run(["docker", "restart", container_name], check=True)
        return f"Container {container_name} is restarting."
    except Exception as e:
        return f"Error: {str(e)}"

def get_container_logs(container_name: str, lines: int = 20) -> str:
    """Views the last N lines of logs for a container."""
    result = subprocess.run(["docker", "logs", "--tail", str(lines), container_name], capture_output=True, text=True)
    return result.stdout

