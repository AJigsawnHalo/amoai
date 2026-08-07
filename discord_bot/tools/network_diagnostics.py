import subprocess

def ping(host: str) -> str:
    """Pings a host to check connectivity."""
    try:
        result = subprocess.run(
            ["ping", "-c", "3", "-W", "3", host],
            capture_output=True, text=True, timeout=15,
        )
        return result.stdout if result.returncode == 0 else "Ping failed."
    except subprocess.TimeoutExpired:
        return f"❌ Ping to {host} timed out."

def check_port(host: str, port: int) -> str:
    """Checks if a TCP port is open on a host."""
    try:
        # Requires 'nc' (netcat) installed
        subprocess.run(
            ["nc", "-zv", "-w", "2", host, str(port)],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return f"Port {port} on {host} is OPEN."
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return f"Port {port} on {host} is CLOSED or unreachable."
    except Exception as e:
        return f"❌ Failed to check port {port} on {host}: {e}"

