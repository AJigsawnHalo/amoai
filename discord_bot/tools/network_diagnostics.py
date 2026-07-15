import subprocess

def ping(host: str) -> str:
    """Pings a host to check connectivity."""
    result = subprocess.run(["ping", "-c", "3", host], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else "Ping failed."

def check_port(host: str, port: int) -> str:
    """Checks if a TCP port is open on a host."""
    try:
        # Requires 'nc' (netcat) installed
        subprocess.run(["nc", "-zv", "-w", "2", host, str(port)], capture_output=True, text=True, check=True)
        return f"Port {port} on {host} is OPEN."
    except:
        return f"Port {port} on {host} is CLOSED or unreachable."

