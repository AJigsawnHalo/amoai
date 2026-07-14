import psutil

def get_server_status() -> str:
    """
    Returns the current CPU, RAM usage, and system temperature (if available).
    Use this tool whenever the user asks about system performance, health, or status.
    """
    # 1. Standard metrics
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    
    # 2. Temperature detection
    temp_str = ""
    try:
        # sensors_temperatures() returns a dict of lists
        temps = psutil.sensors_temperatures()
        if temps:
            # We grab the first available sensor value (usually the CPU package temp)
            for name, entries in temps.items():
                if entries:
                    # 'current' is the temperature in Celsius
                    temp_str = f" | Temp: {entries[0].current}°C"
                    break
    except Exception:
        # Fallback if sensors are not accessible (common in some VMs/containers)
        temp_str = " | Temp: N/A"
        
    return f"CPU: {cpu}% | RAM: {ram}%{temp_str}"
