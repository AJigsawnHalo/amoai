import os
import time
import subprocess
import psutil
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def get_server_status() -> str:
    """
    Returns a formatted system dashboard: CPU, RAM, temperature, disk usage,
    load average, uptime, and the status of any services configured in
    HEALTH_CHECK_SERVICES. Use this tool whenever the user asks about server
    health, resource usage, disk space, uptime, load, or "how's hiryu doing".
    Read-only — does not change anything on the system.
    """
    # 1. Core metrics
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    disk = psutil.disk_usage("/").percent

    # 2. Temperature detection (unchanged from original)
    temp = None
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries:
                    temp = entries[0].current
                    break
    except Exception:
        pass

    # 3. Load average, normalized against core count so the same 🟢/🟡/🔴
    #    thresholds as CPU% stay meaningful (load == core count is "100%")
    load_pct = None
    try:
        load1, _, _ = psutil.getloadavg()
        cores = psutil.cpu_count() or 1
        load_pct = round((load1 / cores) * 100, 1)
    except (AttributeError, OSError):
        pass  # getloadavg isn't available on all platforms

    # 4. Uptime
    uptime_str = None
    try:
        boot_time = psutil.boot_time()
        delta = int(time.time() - boot_time)
        days, rem = divmod(delta, 86400)
        hours, rem = divmod(rem, 3600)
        minutes, _ = divmod(rem, 60)
        parts = []
        if days: parts.append(f"{days}d")
        if hours: parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        uptime_str = " ".join(parts)
    except Exception:
        pass

    # 5. Helper to determine status based on thresholds
    def get_indicator(value, warn=70, crit=90):
        if value is None: return "N/A"
        if value < warn: return f"{value}% 🟢"
        if value < crit: return f"{value}% 🟡"
        return f"{value}% 🔴"

    def get_temp_indicator(value):
        if value is None: return "N/A"
        if value < 60: return f"{value}°C 🟢"
        if value < 80: return f"{value}°C 🟡"
        return f"{value}°C 🔴"

    # 6. Configured service checks, e.g. HEALTH_CHECK_SERVICES=discord-bot,nginx
    service_lines = []
    services_raw = os.getenv("HEALTH_CHECK_SERVICES", "")
    services = [s.strip() for s in services_raw.split(",") if s.strip()]
    for svc in services:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", svc], capture_output=True, text=True, timeout=5
            )
            status = result.stdout.strip()
            icon = "🟢" if status == "active" else "🔴"
            service_lines.append(f"  {icon} {svc}: {status}")
        except Exception as e:
            service_lines.append(f"  🔴 {svc}: error ({e})")

    # 7. Formatted output
    output = (
        f"\n{'='*5} 🖥️  SYSTEM STATUS {'='*5}\n"
        f"CPU  : {get_indicator(cpu)}\n"
        f"RAM  : {get_indicator(ram)}\n"
        f"DISK : {get_indicator(disk)}\n"
        f"LOAD : {get_indicator(load_pct)}\n"
        f"TEMP : {get_temp_indicator(temp)}\n"
        f"UP   : {uptime_str or 'N/A'}\n"
        f"{'='*25}"
    )
    if service_lines:
        output += "\nSERVICES:\n" + "\n".join(service_lines)

    return output


# Example usage:
if __name__ == "__main__":
    print(get_server_status())
