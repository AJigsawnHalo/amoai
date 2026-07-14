import psutil

def get_server_status() -> str:
    """
    Returns a formatted system dashboard string with visual status indicators.
    """
    # 1. Metrics
    cpu = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory().percent
    
    # 2. Temperature detection
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
        
    # 3. Helper to determine status based on thresholds
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

    # 4. Formatted Output
    output = (
        f"\n{'='*5} 🖥️  SYSTEM STATUS {'='*5}\n"
        f"CPU  : {get_indicator(cpu)}\n"
        f"RAM  : {get_indicator(ram)}\n"
        f"TEMP : {get_temp_indicator(temp)}\n"
        f"{'='*25}"
    )
    
    return output

# Example usage:
if __name__ == "__main__":
    print(get_server_status())
