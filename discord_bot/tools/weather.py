import subprocess
import os

def run(location: str) -> str:
    """
    Fetches the current weather, temperature, and conditions for a specific 
    location. Use this tool whenever the user asks about the weather, 
    forecasts, or current atmospheric conditions.
    """
    runner_path = os.path.expanduser("~/.amoai/weather-tool/task_runner.py")
    
    try:
        result = subprocess.run(
            ['python3', runner_path, location],
            capture_output=True,
            text=True,
            check=True
        )
        raw_output = result.stdout.strip()
        
        # If the output isn't formatted how we expect, return it raw
        if ": " not in raw_output:
            return f"Raw Weather Output: {raw_output}"

        # Parse logic
        parts = raw_output.split(", ")
        data = {}
        for p in parts:
            if ": " in p:
                key, val = p.split(": ", 1)
                data[key] = val
        
        # Build the "Pretty" Output
        return (
            f"**🌤️ Weather Report for {data.get('Location', location)}**\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"**Conditions:** {data.get('Weather', 'N/A')}\n"
            f"**🌡️ Temperature:** {data.get('Temp', 'N/A')}\n"
            f"**💨 Wind Speed:** {data.get('Wind', 'N/A')}\n"
            f"**💧 Precipitation:** {data.get('Precip', 'N/A')}\n"
            f"**☂️ Chance of Rain:** {data.get('Chance', 'N/A')}\n"
            f"━━━━━━━━━━━━━━━━━━"
        )
        
    except Exception as e:
        return f"⚠️ Bridge Error: {str(e)}"
