import urllib.request
import urllib.parse
import json

def check_weather(location: str) -> str:
    """
    Fetches weather data for a location and returns a structured, machine-readable
    JSON string. All units are guaranteed to be in metric.
    """
    # %l: Location, %C: Weather, %t: Temp, %f: Feels Like, %w: Wind, %p: Precip (mm), %o: Chance
    fmt = "Location: %l, Weather: %C, Temp: %t, FeelsLike: %f, Wind: %w, Precip: %p, Chance: %o"
    
    safe_location = urllib.parse.quote(location)
    safe_fmt = urllib.parse.quote(fmt)
    # The '?m' parameter forces metric units (Celsius, km/h, mm)
    url = f"https://wttr.in/{safe_location}?m&format={safe_fmt}"
    
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'curl/7.64.1'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            raw_output = response.read().decode('utf-8').strip()
            
        # Fallback if wttr.in returns an unparseable error page
        if ": " not in raw_output:
            return json.dumps({
                "status": "error",
                "message": f"Raw output not in expected format: {raw_output}"
            })

        # Parse key-value string into a dictionary
        parts = raw_output.split(", ")
        data = {}
        for p in parts:
            if ": " in p:
                key, val = p.split(": ", 1)
                data[key.strip()] = val.strip()
        
        # Enforce and format 'mm/h' unit representation
        precip = data.get('Precip', '0mm')
        if 'mm' in precip and '/h' not in precip:
            precip = precip.replace('mm', ' mm/h')
        elif 'mm' not in precip:
            precip = f"{precip} mm/h"

        # Build clean, machine-readable output
        structured_output = {
            "status": "success",
            "location": data.get("Location", location),
            "conditions": data.get("Weather", "N/A"),
            "temperature": data.get("Temp", "N/A"),
            "feels_like_temperature": data.get("FeelsLike", "N/A"),
            "wind_speed": data.get("Wind", "N/A"),
            "precipitation_intensity": precip,
            "chance_of_rain": data.get("Chance", "N/A")
        }
        
        return json.dumps(structured_output, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e)
        })
