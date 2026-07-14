import sys
import urllib.request
import urllib.parse
import argparse

def get_weather(location: str):
    # The format string for the weather report
    fmt = "Location: %l, Weather: %C, Temp: %t, Wind: %w, Precip: %p, Chance: %o"
    
    # URL encode the location to ensure characters like spaces/symbols don't break the URL
    safe_location = urllib.parse.quote(location)
    url = f"https://wttr.in/{safe_location}?format={urllib.parse.quote(fmt)}"
    
    try:
        # Use Python's native network library
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.read().decode('utf-8').strip()
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("location", type=str)
    args = parser.parse_args()
    print(get_weather(args.location))
