import os
import requests
from dotenv import load_dotenv, find_dotenv

# Automatically crawls up parent directories to locate your central .env file
load_dotenv(find_dotenv())

# --- CONFIGURATION ---
HA_URL = os.getenv("HA_URL", "http://homeassistant.local:8123")
HA_TOKEN = os.getenv("HA_TOKEN")  # Long-Lived Access Token from your HA profile page
HA_TIMEOUT = 10


def _headers():
    return {
        "Authorization": f"Bearer {HA_TOKEN}",
        "Content-Type": "application/json",
    }


def _request(method, path, json_data=None):
    """Internal helper. Raises for HTTP errors so callers can wrap in try/except."""
    if not HA_TOKEN:
        raise RuntimeError("HA_TOKEN is not set in .env")
    url = f"{HA_URL.rstrip('/')}{path}"
    resp = requests.request(
        method, url, headers=_headers(), json=json_data, timeout=HA_TIMEOUT
    )
    resp.raise_for_status()
    if resp.text:
        return resp.json()
    return {}


def _call_service(domain, service, entity_id):
    """Shared plumbing for every on/off/climate/script/automation call below."""
    return _request("POST", f"/api/services/{domain}/{service}", {"entity_id": entity_id})


# --- READ TOOLS ---

def ha_get_state(entity_id: str) -> str:
    """
    Gets the current state and attributes of a single Home Assistant entity
    (a sensor, light, switch, climate device, lock, etc). Use this whenever the
    user asks what the temperature is, whether a light or lock is on/off,
    or the status of any device by its entity_id (e.g. 'sensor.living_room_temperature',
    'climate.bedroom_ac', 'light.kitchen').
    """
    try:
        data = _request("GET", f"/api/states/{entity_id}")
        state = data.get("state", "unknown")
        attrs = data.get("attributes", {})
        friendly = attrs.get("friendly_name", entity_id)
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items() if k != "friendly_name")
        return f"{friendly} ({entity_id}): {state}" + (f" [{attr_str}]" if attr_str else "")
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return f"❌ No entity found with id '{entity_id}'."
        return f"❌ Home Assistant error: {e}"
    except Exception as e:
        return f"❌ Failed to reach Home Assistant: {e}"


def ha_list_entities(domain: str = "") -> str:
    """
    Lists entities currently known to Home Assistant, optionally filtered to one
    domain (e.g. 'light', 'switch', 'climate', 'sensor', 'lock', 'automation',
    'script'). Use this when the user isn't sure of the exact entity_id and you
    need to look one up before calling another tool, or when they ask what
    devices/sensors/automations exist.
    """
    try:
        data = _request("GET", "/api/states")
        if domain:
            data = [e for e in data if e["entity_id"].startswith(f"{domain}.")]
        if not data:
            return f"No entities found" + (f" for domain '{domain}'." if domain else ".")
        lines = []
        for e in data[:50]:  # keep responses bounded
            friendly = e.get("attributes", {}).get("friendly_name", e["entity_id"])
            lines.append(f"- {e['entity_id']} ({friendly}): {e['state']}")
        suffix = "\n...(truncated, narrow with a domain)" if len(data) > 50 else ""
        return "\n".join(lines) + suffix
    except Exception as e:
        return f"❌ Failed to reach Home Assistant: {e}"


# --- LIGHTS / SWITCHES / GENERIC ON-OFF ---

def ha_turn_on(entity_id: str) -> str:
    """
    Turns on a Home Assistant entity — a light, switch, fan, or climate device
    (e.g. 'light.kitchen', 'switch.space_heater', 'climate.bedroom_ac'). Use this
    whenever the user asks to turn something on. The domain is inferred from the
    entity_id prefix.
    """
    domain = entity_id.split(".")[0] if "." in entity_id else "homeassistant"
    try:
        _call_service(domain, "turn_on", entity_id)
        return f"✅ Turned on {entity_id}."
    except Exception as e:
        return f"❌ Failed to turn on {entity_id}: {e}"


def ha_turn_off(entity_id: str) -> str:
    """
    Turns off a Home Assistant entity — a light, switch, fan, or climate device
    (e.g. 'light.kitchen', 'switch.space_heater', 'climate.bedroom_ac'). Use this
    whenever the user asks to turn something off. The domain is inferred from the
    entity_id prefix.
    """
    domain = entity_id.split(".")[0] if "." in entity_id else "homeassistant"
    try:
        _call_service(domain, "turn_off", entity_id)
        return f"✅ Turned off {entity_id}."
    except Exception as e:
        return f"❌ Failed to turn off {entity_id}: {e}"


def ha_toggle(entity_id: str) -> str:
    """
    Toggles a Home Assistant entity to the opposite of its current on/off state
    (e.g. a light that's on gets turned off, and vice versa). Use this when the
    user just says 'toggle' or 'flip' a device rather than specifying on or off.
    """
    domain = entity_id.split(".")[0] if "." in entity_id else "homeassistant"
    try:
        _call_service(domain, "toggle", entity_id)
        return f"✅ Toggled {entity_id}."
    except Exception as e:
        return f"❌ Failed to toggle {entity_id}: {e}"


# --- CLIMATE / AC CONTROL ---

def ha_set_climate_temperature(entity_id: str, temperature: float) -> str:
    """
    Sets the target temperature on a climate device such as an AC unit or
    thermostat (e.g. entity_id 'climate.bedroom_ac', temperature 72). Use this
    when the user asks to set, raise, or lower the AC/heat/thermostat to a
    specific degree value.
    """
    try:
        _request(
            "POST",
            "/api/services/climate/set_temperature",
            {"entity_id": entity_id, "temperature": temperature},
        )
        return f"✅ Set {entity_id} target temperature to {temperature}."
    except Exception as e:
        return f"❌ Failed to set temperature on {entity_id}: {e}"


def ha_set_climate_hvac_mode(entity_id: str, hvac_mode: str) -> str:
    """
    Sets the operating mode of a climate device, e.g. entity_id 'climate.bedroom_ac'
    with hvac_mode one of: 'off', 'cool', 'heat', 'heat_cool', 'auto', 'dry', 'fan_only'.
    Use this when the user asks to switch the AC/thermostat mode, not just the
    temperature (e.g. 'switch the AC to cooling mode', 'turn off the thermostat').
    """
    try:
        _request(
            "POST",
            "/api/services/climate/set_hvac_mode",
            {"entity_id": entity_id, "hvac_mode": hvac_mode},
        )
        return f"✅ Set {entity_id} mode to {hvac_mode}."
    except Exception as e:
        return f"❌ Failed to set mode on {entity_id}: {e}"


# --- SELECT ENTITIES (dropdowns) ---

def ha_select_option(entity_id: str, option: str) -> str:
    """
    Sets a Home Assistant 'select' entity to one of its predefined dropdown
    options (e.g. entity_id 'select.vacuum_cleaning_mode' with option 'quiet',
    or a mode/preset selector like 'select.washer_cycle'). Use this when the
    user asks to change a mode, preset, or dropdown-style setting rather than a
    simple on/off or a numeric value. The option must match one of the choices
    exactly as reported by ha_get_state's 'options' attribute for that entity —
    call ha_get_state first if you aren't sure of the valid options.
    """
    try:
        _request(
            "POST",
            "/api/services/select/select_option",
            {"entity_id": entity_id, "option": option},
        )
        return f"✅ Set {entity_id} to '{option}'."
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 400:
            return (f"❌ '{option}' isn't a valid option for {entity_id}. "
                     f"Check ha_get_state's 'options' attribute for the valid choices.")
        return f"❌ Home Assistant error: {e}"
    except Exception as e:
        return f"❌ Failed to set {entity_id}: {e}"


# --- AUTOMATIONS / SCRIPTS ---

def ha_run_script(entity_id: str) -> str:
    """
    Runs a Home Assistant script by its entity_id (e.g. 'script.good_night').
    Use this when the user asks to run/trigger/fire a named script.
    """
    try:
        _call_service("script", "turn_on", entity_id)
        return f"✅ Ran script {entity_id}."
    except Exception as e:
        return f"❌ Failed to run script {entity_id}: {e}"


def ha_trigger_automation(entity_id: str) -> str:
    """
    Manually triggers a Home Assistant automation by its entity_id
    (e.g. 'automation.evening_lights'), regardless of its normal trigger
    conditions. Use this when the user asks to run/trigger/fire an automation
    right now.
    """
    try:
        _call_service("automation", "trigger", entity_id)
        return f"✅ Triggered automation {entity_id}."
    except Exception as e:
        return f"❌ Failed to trigger automation {entity_id}: {e}"


def ha_set_automation_enabled(entity_id: str, enabled: bool) -> str:
    """
    Enables or disables a Home Assistant automation without deleting it
    (entity_id like 'automation.evening_lights', enabled True or False). Use
    this when the user asks to turn an automation on/off or pause/resume it.
    """
    service = "turn_on" if enabled else "turn_off"
    try:
        _call_service("automation", service, entity_id)
        state = "enabled" if enabled else "disabled"
        return f"✅ Automation {entity_id} {state}."
    except Exception as e:
        return f"❌ Failed to update automation {entity_id}: {e}"
