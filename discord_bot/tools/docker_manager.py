import os
import re
import subprocess
from pathlib import Path

# Fail-closed, same reasoning as process_manager.py's ALLOWED_RESTART_SERVICES:
# restart_container takes a raw string from the LLM, so without a list it
# could restart any container on the host. list_containers/get_container_logs
# stay unrestricted since they're read-only.
_ALLOWED_CONTAINERS = {
    s.strip() for s in os.getenv("CONTAINER_RESTART_ALLOWLIST", "").split(",") if s.strip()
}

# repo root: discord_bot/tools/docker_manager.py -> discord_bot/ -> repo root
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

# --- Allowlist admin helpers -------------------------------------------------
# These are intentionally NOT callable by the LLM: bot.py's register_tools()
# only auto-registers public (non-underscore) functions as tool-calling
# targets, so anything meant to stay owner-only via the !allowlist Discord
# command (bot.py) must keep its leading underscore. Do not rename these to
# drop the underscore — doing so would let the LLM add/remove containers from
# the restart allowlist itself, defeating the whole point of the gate.

def get_allowlist() -> set:
    """Read-only snapshot of the current restart allowlist. Safe to expose —
    it's informational only and doesn't grant restart capability by itself."""
    return set(_ALLOWED_CONTAINERS)


def _persist_allowlist_to_env(containers: set) -> None:
    """Writes the given container set back to CONTAINER_RESTART_ALLOWLIST in
    .env, leaving every other line untouched. Raises on failure so the caller
    can report it rather than silently drifting from what's on disk."""
    if not _ENV_PATH.exists():
        raise FileNotFoundError(f".env not found at {_ENV_PATH}")

    text = _ENV_PATH.read_text()
    new_line = f'CONTAINER_RESTART_ALLOWLIST="{",".join(sorted(containers))}"'
    pattern = re.compile(r'^CONTAINER_RESTART_ALLOWLIST=.*$', re.MULTILINE)

    if pattern.search(text):
        text = pattern.sub(new_line, text, count=1)
    else:
        text = text.rstrip("\n") + f"\n{new_line}\n"

    _ENV_PATH.write_text(text)


def _add_to_allowlist(container_name: str) -> str:
    """Owner-only, called from bot.py's !allowlist command handler — never
    registered as an LLM tool. Updates the in-memory set and persists to .env
    so the change survives a bot restart."""
    container_name = container_name.strip()
    if not container_name:
        return "❌ No container name given."
    if container_name in _ALLOWED_CONTAINERS:
        return f"'{container_name}' is already on the allowlist."
    _ALLOWED_CONTAINERS.add(container_name)
    try:
        _persist_allowlist_to_env(_ALLOWED_CONTAINERS)
    except Exception as e:
        _ALLOWED_CONTAINERS.discard(container_name)  # don't leave memory and disk disagreeing
        return f"❌ Failed to persist to .env, change rolled back: {e}"
    return f"✅ '{container_name}' added to the restart allowlist and saved to .env."


def _remove_from_allowlist(container_name: str) -> str:
    """Owner-only, called from bot.py's !allowlist command handler — never
    registered as an LLM tool. Updates the in-memory set and persists to .env
    so the change survives a bot restart."""
    container_name = container_name.strip()
    if container_name not in _ALLOWED_CONTAINERS:
        return f"'{container_name}' isn't on the allowlist."
    _ALLOWED_CONTAINERS.discard(container_name)
    try:
        _persist_allowlist_to_env(_ALLOWED_CONTAINERS)
    except Exception as e:
        _ALLOWED_CONTAINERS.add(container_name)  # don't leave memory and disk disagreeing
        return f"❌ Failed to persist to .env, change rolled back: {e}"
    return f"🗑️ '{container_name}' removed from the restart allowlist and saved to .env."
# -----------------------------------------------------------------------------

def list_containers() -> str:
    """Lists status of all Docker containers."""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "❌ Timed out listing containers."
    return result.stdout

def restart_container(container_name: str) -> str:
    """Restarts a specific Docker container. Only containers listed in the
    CONTAINER_RESTART_ALLOWLIST env var can be restarted."""
    if container_name not in _ALLOWED_CONTAINERS:
        return (
            f"❌ '{container_name}' isn't on the allowed restart list. Add it to "
            f"CONTAINER_RESTART_ALLOWLIST in .env (comma-separated exact container "
            f"names) if it should be restartable."
        )
    try:
        subprocess.run(["docker", "restart", container_name], check=True, timeout=30)
        return f"Container {container_name} is restarting."
    except subprocess.TimeoutExpired:
        return f"❌ Restarting {container_name} timed out after 30s."
    except Exception as e:
        return f"Error: {str(e)}"

def get_container_logs(container_name: str, lines: int = 20) -> str:
    """Views the last N lines of logs for a container."""
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(lines), container_name],
            capture_output=True, text=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        return f"❌ Timed out reading logs for {container_name}."
    return result.stdout

