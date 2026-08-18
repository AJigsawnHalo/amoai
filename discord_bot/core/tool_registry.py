"""
tool_registry.py — discovers tools/ modules at startup, builds their Ollama
function-calling schemas from docstrings, and tracks which ones need a
confirmation reaction before running. Also owns the tool-call audit log.

OLLAMA_SCHEMAS and TOOL_REGISTRY are mutable — populated once by
register_tools() at import time, then read everywhere else in core/.
Callers must `import tool_registry` and reference
`tool_registry.OLLAMA_SCHEMAS` / `tool_registry.TOOL_REGISTRY`, never
`from tool_registry import OLLAMA_SCHEMAS`, which would capture an empty
list/dict from before register_tools() ran.
"""
import inspect
import importlib
import json
import pkgutil
import re
from datetime import datetime, timezone

import tools
import config

# --- DYNAMIC REGISTRY ---
OLLAMA_SCHEMAS = []
TOOL_REGISTRY = {}

# --- CONFIRMATION-GATED TOOLS ---
CONFIRMATION_REQUIRED_TOOLS = {
    "restart_service", "restart_container", "nyaadle_check_now",
    "move_file", "delete_file", "delete_calendar_event", "clear_failure_logs",
}
OVERWRITE_GATED_TOOLS = {"write_file", "copy_file", "move_file"}


def needs_confirmation(name: str, args: dict) -> bool:
    if name in CONFIRMATION_REQUIRED_TOOLS:
        return True
    if name in OVERWRITE_GATED_TOOLS and args.get("overwrite") is True:
        return True
    return False


def map_python_type_to_json(py_type):
    mapping = {str: "string", int: "number", float: "number", bool: "boolean"}
    return mapping.get(py_type, "string")


# Matches a ":param name: description text" line, capturing the name and
# the description. The description can wrap onto following indented lines
# (anything not starting a new ":param"/":return" tag), which is how the
# tool docstrings in tools/ are written for longer explanations.
_PARAM_LINE_RE = re.compile(r"^\s*:param\s+(\w+):\s?(.*)$")


def _parse_docstring(doc: str) -> "tuple[str, dict[str, str]]":
    """Splits a Google/Sphinx-style tool docstring into (summary, params).

    summary is everything before the first ':param' line, dedented and
    stripped — this becomes the tool-level description. params maps each
    documented parameter name to its description text, so it can be
    attached to that parameter's own schema entry instead of being left
    buried inside one big blob of text the model has to parse itself out
    of prose.
    """
    if not doc:
        return "No description", {}

    lines = doc.splitlines()
    summary_lines = []
    params: dict = {}
    current_param = None

    for line in lines:
        match = _PARAM_LINE_RE.match(line)
        if match:
            current_param = match.group(1)
            params[current_param] = match.group(2).strip()
        elif current_param is not None:
            # Continuation of the previous :param's description, unless
            # this line is blank (end of the docstring's tag block) or
            # looks like a new tag (":return:", ":raises:", etc.).
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                current_param = None
            else:
                params[current_param] += " " + stripped
        else:
            summary_lines.append(line)

    summary = "\n".join(summary_lines).strip()
    return (summary or "No description"), params


def register_tools():
    print("[SYSTEM] Discovering tools...")
    for _, module_name, _ in pkgutil.iter_modules(tools.__path__):
        module = importlib.import_module(f"tools.{module_name}")
        for attr_name in dir(module):
            func = getattr(module, attr_name)
            if callable(func) and not inspect.isclass(func) and not attr_name.startswith('_') and getattr(func, '__module__', None) == f"tools.{module_name}":
                sig = inspect.signature(func)
                params = sig.parameters
                summary, param_docs = _parse_docstring(func.__doc__)

                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                for name, param in params.items():
                    if name == "user_id":
                        continue
                    prop = {"type": map_python_type_to_json(param.annotation)}
                    if name in param_docs:
                        prop["description"] = param_docs[name]
                    parameters["properties"][name] = prop
                    if param.default == inspect.Parameter.empty:
                        parameters["required"].append(name)

                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": summary,
                        "parameters": parameters
                    }
                }
                OLLAMA_SCHEMAS.append(tool_schema)
                TOOL_REGISTRY[func.__name__] = func
                print(f"[SYSTEM] Loaded tool: {func.__name__}")


register_tools()

from tools.job_manager import set_tool_registry
set_tool_registry(TOOL_REGISTRY)

# --- TOOL CALL AUDIT LOG ---
TOOL_LOG_FILE = config.DATA_DIR / "tool_call_log.jsonl"


def log_tool_call(name: str, args: dict, result, source: str = "llm"):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "tool": name,
        "args": args,
        "result": str(result)[:500],
    }
    try:
        with open(TOOL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"[AUDIT] Failed to write tool log: {e}")
