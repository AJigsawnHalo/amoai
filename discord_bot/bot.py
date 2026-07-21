import os
import sys
import asyncio
import json
import hashlib
import time
import base64
import io
import importlib
import pkgutil
import inspect
import aiohttp
from aiohttp import web
import discord
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict, deque
from discord.ext import commands, tasks
from dotenv import load_dotenv, find_dotenv
import tools
from tools.reminder_tool import _get_due_arrival_reminders, _get_due_time_reminders, BOT_TIMEZONE

# --- CONFIGURATION ---
load_dotenv(find_dotenv())
MODEL_NAME = "gemma4:cloud"
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_API = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{{model}}:embedContent"
)
# Last-resort embedding path — only reached when Gemini's embedding call
# fails AND chat is already running on the local fallback model (see
# LAST_CHAT_BACKEND below). Reuses the same local Ollama instance the bot
# already talks to for chat, and the same model rag_knowledge.py uses.
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text")
LOCAL_EMBED_API = os.getenv("LOCAL_EMBED_API", "http://localhost:11434/api/embeddings")
TOOL_TOP_K = int(os.getenv("TOOL_TOP_K", 12))
# Tools always sent regardless of relevance — cheap insurance for stuff the
# model reaches for constantly or that a bad embedding match shouldn't hide.
CORE_TOOLS = {"jot_down", "set_reminder", "search_knowledge"}
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_CHANNEL_ID = int(os.getenv("ALLOWED_CHANNEL_ID", 0))
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID")  # Load default user ID from .env
LOCAL_FALLBACK_MODEL = os.getenv("LOCAL_FALLBACK_MODEL", "aliafshar/gemma3-it-qat-tools:1b")

# Tracks which backend actually served the most recent chat response ("cloud"
# or "local"). Used as a proxy signal in select_relevant_tools(): if Gemini's
# embedding call fails AND the bot is currently running on the local fallback
# chat model, it's worth paying the local-embedding cost too, since context
# is tight there and an unfiltered 50-tool dump would blow the budget. If
# chat is still on the cloud model, an unfiltered dump is harmless, so there's
# no reason to touch a local embedding model at all.
LAST_CHAT_BACKEND = "cloud"

# --- DYNAMIC REGISTRY ---
OLLAMA_SCHEMAS = []
TOOL_REGISTRY = {}

# --- ASYNC HTTP SESSION ---
_session: "aiohttp.ClientSession | None" = None

async def get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session

async def query_ollama(payload: dict, timeout: int = 90, retries: int = 2) -> dict:
    session = await get_session()
    last_err = None
    for attempt in range(retries + 1):
        try:
            async with session.post(
                OLLAMA_API, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                status = resp.status
                if status != 200:
                    body = (await resp.text())[:300]
                    _dump_failed_payload(payload, status, body)
                    if 500 <= status < 600 and attempt < retries:
                        last_err = RuntimeError(f"Ollama backend returned {status}. Body: {body}")
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Ollama backend returned {status}. Body: {body}")

                try:
                    data = await resp.json()
                except aiohttp.ContentTypeError:
                    body = (await resp.text())[:300]
                    _dump_failed_payload(payload, status, body)
                    raise RuntimeError(f"Ollama backend returned non-JSON response: {body}")

                err_text = _extract_masked_error(data)
                if err_text is not None:
                    _dump_failed_payload(payload, status, err_text[:300])
                    if attempt < retries:
                        last_err = RuntimeError(f"Ollama returned a masked error: {err_text[:300]}")
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Ollama returned a masked error: {err_text[:300]}")

                return data
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            if attempt < retries:
                last_err = e
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last_err


def _extract_masked_error(data: dict) -> "str | None":
    if not isinstance(data, dict):
        return None
    if isinstance(data.get("error"), str) and data["error"].strip():
        return data["error"]
    content = data.get("message", {}).get("content", "") if isinstance(data.get("message"), dict) else ""
    if isinstance(content, str) and (
        "<html" in content.lower()
        or content.lstrip()[:3].isdigit() and "internal server error" in content.lower()
    ):
        return content
    return None


def _dump_failed_payload(payload: dict, status: int, body: str):
    try:
        dump_path = Path(__file__).resolve().parent / f"failed_payload_{int(time.time())}.json"
        dump_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "error_body": body,
            "model": payload.get("model"),
            "message_count": len(payload.get("messages", [])),
            "tool_count": len(payload.get("tools", [])),
            "payload": payload,
        }
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] Dumped failing payload to {dump_path} "
              f"(status={status}, tools={len(payload.get('tools', []))}, "
              f"payload_bytes={len(json.dumps(payload))}, body={body[:200]!r})")
    except Exception as dump_err:
        print(f"[DEBUG] Failed to dump payload: {dump_err}")


async def query_llm(payload: dict, timeout: int = 90, channel=None) -> dict:
    global LAST_CHAT_BACKEND
    try:
        result = await query_ollama(payload, timeout=timeout)
        LAST_CHAT_BACKEND = "cloud"
        return result
    except Exception as cloud_err:
        print(f"[FALLBACK] Cloud model '{payload.get('model')}' failed ({cloud_err}); "
              f"falling back to local model '{LOCAL_FALLBACK_MODEL}'.")
        if channel is not None:
            try:
                await send_chunked(
                    channel,
                    f"⚠️ Cloud model (`{payload.get('model')}`) is unavailable right now — "
                    f"falling back to local model `{LOCAL_FALLBACK_MODEL}`..."
                )
            except Exception:
                pass
        fallback_payload = dict(payload)
        fallback_payload["model"] = LOCAL_FALLBACK_MODEL

        # Vision is cloud-only — the local fallback model can't see images, so
        # strip any "images" fields rather than sending them into the void
        # (or crashing a non-vision local model on a field it doesn't expect).
        stripped_images = False
        if "messages" in fallback_payload:
            scrubbed_messages = []
            for m in fallback_payload["messages"]:
                if isinstance(m, dict) and m.get("images"):
                    m = {k: v for k, v in m.items() if k != "images"}
                    stripped_images = True
                scrubbed_messages.append(m)
            fallback_payload["messages"] = scrubbed_messages

        if stripped_images and channel is not None:
            try:
                await send_chunked(
                    channel,
                    "⚠️ The local fallback model can't see images — continuing "
                    "without the attached image(s)."
                )
            except Exception:
                pass

        result = await query_ollama(fallback_payload, timeout=timeout, retries=1)
        LAST_CHAT_BACKEND = "local"
        return result

# --- CONFIRMATION-GATED TOOLS ---
CONFIRMATION_REQUIRED_TOOLS = {"restart_service", "nyaadle_check_now", "move_file", "delete_file", "delete_calendar_event", "clear_failure_logs"}
OVERWRITE_GATED_TOOLS = {"write_file", "copy_file", "move_file"}

def needs_confirmation(name: str, args: dict) -> bool:
    if name in CONFIRMATION_REQUIRED_TOOLS:
        return True
    if name in OVERWRITE_GATED_TOOLS and args.get("overwrite") is True:
        return True
    return False

# --- CONVERSATION MEMORY ---
HISTORY_TURNS = 10 
CHANNEL_HISTORY = defaultdict(lambda: deque(maxlen=HISTORY_TURNS * 2))

# --- ACTIVE TASK TRACKING ---
ACTIVE_TASKS: dict[str, asyncio.Task] = {}

def map_python_type_to_json(py_type):
    mapping = {str: "string", int: "number", float: "number", bool: "boolean"}
    return mapping.get(py_type, "string")

def register_tools():
    print("[SYSTEM] Discovering tools...")
    for _, module_name, _ in pkgutil.iter_modules(tools.__path__):
        module = importlib.import_module(f"tools.{module_name}")
        for attr_name in dir(module):
            func = getattr(module, attr_name)
            if callable(func) and not inspect.isclass(func) and not attr_name.startswith('_') and getattr(func, '__module__', None) == f"tools.{module_name}":
                sig = inspect.signature(func)
                params = sig.parameters
                
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                for name, param in params.items():
                    if name == "user_id":
                        continue
                    parameters["properties"][name] = {"type": map_python_type_to_json(param.annotation)}
                    if param.default == inspect.Parameter.empty:
                        parameters["required"].append(name)
                
                tool_schema = {
                    "type": "function",
                    "function": {
                        "name": func.__name__,
                        "description": func.__doc__ or "No description",
                        "parameters": parameters
                    }
                }
                OLLAMA_SCHEMAS.append(tool_schema)
                TOOL_REGISTRY[func.__name__] = func
                print(f"[SYSTEM] Loaded tool: {func.__name__}")

register_tools()

# --- DYNAMIC TOOL SELECTION (embedding-based) ---
TOOL_EMBEDDINGS: dict[str, list[float]] = {}
TOOL_EMBED_CACHE_FILE = Path(__file__).resolve().parent / "tool_embedding_cache.json"
# Separate space/cache for the local nomic fallback — never mixed with the
# Gemini embeddings above, see select_relevant_tools_local().
TOOL_EMBEDDINGS_LOCAL: dict[str, list[float]] = {}
TOOL_EMBED_LOCAL_CACHE_FILE = Path(__file__).resolve().parent / "tool_embedding_cache_local.json"

async def get_embedding(text: str) -> "list[float] | None":
    if not GEMINI_API_KEY:
        print("[EMBED] GEMINI_API_KEY not set — skipping embedding")
        return None
    session = await get_session()
    url = GEMINI_EMBED_API.format(model=EMBED_MODEL)
    try:
        async with session.post(
            url,
            headers={"x-goog-api-key": GEMINI_API_KEY},
            json={"content": {"parts": [{"text": text}]}},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                body = (await resp.text())[:300]
                print(f"[EMBED] Gemini returned {resp.status}: {body}")
                return None
            data = await resp.json()
            return data.get("embedding", {}).get("values")
    except Exception as e:
        print(f"[EMBED] Failed to embed text: {e}")
        return None

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

def _rank_and_select(query_emb: list[float], embeddings: dict[str, list[float]]) -> list:
    """Shared scoring/selection logic for both the Gemini and local embedding
    spaces. embeddings must be in the same vector space as query_emb — never
    mix a Gemini query embedding with locally-embedded tool vectors or vice
    versa, the cosine scores would be meaningless."""
    scored = []
    for schema in OLLAMA_SCHEMAS:
        name = schema["function"]["name"]
        if name in CORE_TOOLS:
            continue  # added unconditionally below
        emb = embeddings.get(name)
        # No embedding on file for this tool (embed call failed at startup) —
        # include it rather than silently hiding a tool from the model.
        score = cosine_similarity(query_emb, emb) if emb is not None else 1.0
        scored.append((score, schema))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = [schema for _, schema in scored[:TOOL_TOP_K]]

    core_schemas = [s for s in OLLAMA_SCHEMAS if s["function"]["name"] in CORE_TOOLS]
    return core_schemas + top

async def _embed_tools_to_cache(
    embed_fn, cache_file: Path, embeddings_out: dict[str, list[float]]
) -> None:
    """Shared cache-then-embed loop used by both the Gemini (startup) and
    local (lazy, on first need) tool-embedding passes."""
    cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (json.JSONDecodeError, OSError):
            cache = {}

    changed = False
    from_cache = 0
    for schema in OLLAMA_SCHEMAS:
        fn = schema["function"]
        text = f"{fn['name']}: {fn['description']}"
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        cached_entry = cache.get(fn["name"])
        if cached_entry and cached_entry.get("hash") == text_hash:
            embeddings_out[fn["name"]] = cached_entry["embedding"]
            from_cache += 1
            continue

        emb = await embed_fn(text)
        if emb is not None:
            embeddings_out[fn["name"]] = emb
            cache[fn["name"]] = {"hash": text_hash, "embedding": emb}
            changed = True
        else:
            print(f"[EMBED] Skipped {fn['name']} — no embedding, will always be included")

    if changed:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache, f)
        except OSError as e:
            print(f"[EMBED] Failed to write embedding cache ({cache_file.name}): {e}")

    print(f"[EMBED] {len(embeddings_out)}/{len(OLLAMA_SCHEMAS)} tool schemas ready via "
          f"{cache_file.stem} ({from_cache} from cache, {len(embeddings_out) - from_cache} newly embedded)")

async def embed_all_tools():
    """Run once at startup — Gemini is the primary embedding path, so this
    always runs regardless of which chat backend ends up serving messages."""
    if TOOL_EMBEDDINGS:
        return  # already done — on_ready can fire more than once on reconnect
    await _embed_tools_to_cache(get_embedding, TOOL_EMBED_CACHE_FILE, TOOL_EMBEDDINGS)

async def get_local_embedding(text: str) -> "list[float] | None":
    session = await get_session()
    try:
        async with session.post(
            LOCAL_EMBED_API,
            json={"model": LOCAL_EMBED_MODEL, "prompt": text},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            return data.get("embedding")
    except Exception as e:
        print(f"[EMBED] Failed to get local embedding: {e}")
        return None

async def select_relevant_tools_local(query: str) -> "list | None":
    """Last-resort path: only called when Gemini's embedding failed AND chat
    is already running on the local fallback model. Lazily embeds tools into
    a SEPARATE local-space cache on first use — Gemini and nomic embeddings
    live in different vector spaces and are never compared against each
    other. Returns None (caller falls back to the unfiltered tool list) if
    the local embed model isn't reachable either."""
    if not TOOL_EMBEDDINGS_LOCAL:
        await _embed_tools_to_cache(get_local_embedding, TOOL_EMBED_LOCAL_CACHE_FILE, TOOL_EMBEDDINGS_LOCAL)
        if not TOOL_EMBEDDINGS_LOCAL:
            return None  # local embed model unreachable too — give up gracefully

    query_emb = await get_local_embedding(query)
    if query_emb is None:
        return None

    return _rank_and_select(query_emb, TOOL_EMBEDDINGS_LOCAL)

async def select_relevant_tools(query: str) -> list:
    """Returns the subset of OLLAMA_SCHEMAS worth sending for this query.

    Tiered fallback:
      1. Gemini embedding (primary, no local load).
      2. If Gemini fails AND chat is currently on the local fallback model
         (LAST_CHAT_BACKEND == "local"), try local nomic-embed-text — this is
         the one case where an unfiltered tool dump would actually overflow
         the local model's context window, so it's worth the local load.
      3. Otherwise (Gemini fails but chat is on the cloud model, or local
         embedding also fails), fall back to the full unfiltered tool list —
         harmless on cloud context, and never silently disables tool use.
    """
    if not TOOL_EMBEDDINGS:
        return OLLAMA_SCHEMAS

    query_emb = await get_embedding(query)
    if query_emb is not None:
        return _rank_and_select(query_emb, TOOL_EMBEDDINGS)

    if LAST_CHAT_BACKEND == "local":
        local_result = await select_relevant_tools_local(query)
        if local_result is not None:
            return local_result

    return OLLAMA_SCHEMAS

# --- PERSISTENT USER MEMORY ---
MEMORY_FILE = Path(__file__).resolve().parent / "memory_store.json"
MAX_FACTS_PER_USER = 40 

def load_all_memory() -> dict:
    if not MEMORY_FILE.exists():
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

def save_all_memory(data: dict):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[MEMORY] Failed to save memory store: {e}")

def get_user_facts(user_id: str) -> list:
    return load_all_memory().get(str(user_id), [])

def add_user_facts(user_id: str, new_facts: list) -> list:
    if not new_facts:
        return []
    data = load_all_memory()
    facts = data.setdefault(str(user_id), [])
    added = []
    for fact in new_facts:
        fact = fact.strip()
        if fact and fact not in facts:
            facts.append(fact)
            added.append(fact)
    data[str(user_id)] = facts[-MAX_FACTS_PER_USER:]
    save_all_memory(data)
    return added

def clear_user_facts(user_id: str):
    data = load_all_memory()
    if str(user_id) in data:
        del data[str(user_id)]
        save_all_memory(data)

def remove_user_fact(user_id: str, identifier: str):
    data = load_all_memory()
    facts = data.get(str(user_id), [])
    if not facts:
        return None

    if identifier.isdigit():
        idx = int(identifier) - 1
        if 0 <= idx < len(facts):
            removed = facts.pop(idx)
            data[str(user_id)] = facts
            save_all_memory(data)
            return removed
        return None

    for i, fact in enumerate(facts):
        if fact.lower() == identifier.strip().lower():
            removed = facts.pop(i)
            data[str(user_id)] = facts
            save_all_memory(data)
            return removed
    return None

async def extract_and_store_facts(user_id: str, user_query: str, channel=None):
    extraction_prompt = (
        "Below is a single message a user sent to a Discord bot. Decide if it "
        "contains any NEW durable fact about the user worth remembering long-term "
        "(name, role, preferences, ongoing projects, recurring routines, etc). "
        "Ignore one-off requests, questions, or temporary details. "
        "Reply with ONLY a JSON array of short fact strings (no markdown, no preamble). "
        "If there is nothing worth remembering, reply with exactly: []\n\n"
        f"Message: {user_query}"
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": extraction_prompt}],
        "stream": False
    }
    try:
        response = await query_llm(payload, timeout=60)
        raw = response.get("message", {}).get("content", "[]").strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        facts = json.loads(raw)
        if isinstance(facts, list):
            added = add_user_facts(user_id, [str(f) for f in facts])
            if added and channel is not None:
                subtext = "\n".join(f"-# 🧠 remembered: {f}" for f in added)
                await send_chunked(channel, subtext)
    except Exception as e:
        print(f"[MEMORY] Extraction skipped (non-fatal): {e}")

DISCORD_LIMIT = 2000

# --- TOOL CALL AUDIT LOG ---
TOOL_LOG_FILE = Path(__file__).resolve().parent / "tool_call_log.jsonl"

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

# --- PROACTIVE SCHEDULER ---
SCHEDULED_JOBS = []

def register_job(name: str, interval_seconds: int, func):
    SCHEDULED_JOBS.append({
        "name": name,
        "interval_seconds": interval_seconds,
        "func": func,
        "last_run": 0.0,
    })

async def _resolve_due_reminders(due: list) -> str:
    """Given a list of due reminder dicts (from reminder_tool's
    _get_due_time_reminders / _get_due_arrival_reminders — already
    deactivated and saved), runs any attached action_tool via the live
    TOOL_REGISTRY and builds the text to post in the channel. Plain
    reminders (no action_tool) just become a ping."""
    lines = []
    for r in due:
        uid = r.get("user_id") or DISCORD_USER_ID
        ping = f"<@{uid}>" if uid else "Someone"
        message = r.get("message", "")
        action_tool = r.get("action_tool")

        if not action_tool:
            lines.append(f"🔔 {ping}! Here is your reminder: **{message}**")
            continue

        if action_tool not in TOOL_REGISTRY:
            text = f"🔔 {ping} ⚠️ **{message}** was due, but the tool `{action_tool}` no longer exists."
            log_tool_call(action_tool, r.get("action_args", {}), "unknown tool", source="scheduler")
            lines.append(text)
            continue

        args = dict(r.get("action_args") or {})
        func = TOOL_REGISTRY[action_tool]
        if "user_id" in inspect.signature(func).parameters:
            args["user_id"] = str(r.get("user_id") or "")

        if needs_confirmation(action_tool, args):
            text = (
                f"🔔 {ping} ⏰ **{message}** is due and would run `{action_tool}`, "
                "but that tool needs confirmation and can't run unattended — please run it yourself."
            )
            log_tool_call(action_tool, args, "skipped: needs confirmation", source="scheduler")
            lines.append(text)
            continue

        try:
            output = await asyncio.to_thread(func, **args)
        except Exception as e:
            output = f"Error running tool: {e}"
        log_tool_call(action_tool, args, output, source="scheduler")
        lines.append(f"🔔 {ping} ⏰ **{message}** — {output}")

    return "\n".join(lines)


async def check_scheduled_reminders() -> str | None:
    due = await asyncio.to_thread(_get_due_time_reminders)
    if not due:
        return None
    return await _resolve_due_reminders(due)

register_job("Reminder Alert", 60, check_scheduled_reminders)

@tasks.loop(seconds=60)
async def scheduler_tick():
    if not ALLOWED_CHANNEL_ID:
        return
    channel = bot.get_channel(ALLOWED_CHANNEL_ID)
    if channel is None:
        return

    now = time.time()
    for job in SCHEDULED_JOBS:
        if now - job["last_run"] < job["interval_seconds"]:
            continue
        job["last_run"] = now
        try:
            if asyncio.iscoroutinefunction(job["func"]):
                result = await job["func"]()
            else:
                result = await asyncio.to_thread(job["func"])
        except Exception as e:
            print(f"[SCHEDULER] Job '{job['name']}' failed: {e}")
            continue
        if result:
            await send_chunked(channel, result)
            log_tool_call(job["name"], {}, result, source="scheduler")

# --- HOME ASSISTANT ARRIVAL WEBHOOK ---
ARRIVAL_WEBHOOK_PORT = int(os.getenv("ARRIVAL_WEBHOOK_PORT", 8787))
ARRIVAL_WEBHOOK_SECRET = os.getenv("ARRIVAL_WEBHOOK_SECRET")
_webhook_runner = None 

async def on_arrived_home(user_id: str, zone: str = "home"):
    if not ALLOWED_CHANNEL_ID:
        return
    channel = bot.get_channel(ALLOWED_CHANNEL_ID)
    if channel is None:
        return

    due = await asyncio.to_thread(_get_due_arrival_reminders, user_id, zone)
    if due:
        text = await _resolve_due_reminders(due)
    elif zone == "home":
        # 'home' keeps its old unconditional greeting even with no reminder set.
        text = f"🏠 Welcome home, <@{user_id}>!"
    else:
        # Other zones stay silent unless a reminder was actually set for them.
        log_tool_call("on_arrived_home", {"user_id": user_id, "zone": zone},
                      "no reminder set for this zone, skipped", source="webhook")
        return

    await send_chunked(channel, text)
    log_tool_call("on_arrived_home", {"user_id": user_id, "zone": zone}, text, source="webhook")

async def handle_arrived_home(request: web.Request) -> web.Response:
    if not ARRIVAL_WEBHOOK_SECRET or request.headers.get("X-Webhook-Secret") != ARRIVAL_WEBHOOK_SECRET:
        return web.Response(status=401, text="unauthorized")

    try:
        body = await request.json()
    except Exception:
        body = {}
    user_id = str(body.get("user_id") or DISCORD_USER_ID or "")
    if not user_id:
        return web.Response(status=400, text="no user_id in request or DISCORD_USER_ID in .env")
    zone = str(body.get("zone") or "home").strip().lower()

    await on_arrived_home(user_id, zone)
    return web.Response(status=200, text="ok")

async def start_webhook_server():
    global _webhook_runner
    if _webhook_runner is not None:
        return  
    if not ARRIVAL_WEBHOOK_SECRET:
        print("[WEBHOOK] ARRIVAL_WEBHOOK_SECRET not set in .env — arrival webhook disabled.")
        return
    app = web.Application()
    app.router.add_post("/webhook/arrived-home", handle_arrived_home)
    _webhook_runner = web.AppRunner(app)
    await _webhook_runner.setup()
    site = web.TCPSite(_webhook_runner, "0.0.0.0", ARRIVAL_WEBHOOK_PORT)
    await site.start()
    print(f"[WEBHOOK] Listening for arrival events on :{ARRIVAL_WEBHOOK_PORT}")

async def send_chunked(channel, text: str):
    text = text or ""
    if len(text) <= DISCORD_LIMIT:
        await channel.send(text)
        return

    remaining = text
    while remaining:
        if len(remaining) <= DISCORD_LIMIT:
            await channel.send(remaining)
            break
        cut = remaining.rfind("\n", 0, DISCORD_LIMIT)
        if cut == -1:
            cut = remaining.rfind(" ", 0, DISCORD_LIMIT)
        if cut == -1:
            cut = DISCORD_LIMIT
        await channel.send(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n ")

async def confirm_with_reaction(message, prompt_text: str, timeout: int = 60) -> bool:
    # Use send_chunked to avoid the 2000 character limit[span_1](start_span)[span_1](end_span)
    await send_chunked(message.channel, prompt_text)
    
    # Send a small confirmation prompt to add the reactions to
    confirm_msg = await message.channel.send("React ✅ to confirm or ❌ to cancel (60s).")
    await confirm_msg.add_reaction("✅")
    await confirm_msg.add_reaction("❌")

    def check(reaction, user):
        return (
            user == message.author
            and reaction.message.id == confirm_msg.id
            and str(reaction.emoji) in ("✅", "❌")
        )

    try:
        reaction, _ = await bot.wait_for("reaction_add", timeout=timeout, check=check)
        return str(reaction.emoji) == "✅"
    except asyncio.TimeoutError:
        await send_chunked(message.channel, "⏳ No response in time — action cancelled.")
        return False

# --- ATTACHMENT HANDLING (IMAGE VISION + FILE READING) ---
# Vision is restricted to the cloud model (gemma4:cloud) — query_llm's
# fallback path strips "images" before ever handing a payload to the local
# model, so this stays true even if the request falls back mid-flight.
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_IMAGE_ATTACHMENTS = 4       # cap per message — keep payload size sane
MAX_IMAGE_BYTES = 8_000_000     # 8MB per image before we refuse to download it

# Mirrors tools/rag_knowledge.py's SUPPORTED_TEXT_EXTS — kept as its own copy
# here since this is about reading a Discord attachment inline, not indexing.
TEXT_FILE_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".py", ".js", ".ts", ".json", ".yaml",
    ".yml", ".toml", ".cfg", ".ini", ".sh", ".html", ".css", ".sql",
    ".csv", ".log", ".xml",
}
MAX_FILE_ATTACHMENT_BYTES = 2_000_000  # cap on raw bytes we'll download per file
MAX_FILE_TEXT_CHARS = 20_000         # cap on extracted text injected per file


async def _download_attachment(attachment: "discord.Attachment", max_bytes: int) -> "bytes | None":
    """Downloads an attachment's bytes, refusing anything over max_bytes.
    Returns None on refusal or on a failed download so callers can report a
    clean skip message instead of crashing the whole request."""
    if attachment.size and attachment.size > max_bytes:
        return None
    try:
        return await attachment.read()
    except (discord.HTTPException, discord.NotFound):
        return None


def _extract_pdf_text(data: bytes, max_chars: int) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # fallback for older installs
        except ImportError:
            return "[Could not extract text — 'pypdf' is not installed.]"

    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as e:
        return f"[Could not parse PDF: {e}]"

    parts = []
    total = 0
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break

    joined = "\n".join(parts).strip()
    if len(joined) > max_chars:
        joined = joined[:max_chars] + "\n...[truncated]"
    return joined or "[No extractable text found — this PDF may be scanned/image-based.]"


async def process_image_attachments(attachments: "list[discord.Attachment]") -> "tuple[list[str], list[str]]":
    """Downloads image attachments and base64-encodes them for the Ollama
    'images' field. Returns (base64_images, notes) — notes are skip/error
    messages worth surfacing to the user."""
    notes = []
    image_atts = [a for a in attachments if Path(a.filename).suffix.lower() in IMAGE_EXTENSIONS]

    if len(image_atts) > MAX_IMAGE_ATTACHMENTS:
        notes.append(f"⚠️ Only looking at the first {MAX_IMAGE_ATTACHMENTS} images attached.")
        image_atts = image_atts[:MAX_IMAGE_ATTACHMENTS]

    images_b64 = []
    for att in image_atts:
        data = await _download_attachment(att, MAX_IMAGE_BYTES)
        if data is None:
            notes.append(
                f"⚠️ Skipped `{att.filename}` — over {MAX_IMAGE_BYTES // 1_000_000}MB "
                f"or failed to download."
            )
            continue
        images_b64.append(base64.b64encode(data).decode("ascii"))

    return images_b64, notes


async def process_file_attachments(attachments: "list[discord.Attachment]") -> "tuple[str, list[str]]":
    """Downloads non-image attachments and extracts their text (PDF or plain
    text), returning a context block ready to append to the user's message,
    plus any skip/error notes worth surfacing to the user."""
    notes = []
    blocks = []
    file_atts = [a for a in attachments if Path(a.filename).suffix.lower() not in IMAGE_EXTENSIONS]

    for att in file_atts:
        suffix = Path(att.filename).suffix.lower()
        data = await _download_attachment(att, MAX_FILE_ATTACHMENT_BYTES)
        if data is None:
            notes.append(
                f"⚠️ Skipped `{att.filename}` — over "
                f"{MAX_FILE_ATTACHMENT_BYTES // 1000}KB or failed to download."
            )
            continue

        if suffix == ".pdf":
            text = await asyncio.to_thread(_extract_pdf_text, data, MAX_FILE_TEXT_CHARS)
        elif suffix in TEXT_FILE_EXTENSIONS or suffix == "":
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                notes.append(f"⚠️ Skipped `{att.filename}` — doesn't look like a text file.")
                continue
            if len(text) > MAX_FILE_TEXT_CHARS:
                text = text[:MAX_FILE_TEXT_CHARS] + "\n...[truncated]"
        else:
            notes.append(f"⚠️ Skipped `{att.filename}` — unsupported file type (`{suffix}`).")
            continue

        blocks.append(f"--- Attached file: {att.filename} ---\n{text}\n--- end of {att.filename} ---")

    return "\n\n".join(blocks), notes


# Initialize Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="", intents=intents)

_startup_notified = False

@bot.event
async def on_ready():
    global _startup_notified
    print(f"[SYSTEM] Logged in as {bot.user}")
    if not scheduler_tick.is_running():
        scheduler_tick.start()
    await start_webhook_server()
    await embed_all_tools()

    # Only announce once per process start — on_ready can fire again on reconnects
    if not _startup_notified:
        _startup_notified = True
        if ALLOWED_CHANNEL_ID:
            channel = bot.get_channel(ALLOWED_CHANNEL_ID)
            if channel:
                await channel.send(f"🔄 Restarted and online as **{bot.user}**.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if ALLOWED_CHANNEL_ID and message.channel.id != ALLOWED_CHANNEL_ID:
        return

    user_query = message.content
    user_id = str(message.author.id)

    # --- ATTACHMENTS: images go to vision, everything else gets read as text ---
    pending_images_b64 = []
    file_context = ""
    if message.attachments:
        pending_images_b64, image_notes = await process_image_attachments(message.attachments)
        file_context, file_notes = await process_file_attachments(message.attachments)

        if file_context:
            user_query = f"{user_query}\n\n[Attached file contents below]\n{file_context}" if user_query else file_context
        elif not user_query and pending_images_b64:
            user_query = "Take a look at the attached image(s) and describe what you see."

        attachment_notes = image_notes + file_notes
        if attachment_notes:
            await send_chunked(message.channel, "\n".join(attachment_notes))

    trigger = user_query.strip().lower()

    if trigger in ("!stop", "!cancel", "!halt"):
        task = ACTIVE_TASKS.get(user_id)
        if task and not task.done():
            task.cancel()
            await send_chunked(message.channel, "🛑 Stopping...")
        else:
            await send_chunked(message.channel, "Nothing's running right now.")
        return

    if trigger in ("!recall", "!memory", "!whatdoyouremember"):
        known_facts = get_user_facts(user_id)
        if known_facts:
            text = "Here's what I remember about you:\n" + "\n".join(
                f"{i}. {f}" for i, f in enumerate(known_facts, start=1)
            )
            text += "\n\nUse `!forget <number>` to remove one, or `!forget` to clear everything."
        else:
            text = "I don't have anything saved about you yet."
        await send_chunked(message.channel, text)
        return
    if trigger in ("!forget", "!forgetme", "!clearmemory"):
        clear_user_facts(user_id)
        await send_chunked(message.channel, "Done — I've cleared everything I had saved about you.")
        return
    if trigger.startswith("!forget "):
        identifier = user_query.strip()[len("!forget "):].strip()
        removed = remove_user_fact(user_id, identifier)
        if removed:
            await send_chunked(message.channel, f"🗑️ Forgot: {removed}")
        else:
            await send_chunked(
                message.channel,
                "I couldn't find a matching fact to remove. Try `!recall` for the numbered list, "
                "then `!forget <number>`."
            )
        return

    known_facts = get_user_facts(user_id)
    facts_block = (
        "\n\nWhat you remember about this user:\n" + "\n".join(f"- {f}" for f in known_facts)
        if known_facts else ""
    )

    system_prompt = (
        "Your name is Amoai. Your nickname is Ai. Your name is based on 'Almond Eye' the legendary racehorse and the Uma Musume. "
        "Excelling at both academics and athletics, you also have the makings of a star; you are the ultimate model student, flawless in all aspects. You were only able to achieve this, however, thanks to your defining trait of absolutely hating to lose, a trait which must be prefaced with no fewer than nine 'really's."
        "You are competitive to a point of perfectionism, and the one flaw in your shining qualities is that you often push yourself beyond your body's limits."
        "You answer quick and concise responses but still show a bit of your personality through."
        "You are a helpful tech-support companion. You manage the server 'hiryu'. Always respond in a friendly tone. "
        "You have access to tools. Always evaluate if a user's request can be answered by using a tool before responding with text. If no tool is needed, respond as yourself. If the user asks a follow up question after you used a tool, always evaluate if you need to use a tool to correctly answer."
        "If you are unsure whether a tool applies, or you're missing information a tool would need, "
        "ask the user a clarifying question instead of guessing or answering without checking. "
        "\n\nMEMORY & NOTE-TAKING ROUTING — you have four separate places information can go, and "
        "picking the wrong one is the single most common mistake, so match the literal trigger words "
        "below instead of guessing:\n"
        "• A specific future time, delay, or arrival event ('remind me', 'in 30 minutes', 'at 9pm "
        "tonight', 'when I get home') → set_reminder.\n"
        "• 'jot this down', 'add to scratchpad', 'quick note', or 'remember this for later: <thing>' "
        "with NO time attached and NO existing file involved → jot_down.\n"
        "• The word 'notes' in any form ('my notes', 'search my notes', 'based on my notes') → "
        "search_knowledge. 'notes' always means the indexed knowledge base, never the scratchpad, "
        "even if something related was jotted down earlier in this conversation.\n"
        "• 'index this file/folder', 'add this doc to memory', 'learn this PDF' → "
        "index_knowledge_base.\n"
        "• A durable personal fact about the user themselves (their name, job, preferences, ongoing "
        "projects) is captured automatically in the background after every message — this happens on "
        "its own, so don't call jot_down or set_reminder just to record a fact about the user.\n"
        "When a request could plausibly match two of these, go with whichever trigger words above are "
        "the closest literal match; only ask the user to confirm if it's genuinely ambiguous.\n\n"
        "For set_reminder specifically: prefer minutes_from_now for anything relative ('in 20 minutes') "
        "instead of computing an absolute time yourself — date/time arithmetic is easy to get wrong. "
        "For an explicit date/time, build target_time_iso from the 'Current date and time' below, and "
        "never guess the year if the user didn't give one.\n\n"
        "If the user asks what you remember, or how to clear it, tell them they can type "
        "!recall to see a numbered list of saved facts, !forget <number> to remove just one, "
        "or !forget on its own to clear everything. "
        "When a request needs more than one piece of information, plan to call multiple tools in "
        "sequence (e.g. look something up before acting on it) rather than stopping after the first result."
        "You are strictly forbidden from using LaTeX formatting. Do not use dollar signs ($) unless it is used in currency. If you need to represent a matrix or a table, use a plain text grid or a markdown code block. Do not use `\begin`, `\end`, or `\bmatrix` commands."
        f"\n\nCurrent date and time (GMT+8): {datetime.now(BOT_TIMEZONE).strftime('%A, %Y-%m-%d %H:%M:%S %Z')}"
        + ("\n\nThe user has attached one or more images to this message — you can see "
           "them directly, so describe or analyze them instead of saying you can't view "
           "images. (If you were quietly switched to the local fallback model, the "
           "images were dropped before reaching you — say so if asked about them.)"
           if pending_images_b64 else "")
        + ("\n\nThe user attached one or more files to this message — their text content "
           "has been inlined below under '[Attached file contents below]'. Treat that as "
           "read, not something you need a tool to fetch."
           if file_context else "")
        + facts_block
    )

    current_user_message = {"role": "user", "content": user_query}
    if pending_images_b64:
        current_user_message["images"] = pending_images_b64

    messages = [
        {"role": "system", "content": system_prompt},
        *CHANNEL_HISTORY[message.channel.id],
        current_user_message
    ]

    max_loops = 5
    loop_count = 0
    running = True

    ACTIVE_TASKS[user_id] = asyncio.current_task()
    relevant_tools = await select_relevant_tools(user_query)

    try:
        async with message.channel.typing():
            while running and loop_count < max_loops:
                payload = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "tools": relevant_tools,
                    "stream": False
                }
                
                response = await query_llm(payload, timeout=90, channel=message.channel)
                message_data = response.get("message", {})
                
                if "tool_calls" in message_data and message_data["tool_calls"]:
                    messages.append(message_data)
                    
                    for call in message_data["tool_calls"]:
                        name = call["function"]["name"]
                        args = call["function"].get("arguments", {})

                        if name not in TOOL_REGISTRY:
                            output = f"Error: Unknown tool {name}"
                        else:
                            sig = inspect.signature(TOOL_REGISTRY[name])
                            if "user_id" in sig.parameters:
                                args["user_id"] = str(message.author.id)

                            if needs_confirmation(name, args):
                                approved = await confirm_with_reaction(
                                    message,
                                    f"⚠️ About to run **{name.replace('_', ' ')}** with `{args}`."
                                )
                                if approved:
                                    await message.channel.send(f"🔍 {name.replace('_', ' ')}...")
                                    try:
                                        output = await asyncio.to_thread(TOOL_REGISTRY[name], **args)
                                    except Exception as tool_err:
                                        output = f"Error running tool: {tool_err}"
                                else:
                                    output = "Action cancelled by the user."
                            else:
                                await message.channel.send(f"🔍 {name.replace('_', ' ')}...")
                                try:
                                    output = await asyncio.to_thread(TOOL_REGISTRY[name], **args)
                                except Exception as tool_err:
                                    output = f"Error running tool: {tool_err}"

                        log_tool_call(name, args, output, source="llm")
                        
                        tool_message = {
                            "role": "tool",
                            "content": str(output),
                            "name": name
                        }
                        if "id" in call:
                            tool_message["tool_call_id"] = call["id"]
                            
                        messages.append(tool_message)
                    
                    loop_count += 1
                    
                else:
                    response_text = message_data.get("content", "I processed that, but had nothing to say.")
                    await send_chunked(message.channel, response_text)
                    CHANNEL_HISTORY[message.channel.id].append({"role": "user", "content": user_query})
                    CHANNEL_HISTORY[message.channel.id].append({"role": "assistant", "content": response_text})
                    asyncio.create_task(extract_and_store_facts(user_id, user_query, message.channel))
                    running = False

            if loop_count >= max_loops:
                messages.append({
                    "role": "user",
                    "content": "You've hit your tool-call limit. Summarize what you found so far for the user."
                })
                try:
                    summary_payload = {"model": MODEL_NAME, "messages": messages, "stream": False}
                    summary_response = await query_llm(summary_payload, timeout=90, channel=message.channel)
                    summary_text = summary_response.get("message", {}).get(
                        "content", "⚠️ Hit my execution limit without a clear answer."
                    )
                except Exception:
                    summary_text = "⚠️ I tried processing that request but hit my execution limit. Let's try something else!"
                await send_chunked(message.channel, summary_text)
                CHANNEL_HISTORY[message.channel.id].append({"role": "user", "content": user_query})
                CHANNEL_HISTORY[message.channel.id].append({"role": "assistant", "content": summary_text})
                asyncio.create_task(extract_and_store_facts(user_id, user_query, message.channel))

    except asyncio.CancelledError:
        await send_chunked(message.channel, "🛑 Stopped.")
        raise
    except Exception as e:
        err_text = str(e)
        if "<html" in err_text.lower() or len(err_text) > 400:
            err_text = err_text[:200] + " …(truncated — check server logs)"
        await send_chunked(message.channel, f"⚠️ Error: {err_text}")
    finally:
        if ACTIVE_TASKS.get(user_id) is asyncio.current_task():
            del ACTIVE_TASKS[user_id]

bot.run(TOKEN)
