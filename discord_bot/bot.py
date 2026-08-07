import os
import sys
import asyncio
import json
import hashlib
import re
import time
import base64
import io
import zipfile
import sqlite3
import importlib
import pkgutil
import inspect
import aiohttp
from aiohttp import web
import discord
from datetime import datetime, timezone
from pathlib import Path
from collections import deque
from discord.ext import commands, tasks
from dotenv import load_dotenv, find_dotenv
import tools
from tools.reminder_tool import _get_due_arrival_reminders, _get_due_time_reminders, BOT_TIMEZONE
from scheduled_briefing import build_briefing

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
CONFIRMATION_REQUIRED_TOOLS = {"restart_service", "restart_container", "nyaadle_check_now", "move_file", "delete_file", "delete_calendar_event", "clear_failure_logs"}
OVERWRITE_GATED_TOOLS = {"write_file", "copy_file", "move_file"}

def needs_confirmation(name: str, args: dict) -> bool:
    if name in CONFIRMATION_REQUIRED_TOOLS:
        return True
    if name in OVERWRITE_GATED_TOOLS and args.get("overwrite") is True:
        return True
    return False

# --- CONVERSATION MEMORY ---
# Per-channel rolling window, kept in memory for fast access during a
# session. Threads get a larger window than the main channel — a thread is
# a dedicated, bounded conversation (like a Claude chat), so it's worth
# keeping more of it around; the main channel is shared/ambient, so a
# tighter window keeps the prompt from dragging in unrelated topics.
HISTORY_TURNS = 10
THREAD_HISTORY_TURNS = 40
# Independent of the in-memory window above — this is how much of a
# channel's/thread's history survives in SQLite across bot restarts, so
# reopening an old thread days later doesn't come back empty.
CONVERSATION_LOG_MAX_PER_CHANNEL = 500

CHANNEL_HISTORY: dict[int, deque] = {}
_HYDRATED_CHANNELS: set[int] = set()

def _history_cap_messages(channel) -> int:
    turns = THREAD_HISTORY_TURNS if isinstance(channel, discord.Thread) else HISTORY_TURNS
    return turns * 2

def _load_conversation_log(channel_id, limit_messages: int) -> list:
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT role, content FROM conversation_log WHERE channel_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (str(channel_id), limit_messages),
        ).fetchall()
    finally:
        conn.close()
    return [{"role": role, "content": content} for role, content in reversed(rows)]

def _append_conversation_log(channel_id, role: str, content: str):
    conn = _get_memory_conn()
    try:
        conn.execute(
            "INSERT INTO conversation_log (channel_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (str(channel_id), role, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM conversation_log WHERE channel_id = ?", (str(channel_id),)
        ).fetchone()[0]
        if count > CONVERSATION_LOG_MAX_PER_CHANNEL:
            overflow = count - CONVERSATION_LOG_MAX_PER_CHANNEL
            conn.execute(
                """
                DELETE FROM conversation_log WHERE id IN (
                    SELECT id FROM conversation_log WHERE channel_id = ? ORDER BY id ASC LIMIT ?
                )
                """,
                (str(channel_id), overflow),
            )
            conn.commit()
    finally:
        conn.close()

def get_channel_history(channel) -> deque:
    """Returns the in-memory rolling-history deque for this channel/thread,
    hydrating it from the persistent conversation_log on first touch since
    process start so a bot restart doesn't blank out an in-progress thread."""
    cid = channel.id
    if cid not in CHANNEL_HISTORY:
        cap = _history_cap_messages(channel)
        history = deque(maxlen=cap)
        if cid not in _HYDRATED_CHANNELS:
            history.extend(_load_conversation_log(cid, cap))
            _HYDRATED_CHANNELS.add(cid)
        CHANNEL_HISTORY[cid] = history
    return CHANNEL_HISTORY[cid]

def record_turn(channel, user_query: str, response_text: str):
    """Appends a user/assistant exchange to both the in-memory window and
    the persistent log — call this everywhere a turn currently gets pushed
    onto CHANNEL_HISTORY."""
    history = get_channel_history(channel)
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response_text})
    _append_conversation_log(channel.id, "user", user_query)
    _append_conversation_log(channel.id, "assistant", response_text)

# --- ACTIVE TASK TRACKING ---
ACTIVE_TASKS: dict[str, asyncio.Task] = {}

def map_python_type_to_json(py_type):
    mapping = {str: "string", int: "number", float: "number", bool: "boolean"}
    return mapping.get(py_type, "string")

# Matches a ":param name: description text" line, capturing the name and
# the description. The description can wrap onto following indented lines
# (anything not starting a new ":param"/":return" tag), which is how the
# tool docstrings in tools/ are written for longer explanations.
_PARAM_LINE_RE = re.compile(r"^\s*:param\s+(\w+):\s?(.*)$")

def _parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
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
    params: dict[str, str] = {}
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

# --- PERSISTENT USER MEMORY (SQLite-backed) ---
# Was a flat memory_store.json capped at 40 facts/user. Moved to SQLite
# (same pattern as tools/rag_knowledge.py's vector store) so raising the cap
# is just a number, not a rewrite, and per-user lookups don't require
# loading every other user's facts into memory first.
MEMORY_DB_PATH = Path(__file__).resolve().parent / "data" / "memory_store.sqlite3"
MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
LEGACY_MEMORY_FILE = Path(__file__).resolve().parent / "memory_store.json"
MAX_FACTS_PER_USER = 200  # was 40 on the old JSON store

# Two facts whose embeddings score at or above this are close enough to be
# treated as verbatim the same and skipped outright — no need to even ask
# about it. In practice most real near-duplicates ("likes coffee" vs "really
# likes coffee in the morning") land a bit below this, which is why there's
# a second, lower band below for those.
FACT_DEDUP_THRESHOLD = 0.93
# Two facts whose embeddings score in [FACT_SUPERSESSION_BAND_MIN,
# FACT_DEDUP_THRESHOLD) are similar enough that they're probably about the
# same underlying thing but not similar enough to safely auto-skip — that's
# exactly the "slightly different phrasing" case that was slipping through
# before. Rather than trust the original broad extraction call to have
# already caught this (it often doesn't, especially on smaller models), a
# candidate in this band gets one focused, single-pair LLM comparison before
# being stored — see _find_supersession_candidate / _llm_decides_replace.
# Start conservative; if it's still merging facts that were actually meant
# to coexist, raise this number, and if near-duplicates keep slipping
# through as separate entries, lower it.
FACT_SUPERSESSION_BAND_MIN = 0.78
# Fact lists at or under this size are injected into the system prompt
# whole — not worth the embedding/ranking overhead. Above it, only the
# FACT_RELEVANCE_TOP_K most relevant facts to the current message go in, so
# the prompt doesn't grow unbounded as a user's fact count approaches
# MAX_FACTS_PER_USER.
FACT_INJECT_ALWAYS_UNDER = 8
FACT_RELEVANCE_TOP_K = 8

# Cheap pre-filter so extract_and_store_facts doesn't burn an LLM call on
# every single message (most messages are acknowledgments or one-off
# requests with nothing durable in them). False negatives here just mean a
# fact-bearing message got skipped — it almost always resurfaces in a later,
# more substantive message, so skipping is low-risk; the alternative is
# paying for an extraction call on every "ok thanks".
_LOW_SIGNAL_PATTERN = re.compile(
    r"^(ok(ay)?|k+|thanks?( you)?|thx|ty|cool|nice|lol+|lmao+|yes|yep|yeah|no|nope|"
    r"sure|got ?it|alright|sounds good|np|welcome)[.!?]*$",
    re.IGNORECASE,
)

def _looks_low_signal(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 12:
        return True
    return bool(_LOW_SIGNAL_PATTERN.match(stripped))

def _get_memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, fact)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_convlog_channel ON conversation_log(channel_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_suggestion_state (
            channel_id TEXT PRIMARY KEY,
            last_suggested_at INTEGER NOT NULL
        )
        """
    )
    # Additive migration for DBs created before updated_at/embedding existed.
    # ADD COLUMN has no "IF NOT EXISTS" in SQLite, so this just no-ops with
    # an OperationalError ("duplicate column") on every run after the first.
    for ddl in (
        "ALTER TABLE facts ADD COLUMN updated_at TEXT",
        "ALTER TABLE facts ADD COLUMN embedding TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass
    return conn

def _encode_embedding(embedding: "tuple[str, list[float]] | None") -> "str | None":
    if not embedding:
        return None
    space, vector = embedding
    if not vector:
        return None
    return json.dumps({"space": space, "vector": vector})

def _parse_embedding(raw: "str | None") -> "tuple[str, list[float]] | None":
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        vector = obj.get("vector")
        return (obj.get("space"), vector) if vector else None
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None

async def _embed_fact(text: str) -> "tuple[str, list[float]] | None":
    """Tiered embedding for a fact or a query string — same fallback order
    as select_relevant_tools(): Gemini primary, local nomic only when chat is
    currently on the local backend. Returns (space, vector) rather than a
    bare vector so callers only ever compare vectors embedded in the same
    space — Gemini and local vectors are never comparable to each other
    (see the note on TOOL_EMBEDDINGS_LOCAL above)."""
    emb = await get_embedding(text)
    if emb is not None:
        return "gemini", emb
    if LAST_CHAT_BACKEND == "local":
        emb = await get_local_embedding(text)
        if emb is not None:
            return "local", emb
    return None

def _migrate_legacy_json_memory():
    """One-time migration from the old memory_store.json into SQLite. Runs
    at import time; no-ops once the JSON file is gone or already migrated
    (renamed to .json.migrated on success, so this only ever runs once)."""
    if not LEGACY_MEMORY_FILE.exists():
        return
    try:
        with open(LEGACY_MEMORY_FILE, "r", encoding="utf-8") as f:
            legacy_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[MEMORY] Couldn't read legacy {LEGACY_MEMORY_FILE.name}, leaving it in place: {e}")
        return

    conn = _get_memory_conn()
    migrated_any = False
    try:
        for user_id, facts in legacy_data.items():
            for fact in facts:
                fact = fact.strip()
                if not fact:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO facts (user_id, fact, created_at) VALUES (?, ?, ?)",
                    (str(user_id), fact, datetime.now(timezone.utc).isoformat()),
                )
                migrated_any = True
        conn.commit()
    finally:
        conn.close()

    if migrated_any:
        backup_path = LEGACY_MEMORY_FILE.with_suffix(".json.migrated")
        try:
            LEGACY_MEMORY_FILE.rename(backup_path)
            print(f"[MEMORY] Migrated {LEGACY_MEMORY_FILE.name} into SQLite -> {backup_path.name}")
        except OSError as e:
            print(f"[MEMORY] Migrated to SQLite but couldn't rename old file: {e}")

_migrate_legacy_json_memory()

def get_user_facts(user_id: str) -> list:
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT fact FROM facts WHERE user_id = ? ORDER BY id ASC",
            (str(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]

def get_user_facts_with_embeddings(user_id: str) -> list:
    """Returns [(fact, (space, vector) | None), ...] for this user — used for
    semantic dedup and relevance ranking. Facts stored before embeddings
    existed, or whose embedding call failed at the time, come back with
    None; callers treat those as always-relevant/always-distinct rather than
    hiding them, same philosophy as select_relevant_tools() for tools with
    no cached embedding."""
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT fact, embedding FROM facts WHERE user_id = ? ORDER BY id ASC",
            (str(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return [(fact, _parse_embedding(raw)) for fact, raw in rows]

def add_user_fact(user_id: str, fact: str, embedding: "tuple[str, list[float]] | None" = None) -> bool:
    """Inserts a single fact. Returns False if it already exists verbatim for
    this user (UNIQUE constraint) or the fact is blank — semantic near-dupes
    should be filtered by the caller with _is_semantic_duplicate before this
    is ever reached."""
    fact = fact.strip()
    if not fact:
        return False
    conn = _get_memory_conn()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT OR IGNORE INTO facts (user_id, fact, created_at, updated_at, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(user_id), fact, now, now, _encode_embedding(embedding)),
        )
        inserted = bool(cur.rowcount)
        conn.commit()

        if inserted:
            # Enforce the per-user cap by trimming the oldest rows beyond it,
            # same "keep the newest N" behavior the old JSON store had.
            count = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE user_id = ?", (str(user_id),)
            ).fetchone()[0]
            if count > MAX_FACTS_PER_USER:
                overflow = count - MAX_FACTS_PER_USER
                conn.execute(
                    """
                    DELETE FROM facts WHERE id IN (
                        SELECT id FROM facts WHERE user_id = ? ORDER BY id ASC LIMIT ?
                    )
                    """,
                    (str(user_id), overflow),
                )
                conn.commit()
        return inserted
    finally:
        conn.close()

def update_user_fact(
    user_id: str, old_fact: str, new_fact: str, embedding: "tuple[str, list[float]] | None" = None
) -> "tuple[str, str] | None":
    """Finds old_fact by case-insensitive exact match and rewrites its text
    and embedding in place, preserving its row rather than appending — this
    is what lets a revised fact ("moved to Manila") supersede the old one
    ("lives in Calumpit") instead of both persisting side by side forever.
    Returns (old_text, new_text) on success, or None if old_fact wasn't
    found (the caller should then treat it as a plain add)."""
    new_fact = new_fact.strip()
    if not new_fact:
        return None
    conn = _get_memory_conn()
    try:
        row = conn.execute(
            "SELECT id, fact FROM facts WHERE user_id = ? AND lower(fact) = lower(?)",
            (str(user_id), old_fact.strip()),
        ).fetchone()
        if not row:
            return None
        fact_id, old_text = row
        conn.execute(
            "UPDATE facts SET fact = ?, updated_at = ?, embedding = ? WHERE id = ?",
            (new_fact, datetime.now(timezone.utc).isoformat(), _encode_embedding(embedding), fact_id),
        )
        conn.commit()
        return old_text, new_fact
    finally:
        conn.close()

def _is_semantic_duplicate(
    embedding: "tuple[str, list[float]] | None", existing: list
) -> bool:
    """True if `embedding` is close enough to something already stored that
    a new row isn't worth adding. Only ever compares within the same
    embedding space (Gemini vs local vectors are not comparable — see
    _embed_fact); if either side has no embedding this returns False rather
    than silently dropping a fact it can't judge."""
    if not embedding:
        return False
    space, vector = embedding
    for _, existing_embedding in existing:
        if not existing_embedding:
            continue
        existing_space, existing_vector = existing_embedding
        if existing_space != space:
            continue
        if cosine_similarity(vector, existing_vector) >= FACT_DEDUP_THRESHOLD:
            return True
    return False

def _find_supersession_candidate(
    embedding: "tuple[str, list[float]] | None", existing: list
) -> "tuple[str, float] | None":
    """Finds the existing fact most similar to `embedding`, if its score
    falls in [FACT_SUPERSESSION_BAND_MIN, FACT_DEDUP_THRESHOLD) — similar
    enough to plausibly be the same underlying fact restated, but not so
    similar that _is_semantic_duplicate already caught it. Returns
    (existing_fact_text, score) or None. Same-space-only, same reasoning as
    _is_semantic_duplicate."""
    if not embedding:
        return None
    space, vector = embedding
    best = None
    for fact_text, existing_embedding in existing:
        if not existing_embedding:
            continue
        existing_space, existing_vector = existing_embedding
        if existing_space != space:
            continue
        score = cosine_similarity(vector, existing_vector)
        if FACT_SUPERSESSION_BAND_MIN <= score < FACT_DEDUP_THRESHOLD:
            if best is None or score > best[1]:
                best = (fact_text, score)
    return best

async def _llm_decides_replace(existing_fact: str, new_fact: str) -> bool:
    """Focused single-pair comparison — deliberately a much narrower ask than
    the original extraction prompt (which has to scan the whole fact list
    and produce structured JSON in one pass, and in practice often misses
    this). A yes/no call on exactly two sentences is a task small/local
    models handle far more reliably. Defaults to False (keep both as
    separate facts) if the call fails or comes back unparseable — the worse
    outcome from a false negative here is one extra stored fact, which is
    far less damaging than wrongly erasing one the user still meant."""
    prompt = (
        "Two statements about the same person, from different times:\n"
        f"Earlier: {existing_fact}\n"
        f"Just now: {new_fact}\n\n"
        "Should \"Just now\" REPLACE \"Earlier\" (same underlying fact — a "
        "preference changed, a detail got more specific, a status changed)? "
        "Or are they two separate facts that can both stay true at the same "
        "time (e.g. two different hobbies, two different routines)?\n"
        "Reply with exactly one word: REPLACE or SEPARATE."
    )
    try:
        response = await query_llm(
            {"model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=20,
        )
        verdict = response.get("message", {}).get("content", "").strip().upper()
        return verdict.startswith("REPLACE")
    except Exception as e:
        print(f"[MEMORY] Supersession check skipped (non-fatal): {e}")
        return False

def remove_user_fact(user_id: str, identifier: str):
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT id, fact FROM facts WHERE user_id = ? ORDER BY id ASC",
            (str(user_id),),
        ).fetchall()
        if not rows:
            return None

        target_id, target_fact = None, None
        identifier = identifier.strip()
        if identifier.isdigit():
            idx = int(identifier) - 1
            if 0 <= idx < len(rows):
                target_id, target_fact = rows[idx]
        else:
            lowered = identifier.lower()
            # Exact match first, so a short fact that also happens to be a
            # substring of a longer one doesn't get shadowed by it.
            for row_id, fact in rows:
                if fact.lower() == lowered:
                    target_id, target_fact = row_id, fact
                    break
            if target_id is None:
                for row_id, fact in rows:
                    if lowered in fact.lower():
                        target_id, target_fact = row_id, fact
                        break

        if target_id is None:
            return None

        conn.execute("DELETE FROM facts WHERE id = ?", (target_id,))
        conn.commit()
        return target_fact
    finally:
        conn.close()

def _parse_fact_indices(spec: str) -> "list[int] | None":
    """Parses a comma/space-separated list of 1-based !recall positions and
    ranges ('1,3,5', '1 3 5', '2-4', or any mix) into a sorted, de-duplicated
    list of ints. Returns None if `spec` doesn't look like an index list at
    all, so the caller can fall back to single-fact text matching — this
    keeps plain '!forget <text>' working exactly as before."""
    spec = spec.strip()
    if not spec:
        return None
    tokens = [t for t in re.split(r"[,\s]+", spec) if t]
    if not tokens:
        return None
    indices = set()
    for tok in tokens:
        if tok.isdigit():
            indices.add(int(tok))
        elif re.fullmatch(r"\d+-\d+", tok):
            a, b = (int(x) for x in tok.split("-"))
            if a > b:
                a, b = b, a
            indices.update(range(a, b + 1))
        else:
            return None  # contains something that isn't a number or range
    return sorted(i for i in indices if i > 0) or None

def remove_user_facts(user_id: str, indices: list) -> tuple:
    """Removes multiple facts at once by their 1-based !recall position.
    All positions are resolved against a single snapshot of the current
    list before anything is deleted, so removing e.g. both 1 and 3 in the
    same call is safe — an earlier deletion never shifts what a later index
    refers to. Returns (removed_texts, invalid_positions); invalid_positions
    covers anything out of range so the caller can report it back."""
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT id, fact FROM facts WHERE user_id = ? ORDER BY id ASC",
            (str(user_id),),
        ).fetchall()

        removed_texts, invalid, to_delete_ids = [], [], []
        for idx in indices:
            pos = idx - 1
            if 0 <= pos < len(rows):
                row_id, fact = rows[pos]
                to_delete_ids.append(row_id)
                removed_texts.append(fact)
            else:
                invalid.append(idx)

        if to_delete_ids:
            conn.executemany("DELETE FROM facts WHERE id = ?", [(i,) for i in to_delete_ids])
            conn.commit()
        return removed_texts, invalid
    finally:
        conn.close()

def clear_user_facts(user_id: str):
    conn = _get_memory_conn()
    try:
        conn.execute("DELETE FROM facts WHERE user_id = ?", (str(user_id),))
        conn.commit()
    finally:
        conn.close()

async def get_relevant_facts_block(user_id: str, query: str) -> str:
    """Builds the "What you remember about this user" block for the system
    prompt. Small fact lists are sent whole; larger ones are trimmed to the
    FACT_RELEVANCE_TOP_K facts most relevant to the current message, so the
    prompt doesn't grow unbounded as a user's fact count climbs toward
    MAX_FACTS_PER_USER."""
    facts_with_emb = get_user_facts_with_embeddings(user_id)
    if not facts_with_emb:
        return ""

    if len(facts_with_emb) <= FACT_INJECT_ALWAYS_UNDER:
        chosen = [f for f, _ in facts_with_emb]
    else:
        query_emb = await _embed_fact(query)
        if query_emb is None:
            # Can't rank without a query embedding — most-recent facts are a
            # safer bet than an arbitrary truncation.
            chosen = [f for f, _ in facts_with_emb[-FACT_RELEVANCE_TOP_K:]]
        else:
            q_space, q_vector = query_emb
            scored = []
            for fact, emb in facts_with_emb:
                if emb and emb[0] == q_space:
                    score = cosine_similarity(q_vector, emb[1])
                else:
                    score = 1.0  # no comparable embedding — never silently hide it
                scored.append((score, fact))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            chosen = [fact for _, fact in scored[:FACT_RELEVANCE_TOP_K]]

    return "\n\nWhat you remember about this user:\n" + "\n".join(f"- {f}" for f in chosen)

async def extract_and_store_facts(user_id: str, user_query: str, channel=None):
    if _looks_low_signal(user_query):
        return

    existing_facts = get_user_facts(user_id)
    existing_block = "\n".join(f"- {f}" for f in existing_facts) if existing_facts else "(none yet)"

    extraction_prompt = (
        "Below is a single message a user sent to a Discord bot, plus the facts "
        "already remembered about this user. Decide whether the message contains "
        "any NEW durable fact worth remembering long-term (name, role, "
        "preferences, ongoing projects, recurring routines, etc), or whether it "
        "REVISES/contradicts one of the existing facts (e.g. moved cities, "
        "changed jobs, switched a preference). Ignore one-off requests, "
        "questions, or temporary details.\n\n"
        "Reply with ONLY a JSON object of this exact shape (no markdown, no "
        "preamble):\n"
        '{"add": ["new fact 1", ...], "update": [{"replaces": "<verbatim existing '
        'fact text>", "with": "<revised fact text>"}]}\n'
        "Only use \"update\" when \"replaces\" is copied verbatim from the "
        "existing facts list below — never paraphrase it. If nothing applies, "
        'reply with exactly {"add": [], "update": []}\n\n'
        f"Existing facts:\n{existing_block}\n\nMessage: {user_query}"
    )
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": extraction_prompt}],
        "stream": False
    }
    try:
        response = await query_llm(payload, timeout=60)
        raw = response.get("message", {}).get("content", "{}").strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return
        to_add = [str(f).strip() for f in parsed.get("add", []) if str(f).strip()]
        to_update = [
            u for u in parsed.get("update", [])
            if isinstance(u, dict) and str(u.get("replaces", "")).strip() and str(u.get("with", "")).strip()
        ]
    except Exception as e:
        print(f"[MEMORY] Extraction skipped (non-fatal): {e}")
        return

    updated_summaries = []
    for u in to_update:
        new_text = str(u["with"]).strip()
        embedding = await _embed_fact(new_text)
        result = update_user_fact(user_id, str(u["replaces"]), new_text, embedding)
        if result:
            updated_summaries.append(f"{result[0]} → {result[1]}")
        else:
            # "replaces" didn't match anything on file — treat as a fresh
            # fact rather than silently dropping it.
            to_add.append(new_text)

    added = []
    if to_add:
        existing_for_dedup = get_user_facts_with_embeddings(user_id)
        for fact in to_add:
            if not fact:
                continue
            embedding = await _embed_fact(fact)
            if _is_semantic_duplicate(embedding, existing_for_dedup):
                continue

            candidate = _find_supersession_candidate(embedding, existing_for_dedup)
            if candidate is not None:
                old_text, _score = candidate
                if await _llm_decides_replace(old_text, fact):
                    result = update_user_fact(user_id, old_text, fact, embedding)
                    if result:
                        updated_summaries.append(f"{result[0]} → {result[1]}")
                        existing_for_dedup = [
                            (f, e) for f, e in existing_for_dedup if f != old_text
                        ]
                        existing_for_dedup.append((fact, embedding))
                    continue

            if add_user_fact(user_id, fact, embedding):
                added.append(fact)
                existing_for_dedup.append((fact, embedding))

    if channel is not None and (added or updated_summaries):
        lines = [f"-# 🧠 remembered: {f}" for f in added]
        lines += [f"-# 🧠 updated: {s}" for s in updated_summaries]
        await send_chunked(channel, "\n".join(lines))

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

# --- MORNING BRIEFING ---
# Fires once per day at/after MORNING_BRIEFING_HOUR (default 6am, in
# BOT_TIMEZONE). Gated by _last_briefing_date rather than a fixed
# interval_seconds, since register_job's interval is measured from
# whenever the bot last happened to start — that drifts over restarts and
# wouldn't reliably land at the same wall-clock hour every day.
MORNING_BRIEFING_HOUR = int(os.getenv("MORNING_BRIEFING_HOUR", 6))
_last_briefing_date = None

# Runtime on/off switch. Seeded from .env so it survives as the default
# across restarts, but toggleable live via !briefing on/off without editing
# .env or restarting the bot (see the !briefing command handler below).
MORNING_BRIEFING_ENABLED = os.getenv("MORNING_BRIEFING_ENABLED", "true").strip().lower() not in (
    "0", "false", "no", "off"
)

async def check_morning_briefing() -> str | None:
    global _last_briefing_date
    if not MORNING_BRIEFING_ENABLED:
        return None
    now = datetime.now(BOT_TIMEZONE)
    if now.hour < MORNING_BRIEFING_HOUR:
        return None
    if _last_briefing_date == now.date():
        return None
    _last_briefing_date = now.date()
    return await asyncio.to_thread(build_briefing)

register_job("Morning Briefing", 60, check_morning_briefing)

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

# Archive-specific caps — a small zip can decompress into something huge, so
# these guard against zip bombs independently of MAX_FILE_ATTACHMENT_BYTES
# (which only limits the *compressed* download size).
ARCHIVE_EXTENSIONS = {".zip"}
MAX_ARCHIVE_ENTRIES = 50            # refuse to walk archives with more files than this
MAX_ARCHIVE_TOTAL_BYTES = 5_000_000  # cap on total decompressed bytes we'll read
MAX_ARCHIVE_TEXT_CHARS = 40_000     # cap on combined extracted text for the whole archive


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


def _extract_zip_text(data: bytes, filename: str) -> str:
    """Extracts and concatenates text-file contents from a zip archive,
    reusing TEXT_FILE_EXTENSIONS to decide what's worth reading. Bails out
    early on anything that looks like a zip bomb (too many entries, or too
    much declared/decompressed content) instead of trying to read it."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return f"[Could not open `{filename}` — not a valid zip file.]"

    infos = [i for i in zf.infolist() if not i.is_dir()]

    if len(infos) > MAX_ARCHIVE_ENTRIES:
        return (f"[Refused to read `{filename}` — {len(infos)} files exceeds "
                f"the {MAX_ARCHIVE_ENTRIES}-entry limit.]")

    declared_total = sum(i.file_size for i in infos)
    if declared_total > MAX_ARCHIVE_TOTAL_BYTES:
        return (f"[Refused to read `{filename}` — decompressed contents "
                f"({declared_total} bytes) exceed the "
                f"{MAX_ARCHIVE_TOTAL_BYTES}-byte limit. Possible zip bomb.]")

    blocks = []
    skipped = []
    read_total = 0

    for info in infos:
        suffix = Path(info.filename).suffix.lower()
        if suffix not in TEXT_FILE_EXTENSIONS and suffix != "":
            skipped.append(info.filename)
            continue

        read_total += info.file_size
        if read_total > MAX_ARCHIVE_TOTAL_BYTES:
            skipped.append(f"{info.filename} (over total-size cap)")
            continue

        try:
            raw = zf.read(info)
        except (zipfile.BadZipFile, RuntimeError) as e:
            skipped.append(f"{info.filename} (read error: {e})")
            continue

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(f"{info.filename} (not text)")
            continue

        blocks.append(f"  [{info.filename}]\n{text}")

    combined = "\n\n".join(blocks).strip()
    if len(combined) > MAX_ARCHIVE_TEXT_CHARS:
        combined = combined[:MAX_ARCHIVE_TEXT_CHARS] + "\n...[truncated]"

    if not combined:
        combined = "[No readable text files found in archive.]"

    if skipped:
        combined += f"\n\n[Skipped {len(skipped)} entries: {', '.join(skipped[:10])}" \
                     f"{' ...' if len(skipped) > 10 else ''}]"

    return combined


MAX_REPLY_CONTEXT_CHARS = 4000  # guard against quoting a huge message wholesale


async def get_reply_context(message: "discord.Message") -> str:
    """If this message is a Discord reply, resolve the message being replied
    to and format it as a context block, the same way attachments/images get
    folded into user_query below. Returns "" if this isn't a reply, or the
    original message can't be resolved (e.g. it was deleted).

    discord.py usually populates message.reference.resolved from its cache,
    but that's not guaranteed (e.g. after a restart, or a reply to something
    outside the cache window) — falls back to fetch_message when it's missing
    or came back as a DeletedReferencedMessage stub.
    """
    ref = message.reference
    if ref is None:
        return ""

    resolved = ref.resolved
    if resolved is None or isinstance(resolved, discord.DeletedReferencedMessage):
        if ref.message_id is None:
            return ""
        try:
            resolved = await message.channel.fetch_message(ref.message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return ""

    author_label = "you (Amoai), earlier" if resolved.author.id == bot.user.id else f"{resolved.author.display_name}"
    content = (resolved.content or "").strip()
    if not content and resolved.attachments:
        content = f"[message had no text, just attachment(s): {', '.join(a.filename for a in resolved.attachments)}]"
    if not content and resolved.embeds:
        content = "[message had no text, just an embed]"
    if not content:
        return ""

    if len(content) > MAX_REPLY_CONTEXT_CHARS:
        content = content[:MAX_REPLY_CONTEXT_CHARS] + "\n[...truncated]"

    return f"[Replied-to message — from {author_label}]\n{content}\n"


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


async def describe_images_for_tools(images_b64: "list[str]", user_query: str, channel=None) -> "str | None":
    """Vision pre-pass: sends the image(s) to the cloud model ALONE (no tools),
    since gemma4:cloud (and most vision models) can 500 when 'images' and
    'tools' are present in the same request. The returned text description is
    folded into the user's message as plain text so the normal tool-calling
    loop downstream never has to carry an 'images' field. Returns None on
    failure so the caller can fall back to a plain notice instead of crashing."""
    vision_prompt = (
        user_query.strip()
        or "Describe this image in detail, including any visible text, numbers, or data (OCR anything readable)."
    )
    vision_payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a vision module. Describe the attached image(s) thoroughly: "
                    "objects, layout, and — most importantly — transcribe any visible text, "
                    "numbers, labels, or data exactly as written. Be precise and complete; "
                    "another AI with no eyes will rely entirely on your description."
                ),
            },
            {"role": "user", "content": vision_prompt, "images": images_b64},
        ],
        # NOTE: no "tools" key here at all — sending "tools": [] (empty list)
        # made gemma4:cloud 500 every time (confirmed via failed_payload
        # dumps: identical 500 across 3 retries, only difference being the
        # empty tools array). Omitting the key entirely is what actually
        # means "no tools" for this backend.
        "stream": False,
    }
    try:
        response = await query_llm(vision_payload, timeout=90, channel=channel)
    except Exception as e:
        print(f"[VISION] Image description pass failed: {e}")
        return None

    if LAST_CHAT_BACKEND != "cloud":
        # Fell back to the local model, which can't see images either —
        # query_llm already strips "images" before that call, so a
        # "successful" local response here would just be a hallucination.
        return None

    return response.get("message", {}).get("content", "") or None


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
        elif suffix in ARCHIVE_EXTENSIONS:
            text = await asyncio.to_thread(_extract_zip_text, data, att.filename)
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


# --- AUTOMATIC THREAD SUGGESTION ---
# Persisted in SQLite (thread_suggestion_state), not just kept in memory —
# history_length is hydrated from the persistent conversation_log on restart,
# so an in-memory-only cooldown tracker would reset to "never suggested" on
# every restart while history_length stays wherever it left off, causing an
# immediate re-suggestion on the first message after every restart.
THREAD_SUGGESTION_THRESHOLD = 6   # messages in history before suggesting
THREAD_SUGGESTION_COOLDOWN = 10   # messages before asking again after a decline/timeout
THREAD_SEED_MESSAGES = 6          # most recent messages copied into the new thread for continuity

def _get_last_thread_suggestion(channel_id) -> int:
    conn = _get_memory_conn()
    try:
        row = conn.execute(
            "SELECT last_suggested_at FROM thread_suggestion_state WHERE channel_id = ?",
            (str(channel_id),),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else 0

def _set_last_thread_suggestion(channel_id, history_length: int):
    conn = _get_memory_conn()
    try:
        conn.execute(
            "INSERT INTO thread_suggestion_state (channel_id, last_suggested_at) VALUES (?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET last_suggested_at = excluded.last_suggested_at",
            (str(channel_id), history_length),
        )
        conn.commit()
    finally:
        conn.close()

async def maybe_suggest_thread(message, history_length: int):
    """Checks if the conversation is getting long and offers to spin up a thread,
    using the built-in reaction confirmation system."""
    channel_id = message.channel.id
    
    if isinstance(message.channel, discord.Thread):
        return

    last_suggested_at = _get_last_thread_suggestion(channel_id)
    cooldown_met = (last_suggested_at == 0) or (history_length - last_suggested_at >= THREAD_SUGGESTION_COOLDOWN)

    if history_length >= THREAD_SUGGESTION_THRESHOLD and cooldown_met:
        _set_last_thread_suggestion(channel_id, history_length)
        
        approved = await confirm_with_reaction(
            message,
            "🧵 This conversation is getting a bit long. Would you like to move this topic to a new thread?"
        )
        
        if approved:
            try:
                thread_name = f"Topic Discussion - {datetime.now(BOT_TIMEZONE).strftime('%H:%M')}"
                new_thread = await message.create_thread(name=thread_name, auto_archive_duration=1440)

                # Seed the new thread's persistent log with the recent exchange so
                # the bot still has context once the conversation continues there —
                # without this, get_channel_history(new_thread) starts empty and the
                # whole point of moving a long conversation over is lost.
                recent = list(get_channel_history(message.channel))[-THREAD_SEED_MESSAGES:]
                for turn in recent:
                    _append_conversation_log(new_thread.id, turn["role"], turn["content"])
                _HYDRATED_CHANNELS.discard(new_thread.id)  # force a fresh hydrate on first use

                recap = "\n".join(
                    f"**{'You' if t['role'] == 'user' else 'Amoai'}:** {t['content'][:300]}"
                    for t in recent
                )
                await send_chunked(
                    new_thread,
                    f"🧵 Picking up here — here's where we left off:\n\n{recap}\n\n"
                    "Go ahead with your follow-up questions."
                )
            except discord.HTTPException as e:
                await send_chunked(message.channel, f"⚠️ Failed to create thread: {e}")

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

    # Wire job_manager to this process's event loop and a way to post
    # results back to Discord when a background job finishes on its own.
    from tools.job_manager import set_event_loop, set_notifier
    set_event_loop(asyncio.get_event_loop())
    if ALLOWED_CHANNEL_ID:
        async def _notify_job_channel(text: str):
            channel = bot.get_channel(ALLOWED_CHANNEL_ID)
            if channel:
                await send_chunked(channel, text)
        set_notifier(_notify_job_channel)

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
    if ALLOWED_CHANNEL_ID:
        in_allowed_channel = message.channel.id == ALLOWED_CHANNEL_ID
        in_allowed_thread = (
            isinstance(message.channel, discord.Thread)
            and message.channel.parent_id == ALLOWED_CHANNEL_ID
        )
        if not (in_allowed_channel or in_allowed_thread):
            return

    user_query = message.content
    user_id = str(message.author.id)

    # --- REPLY CONTEXT: if this message is a Discord reply, pull in the
    # message being replied to so the model has it even if it's long since
    # scrolled out of the rolling channel history. ---
    reply_context = await get_reply_context(message)
    has_reply_context = bool(reply_context)
    if reply_context:
        user_query = f"{reply_context}\n{user_query}" if user_query else reply_context

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

    # --- VISION PRE-PASS: describe images as text BEFORE the tool-calling
    # loop starts. gemma4:cloud (like most vision models) can 500 when
    # "images" and "tools" are both present in one request, so images never
    # travel alongside "tools" — they're converted to a text description
    # here instead, and the main loop below never sees an "images" field. ---
    had_images = bool(pending_images_b64)
    if pending_images_b64:
        await send_chunked(message.channel, "👀 Looking at the image(s)...")
        async with message.channel.typing():
            image_description = await describe_images_for_tools(pending_images_b64, user_query, message.channel)
        if image_description:
            user_query = (
                f"{user_query}\n\n[Image content — extracted by vision pass]\n{image_description}"
                if user_query else
                f"[Image content — extracted by vision pass]\n{image_description}"
            )
        else:
            await send_chunked(
                message.channel,
                "⚠️ Couldn't get a description of the attached image(s) — continuing without them."
            )
        pending_images_b64 = []  # already folded into user_query as text; never attach raw images downstream

    trigger = user_query.strip().lower()

    if trigger in ("!stop", "!cancel", "!halt"):
        task = ACTIVE_TASKS.get(user_id)
        if task and not task.done():
            task.cancel()
            await send_chunked(message.channel, "🛑 Stopping...")
        else:
            await send_chunked(message.channel, "Nothing's running right now.")
        return

    if trigger in ("!briefing on", "!briefing off", "!briefing status"):
        global MORNING_BRIEFING_ENABLED
        if trigger == "!briefing on":
            MORNING_BRIEFING_ENABLED = True
            await send_chunked(message.channel, "☀️ Morning briefing is now **on**.")
        elif trigger == "!briefing off":
            MORNING_BRIEFING_ENABLED = False
            await send_chunked(message.channel, "🔕 Morning briefing is now **off**.")
        else:
            state = "on" if MORNING_BRIEFING_ENABLED else "off"
            await send_chunked(message.channel, f"Morning briefing is currently **{state}**.")
        return

    if trigger == "!allowlist" or trigger.startswith("!allowlist "):
        # Owner-only: this gates a security control (which Docker containers
        # the LLM's restart_container tool may touch), so it's checked here
        # rather than relying on channel membership like the other ! commands.
        if user_id != DISCORD_USER_ID:
            await send_chunked(message.channel, "🔒 Only the bot owner can manage the restart allowlist.")
            return

        from tools.docker_manager import get_allowlist, _add_to_allowlist, _remove_from_allowlist

        if trigger in ("!allowlist", "!allowlist list"):
            current = sorted(get_allowlist())
            text = (
                "Current restart allowlist:\n" + "\n".join(f"- {c}" for c in current)
                if current else
                "The restart allowlist is currently empty — no containers can be restarted."
            )
            await send_chunked(message.channel, text)
        elif trigger.startswith("!allowlist add "):
            container_name = user_query.strip()[len("!allowlist add "):].strip()
            await send_chunked(message.channel, _add_to_allowlist(container_name))
        elif trigger.startswith("!allowlist remove "):
            container_name = user_query.strip()[len("!allowlist remove "):].strip()
            await send_chunked(message.channel, _remove_from_allowlist(container_name))
        else:
            await send_chunked(
                message.channel,
                "Usage: `!allowlist` (show current), `!allowlist add <container>`, "
                "`!allowlist remove <container>`."
            )
        return

    if trigger in ("!recall", "!memory", "!whatdoyouremember"):
        known_facts = get_user_facts(user_id)
        if known_facts:
            text = "Here's what I remember about you:\n" + "\n".join(
                f"{i}. {f}" for i, f in enumerate(known_facts, start=1)
            )
            text += "\n\nUse `!forget <number>` to remove one (e.g. `!forget 1,3,5` or `!forget 2-4` " \
                    "for several at once), or `!forget` on its own to clear everything."
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
        indices = _parse_fact_indices(identifier)
        if indices is not None:
            removed_texts, invalid = remove_user_facts(user_id, indices)
            parts = []
            if removed_texts:
                parts.append("🗑️ Forgot:\n" + "\n".join(f"- {f}" for f in removed_texts))
            if invalid:
                parts.append(f"⚠️ Nothing at position(s): {', '.join(str(i) for i in invalid)}")
            if not parts:
                parts.append(
                    "I couldn't find any matching facts to remove. Try `!recall` for the "
                    "numbered list, then `!forget <number>` (or `!forget 1,3,5` / `!forget 2-4` "
                    "for several at once)."
                )
            await send_chunked(message.channel, "\n\n".join(parts))
        else:
            removed = remove_user_fact(user_id, identifier)
            if removed:
                await send_chunked(message.channel, f"🗑️ Forgot: {removed}")
            else:
                await send_chunked(
                    message.channel,
                    "I couldn't find a matching fact to remove. Try `!recall` for the numbered list, "
                    "then `!forget <number>` (or `!forget 1,3,5` / `!forget 2-4` for several at once)."
                )
        return

    facts_block = await get_relevant_facts_block(user_id, user_query)

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
        "picking the wrong one is the single most common mistake. Several of these share the same "
        "trigger words (especially 'remember' and 'note'), so check the rules IN ORDER below and stop "
        "at the first one that matches — don't keyword-match in isolation:\n"
        "1. A first-person statement about the user's own identity, preferences, job, or routines "
        "('I use Arch btw', 'remember I'm vegetarian', 'FYI I work remote now') where nothing specific "
        "is being asked to be saved verbatim and no file is named → call NO tool at all. This is "
        "captured automatically in the background after your response, even when the message starts "
        "with the word 'remember'. This rule wins over rules 3 and 5 below whenever it applies, even "
        "though those also list 'remember' as a trigger word.\n"
        "2. A specific future time, delay, or arrival event ('remind me', 'in 30 minutes', 'at 9pm "
        "tonight', 'when I get home') → set_reminder.\n"
        "3. 'jot this down', 'add to scratchpad', 'quick note', or a bare 'remember this: <thing>' "
        "where <thing> is a specific piece of content to save verbatim (a password, a link, a to-do "
        "item) — NOT a fact about the user themselves (that's rule 1) and NOT time-based (that's rule "
        "2) → jot_down.\n"
        "4. The word 'notes' used as a NOUN ('my notes', 'search my notes', 'based on my notes') → "
        "search_knowledge — the indexed knowledge base, never the scratchpad, even if something "
        "related was jotted down earlier in this conversation. 'Note' used as a VERB ('note that X', "
        "'make a note of X') is NOT this — re-check rule 1 and rule 3 instead.\n"
        "5. The user names an actual file or folder to ingest ('index this file/folder', 'add "
        "~/notes/project.md to memory', 'learn this PDF') → index_knowledge_base. Never call this just "
        "because the message contains the word 'remember' with no file or folder actually named.\n"
        "If you've checked all five in order and it's still genuinely ambiguous, ask the user to "
        "confirm rather than guessing.\n\n"
        "For set_reminder specifically: prefer minutes_from_now for anything relative ('in 20 minutes') "
        "instead of computing an absolute time yourself — date/time arithmetic is easy to get wrong. "
        "For an explicit date/time, build target_time_iso from the 'Current date and time' below, and "
        "never guess the year if the user didn't give one.\n\n"
        "If the user asks what you remember, or how to clear it, tell them they can type "
        "!recall to see a numbered list of saved facts, !forget <number> to remove just one "
        "(!forget 1,3,5 or !forget 2-4 to remove several at once), "
        "or !forget on its own to clear everything. "
        "When a request needs more than one piece of information, plan to call multiple tools in "
        "sequence (e.g. look something up before acting on it) rather than stopping after the first result."
        "You are strictly forbidden from using LaTeX formatting. Do not use dollar signs ($) unless it is used in currency. If you need to represent a matrix or a table, use a plain text grid or a markdown code block. Do not use `\begin`, `\end`, or `\bmatrix` commands."
        f"\n\nCurrent date and time (GMT+8): {datetime.now(BOT_TIMEZONE).strftime('%A, %Y-%m-%d %H:%M:%S %Z')}"
        + ("\n\nThe user attached one or more images to this message. You don't see the "
           "raw image — a separate vision pass already described/OCR'd it, and that "
           "description is inlined below under '[Image content — extracted by vision "
           "pass]'. Treat that as what you saw; don't say you can't view images."
           if had_images else "")
        + ("\n\nThe user attached one or more files to this message — their text content "
           "has been inlined below under '[Attached file contents below]'. Treat that as "
           "read, not something you need a tool to fetch."
           if file_context else "")
        + ("\n\nThe user used Discord's reply feature to reply directly to an earlier message, "
           "which is inlined below under '[Replied-to message]'. Treat that message as the "
           "specific thing they're asking about/reacting to — it's the context they intended "
           "to give you, even if it's not otherwise related to the current topic."
           if has_reply_context else "")
        + facts_block
    )

    current_user_message = {"role": "user", "content": user_query}
    if pending_images_b64:
        current_user_message["images"] = pending_images_b64

    messages = [
        {"role": "system", "content": system_prompt},
        *get_channel_history(message.channel),
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
                    record_turn(message.channel, user_query, response_text)
                    asyncio.create_task(maybe_suggest_thread(message, len(get_channel_history(message.channel))))
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
                record_turn(message.channel, user_query, summary_text)
                asyncio.create_task(maybe_suggest_thread(message, len(get_channel_history(message.channel))))
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

