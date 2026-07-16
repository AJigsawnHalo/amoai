import os
import sys
import asyncio
import json
import time
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
from tools.reminder_tool import _fire_arrival_reminders

# --- CONFIGURATION ---
load_dotenv(find_dotenv())
MODEL_NAME = "gemma4:cloud"
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_CHANNEL_ID = int(os.getenv("ALLOWED_CHANNEL_ID", 0))
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID")  # Load default user ID from .env

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

async def query_ollama(payload: dict, timeout: int = 90) -> dict:
    session = await get_session()
    async with session.post(
        OLLAMA_API, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
    ) as resp:
        return await resp.json()

# --- CONFIRMATION-GATED TOOLS ---
CONFIRMATION_REQUIRED_TOOLS = {"restart_service", "nyaadle_check_now", "move_file", "delete_file"}
OVERWRITE_GATED_TOOLS = {"write_file", "copy_file", "move_file"}

def needs_confirmation(name: str, args: dict) -> bool:
    if name in CONFIRMATION_REQUIRED_TOOLS:
        return True
    if name in OVERWRITE_GATED_TOOLS and args.get("overwrite") is True:
        return True
    return False

# --- CONVERSATION MEMORY ---
HISTORY_TURNS = 10  # user+assistant pairs kept per channel
CHANNEL_HISTORY = defaultdict(lambda: deque(maxlen=HISTORY_TURNS * 2))

def map_python_type_to_json(py_type):
    """Maps Python types to JSON schema types for the LLM."""
    mapping = {str: "string", int: "number", float: "number", bool: "boolean"}
    return mapping.get(py_type, "string")

def register_tools():
    """Scans 'tools/' folder, maps functions, and builds schema on boot."""
    print("[SYSTEM] Discovering tools...")
    for _, module_name, _ in pkgutil.iter_modules(tools.__path__):
        module = importlib.import_module(f"tools.{module_name}")
        for attr_name in dir(module):
            func = getattr(module, attr_name)
            if callable(func) and not inspect.isclass(func) and not attr_name.startswith('_') and getattr(func, '__module__', None) == f"tools.{module_name}":
                sig = inspect.signature(func)
                params = sig.parameters
                
                # Build JSON Schema
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                for name, param in params.items():
                    # --- CHANGE: Hide user_id from LLM schema, we will auto-inject it backend ---
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

# Initialize and register tools
register_tools()

# --- PERSISTENT USER MEMORY ---
MEMORY_FILE = Path(__file__).resolve().parent / "memory_store.json"
MAX_FACTS_PER_USER = 40  # keep the store from growing unbounded

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
    """Adds new facts for a user, returning only the ones that were
    actually new (not already known)."""
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
    """Removes one fact, matched either by its 1-based position in the
    user's list (as shown by !recall) or by exact text. Returns the removed
    fact string, or None if nothing matched."""
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
        response = await query_ollama(payload, timeout=60)
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

# --- BACKGROUND JOB: CHECK REMINDERS ---
def check_reminders() -> str | None:
    """Background check: Loads reminders and returns a ping message for due items."""
    reminders_file = Path(__file__).resolve().parent / "reminders.json"
    if not reminders_file.exists():
        return None
        
    try:
        with open(reminders_file, "r", encoding="utf-8") as f:
            reminders = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(reminders, list):
        print(f"[SCHEDULER] reminders.json contained {type(reminders).__name__}, not a list — skipping this tick.")
        return None
        
    now = datetime.now(timezone.utc)
    due_reminders = []
    updated_reminders = []
    
    for r in reminders:
        # Arrival-based reminders have no trigger_time to compare — they're
        # fired by the webhook handler instead, not this timer-based job.
        if r.get("trigger_type") == "arrival":
            updated_reminders.append(r)
            continue
        if r.get("active", False):
            try:
                # Clean up timezone suffix 'Z' to offset format
                clean_iso = r["trigger_time"].replace("Z", "+00:00")
                trigger_dt = datetime.fromisoformat(clean_iso)
                if trigger_dt <= now:
                    due_reminders.append(r)
                    r["active"] = False
            except Exception as e:
                print(f"[SCHEDULER] Error parsing reminder time: {e}")
        updated_reminders.append(r)
        
    if not due_reminders:
        return None
        
    # Write back the updated (now deactivated) reminders
    try:
        with open(reminders_file, "w", encoding="utf-8") as f:
            json.dump(updated_reminders, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[SCHEDULER] Failed to save updated reminders: {e}")
        
    # Format pings
    messages = []
    for r in due_reminders:
        # Fall back to env-configured DISCORD_USER_ID if the reminder lacks one
        uid = r.get("user_id") or DISCORD_USER_ID
        ping = f"<@{uid}>" if uid else "Someone"
        messages.append(f"🔔 {ping}! Here is your reminder: **{r['message']}**")
        
    return "\n".join(messages)

# Register background jobs
register_job("Reminder Alert", 60, check_reminders)

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
# A small HTTP listener so Home Assistant can push a "you just got home"
# event the instant it happens, instead of the bot polling HA's REST API on
# a timer. Protected by a shared secret so nothing else on your LAN can
# trigger it. Set ARRIVAL_WEBHOOK_SECRET in .env or this stays disabled.
ARRIVAL_WEBHOOK_PORT = int(os.getenv("ARRIVAL_WEBHOOK_PORT", 8787))
ARRIVAL_WEBHOOK_SECRET = os.getenv("ARRIVAL_WEBHOOK_SECRET")
_webhook_runner = None  # kept as a module-level ref so the site isn't GC'd

async def on_arrived_home(user_id: str):
    """Fires when Home Assistant reports arrival for user_id. Fires any
    pending 'on arrival' reminders for that user; falls back to a generic
    welcome if there weren't any."""
    if not ALLOWED_CHANNEL_ID:
        return
    channel = bot.get_channel(ALLOWED_CHANNEL_ID)
    if channel is None:
        return

    fired = await asyncio.to_thread(_fire_arrival_reminders, user_id)
    text = fired if fired else f"🏠 Welcome home, <@{user_id}>!"
    await send_chunked(channel, text)
    log_tool_call("on_arrived_home", {"user_id": user_id}, text, source="webhook")

async def handle_arrived_home(request: web.Request) -> web.Response:
    if not ARRIVAL_WEBHOOK_SECRET or request.headers.get("X-Webhook-Secret") != ARRIVAL_WEBHOOK_SECRET:
        return web.Response(status=401, text="unauthorized")

    # Home Assistant can optionally POST {"user_id": "<discord id>"} so this
    # supports more than one person's arrival later. Falls back to the
    # single-user default in .env if the body doesn't specify one.
    try:
        body = await request.json()
    except Exception:
        body = {}
    user_id = str(body.get("user_id") or DISCORD_USER_ID or "")
    if not user_id:
        return web.Response(status=400, text="no user_id in request or DISCORD_USER_ID in .env")

    await on_arrived_home(user_id)
    return web.Response(status=200, text="ok")

async def start_webhook_server():
    global _webhook_runner
    if _webhook_runner is not None:
        return  # already running — on_ready can fire again on reconnect
    if not ARRIVAL_WEBHOOK_SECRET:
        print("[WEBHOOK] ARRIVAL_WEBHOOK_SECRET not set in .env — arrival webhook disabled.")
        return
    app = web.Application()
    app.router.add_post("/webhook/arrived-home", handle_arrived_home)
    _webhook_runner = web.AppRunner(app)
    await _webhook_runner.setup()
    # Bind to the LAN interface, not 0.0.0.0-to-the-internet. Adjust if hiryu
    # has a specific LAN IP you'd rather bind explicitly.
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
    confirm_msg = await message.channel.send(prompt_text)
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

# Initialize Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="", intents=intents)

@bot.event
async def on_ready():
    print(f"[SYSTEM] Logged in as {bot.user}")
    if not scheduler_tick.is_running():
        scheduler_tick.start()
    await start_webhook_server()

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if ALLOWED_CHANNEL_ID and message.channel.id != ALLOWED_CHANNEL_ID:
        return

    user_query = message.content
    user_id = str(message.author.id)

    # --- Memory commands ---
    trigger = user_query.strip().lower()
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
        "You have access to tools. Always evaluate if a user's request can be answered by using a tool before responding with text. If no tool is needed, respond as yourself. "
        "If you are unsure whether a tool applies, or you're missing information a tool would need, "
        "ask the user a clarifying question instead of guessing or answering without checking. "
        "If the user asks what you remember, or how to clear it, tell them they can type "
        "!recall to see a numbered list of saved facts, !forget <number> to remove just one, "
        "or !forget on its own to clear everything. "
        "When a request needs more than one piece of information, plan to call multiple tools in "
        "sequence (e.g. look something up before acting on it) rather than stopping after the first result."
        f"\n\nCurrent date and time: {datetime.now().astimezone().strftime('%A, %Y-%m-%d %H:%M:%S %Z')}"
        + facts_block
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *CHANNEL_HISTORY[message.channel.id],
        {"role": "user", "content": user_query}
    ]

    max_loops = 5
    loop_count = 0
    running = True
    
    try:
        async with message.channel.typing():
            while running and loop_count < max_loops:
                payload = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "tools": OLLAMA_SCHEMAS,
                    "stream": False
                }
                
                response = await query_ollama(payload, timeout=90)
                message_data = response.get("message", {})
                
                # Check for tool execution request
                if "tool_calls" in message_data and message_data["tool_calls"]:
                    messages.append(message_data)
                    
                    for call in message_data["tool_calls"]:
                        name = call["function"]["name"]
                        args = call["function"].get("arguments", {})

                        if name not in TOOL_REGISTRY:
                            output = f"Error: Unknown tool {name}"
                        else:
                            # --- CHANGE: Auto-inject User ID backend if the tool function expects it ---
                            sig = inspect.signature(TOOL_REGISTRY[name])
                            if "user_id" in sig.parameters:
                                args["user_id"] = str(message.author.id)

                            if needs_confirmation(name, args):
                                approved = await confirm_with_reaction(
                                    message,
                                    f"⚠️ About to run **{name.replace('_', ' ')}** with `{args}`. "
                                    f"React ✅ to confirm or ❌ to cancel (60s)."
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
                    summary_response = await query_ollama(summary_payload, timeout=90)
                    summary_text = summary_response.get("message", {}).get(
                        "content", "⚠️ Hit my execution limit without a clear answer."
                    )
                except Exception:
                    summary_text = "⚠️ I tried processing that request but hit my execution limit. Let's try something else!"
                await send_chunked(message.channel, summary_text)
                CHANNEL_HISTORY[message.channel.id].append({"role": "user", "content": user_query})
                CHANNEL_HISTORY[message.channel.id].append({"role": "assistant", "content": summary_text})
                asyncio.create_task(extract_and_store_facts(user_id, user_query, message.channel))

    except Exception as e:
        await send_chunked(message.channel, f"⚠️ Error: {e}")

    await bot.process_commands(message)

bot.run(TOKEN)

