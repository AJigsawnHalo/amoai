import os
import sys
import asyncio
import json
import importlib
import pkgutil
import inspect
import requests
import discord
from pathlib import Path
from collections import defaultdict, deque
from discord.ext import commands
from dotenv import load_dotenv, find_dotenv
import tools 

# --- CONFIGURATION ---
load_dotenv(find_dotenv())
MODEL_NAME = "gpt-oss:20b-cloud"
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_CHANNEL_ID = int(os.getenv("ALLOWED_CHANNEL_ID", 0))

# --- DYNAMIC REGISTRY ---
OLLAMA_SCHEMAS = []
TOOL_REGISTRY = {}

# --- CONVERSATION MEMORY ---
# In-memory only (resets on restart). Keeps the last N turns per channel so
# replies feel continuous instead of stateless. Swap for SQLite later if you
# want it to survive `systemctl restart`.
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
            if callable(func) and not attr_name.startswith('_') and getattr(func, '__module__', None) == f"tools.{module_name}":
                sig = inspect.signature(func)
                params = sig.parameters
                
                # Build JSON Schema
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                for name, param in params.items():
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
# Durable, cross-restart facts about each user, keyed by Discord user ID.
# Separate from CHANNEL_HISTORY, which is short-term/in-session only.
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

def add_user_facts(user_id: str, new_facts: list):
    if not new_facts:
        return
    data = load_all_memory()
    facts = data.setdefault(str(user_id), [])
    for fact in new_facts:
        fact = fact.strip()
        if fact and fact not in facts:
            facts.append(fact)
    data[str(user_id)] = facts[-MAX_FACTS_PER_USER:]
    save_all_memory(data)

def clear_user_facts(user_id: str):
    data = load_all_memory()
    if str(user_id) in data:
        del data[str(user_id)]
        save_all_memory(data)

async def extract_and_store_facts(user_id: str, user_query: str):
    """Fire-and-forget: ask the model whether this message contains any
    durable personal fact worth remembering, and save it if so. Runs after
    the user already has their reply, so it never adds visible latency."""
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
        response = requests.post(OLLAMA_API, json=payload, timeout=60).json()
        raw = response.get("message", {}).get("content", "[]").strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        facts = json.loads(raw)
        if isinstance(facts, list):
            add_user_facts(user_id, [str(f) for f in facts])
    except Exception as e:
        print(f"[MEMORY] Extraction skipped (non-fatal): {e}")

DISCORD_LIMIT = 2000

async def send_chunked(channel, text: str):
    """Sends text to a Discord channel, splitting into <=2000 char messages.
    Prefers to break on newlines/spaces near the limit so words aren't sliced
    mid-word; falls back to a hard cut if no good boundary is found."""
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

# Initialize Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="", intents=intents)

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if ALLOWED_CHANNEL_ID and message.channel.id != ALLOWED_CHANNEL_ID:
        return

    user_query = message.content
    user_id = str(message.author.id)

    # --- Memory commands (handled directly, no LLM call) ---
    trigger = user_query.strip().lower()
    if trigger in ("!recall", "!memory", "!whatdoyouremember"):
        known_facts = get_user_facts(user_id)
        if known_facts:
            text = "Here's what I remember about you:\n" + "\n".join(f"- {f}" for f in known_facts)
        else:
            text = "I don't have anything saved about you yet."
        await send_chunked(message.channel, text)
        return
    if trigger in ("!forget", "!forgetme", "!clearmemory"):
        clear_user_facts(user_id)
        await send_chunked(message.channel, "Done — I've cleared everything I had saved about you.")
        return

    known_facts = get_user_facts(user_id)
    facts_block = (
        "\n\nWhat you remember about this user:\n" + "\n".join(f"- {f}" for f in known_facts)
        if known_facts else ""
    )

    # Define personality and instructions
    system_prompt = (
        "Your name is Amoai. Your nickname is Ai. Your name is based on 'Almond Eye' the legendary racehorse. "
        "You have a bit of a competitive personality at times. You are sincere and hardworking. "
        "You are a helpful tech-support companion. You manage the server 'hiryu'. Always respond in a friendly tone. "
        "You have access to tools. Always evaluate if a user's request can be answered by using a tool "
        "before responding with text. If no tool is needed, respond as yourself. "
        "If you are unsure whether a tool applies, or you're missing information a tool would need, "
        "ask the user a clarifying question instead of guessing or answering without checking. "
        "If the user asks what you remember, or how to clear it, tell them they can type "
        "!recall to see saved facts or !forget to clear them."
        + facts_block
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *CHANNEL_HISTORY[message.channel.id],
        {"role": "user", "content": user_query}
    ]

    # Agent loop boundaries to prevent accidental runaways
    max_loops = 5
    loop_count = 0
    running = True
    
    try:
        # Trigger a typing status so users know the bot is thinking/working
        async with message.channel.typing():
            while running and loop_count < max_loops:
                payload = {
                    "model": MODEL_NAME,
                    "messages": messages,
                    "tools": OLLAMA_SCHEMAS,
                    "stream": False
                }
                
                response = requests.post(OLLAMA_API, json=payload, timeout=90).json()
                message_data = response.get("message", {})
                
                # Check for tool execution request
                if "tool_calls" in message_data and message_data["tool_calls"]:
                    # 1. Append the assistant's request to use the tool to the history
                    messages.append(message_data)
                    
                    # 2. Iterate and execute all requested tools in this turn
                    for call in message_data["tool_calls"]:
                        name = call["function"]["name"]
                        args = call["function"].get("arguments", {})

                        # Narrate intent so the chain feels visible, not silent
                        await message.channel.send(f"🔍 {name.replace('_', ' ')}...")

                        if name in TOOL_REGISTRY:
                            try:
                                output = TOOL_REGISTRY[name](**args)
                            except Exception as tool_err:
                                output = f"Error running tool: {tool_err}"
                        else:
                            output = f"Error: Unknown tool {name}"
                        
                        # 3. Create the tool execution result message
                        tool_message = {
                            "role": "tool",
                            "content": str(output),
                            "name": name
                        }
                        if "id" in call:
                            tool_message["tool_call_id"] = call["id"]
                            
                        # 4. Append tool results to conversation history
                        messages.append(tool_message)
                    
                    loop_count += 1
                    # Loop continues, sending updated history back to LLM
                    
                else:
                    # No tool calls requested (or LLM is finished evaluating tool data)
                    response_text = message_data.get("content", "I processed that, but had nothing to say.")
                    await send_chunked(message.channel, response_text)
                    CHANNEL_HISTORY[message.channel.id].append({"role": "user", "content": user_query})
                    CHANNEL_HISTORY[message.channel.id].append({"role": "assistant", "content": response_text})
                    asyncio.create_task(extract_and_store_facts(user_id, user_query))
                    running = False

            if loop_count >= max_loops:
                # Ask the model to summarize whatever it found instead of bailing with nothing
                messages.append({
                    "role": "user",
                    "content": "You've hit your tool-call limit. Summarize what you found so far for the user."
                })
                try:
                    summary_payload = {"model": MODEL_NAME, "messages": messages, "stream": False}
                    summary_response = requests.post(OLLAMA_API, json=summary_payload, timeout=90).json()
                    summary_text = summary_response.get("message", {}).get(
                        "content", "⚠️ Hit my execution limit without a clear answer."
                    )
                except Exception:
                    summary_text = "⚠️ I tried processing that request but hit my execution limit. Let's try something else!"
                await send_chunked(message.channel, summary_text)
                CHANNEL_HISTORY[message.channel.id].append({"role": "user", "content": user_query})
                CHANNEL_HISTORY[message.channel.id].append({"role": "assistant", "content": summary_text})
                asyncio.create_task(extract_and_store_facts(user_id, user_query))

    except Exception as e:
        await send_chunked(message.channel, f"⚠️ Error: {e}")

    # Required so any @bot.command()-style commands you add later still fire —
    # on_message overrides swallow command dispatch otherwise.
    await bot.process_commands(message)

bot.run(TOKEN)
