import os
import sys
import importlib
import pkgutil
import inspect
import requests
import discord
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
                doc = inspect.getdoc(func) or "No description provided."
                
                props = {name: {"type": map_python_type_to_json(p.annotation), "description": f"The {name} argument."}
                         for name, p in sig.parameters.items() if name not in ['args', 'kwargs']}
                req = [n for n, p in sig.parameters.items() if p.default == inspect.Parameter.empty and n not in ['args', 'kwargs']]
                
                OLLAMA_SCHEMAS.append({
                    "type": "function", 
                    "function": {
                        "name": func.__name__, 
                        "description": doc, 
                        "parameters": {"type": "object", "properties": props, "required": req}
                    }
                })
                TOOL_REGISTRY[func.__name__] = func
                print(f"  -> Registered: '{func.__name__}'")

register_tools()

# --- DISCORD BOT ---
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

async def send_long_message(channel, text):
    """Splits a long message into chunks to avoid Discord's 2000-character limit."""
    if len(text) <= 2000:
        await channel.send(text)
    else:
        for i in range(0, len(text), 2000):
            await channel.send(text[i:i+2000])

@bot.event
async def on_message(message):
    if message.author == bot.user or (ALLOWED_CHANNEL_ID and message.channel.id != ALLOWED_CHANNEL_ID):
        return

    if message.content.startswith("! "):
        user_query = message.content[2:].strip()
        async with message.channel.typing():
            # System prompt forces 'Agent' behavior
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "system", "content": "You are a helpful server assistant. You have access to tools. Always evaluate if a user's request can be answered by using a tool before responding with text."},
                    {"role": "user", "content": user_query}
                ],
                "tools": OLLAMA_SCHEMAS,
                "stream": False
            }
            
            try:
                response = requests.post(OLLAMA_API, json=payload, timeout=90).json()
                message_data = response.get("message", {})
                
                # Check for tool execution request
                if "tool_calls" in message_data:
                    for call in message_data["tool_calls"]:
                        name, args = call["function"]["name"], call["function"].get("arguments", {})
                        if name in TOOL_REGISTRY:
                            output = TOOL_REGISTRY[name](**args)
                            await message.channel.send(f"⚙️ **Tool {name} executed.**\nOutput: {output}")
                        else:
                            await message.channel.send(f"❌ Unknown tool: {name}")
                else:
                    # Use the long message handler
                    response_text = message_data.get("content", "I processed that, but had nothing to say.")
                    await send_long_message(message.channel, response_text)
                    
            except Exception as e:
                await message.channel.send(f"⚠️ Error: {e}")

bot.run(TOKEN)
