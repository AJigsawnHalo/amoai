"""
config.py — central environment/config loading.

Other modules should `import config` and reference config.X, rather than
`from config import X`. A couple of these values are read by many modules
and none of them mutate after startup, so this distinction mostly matters
for consistency with the rest of core/ (see LAST_CHAT_BACKEND in llm.py,
TOOL_REGISTRY in tool_registry.py, etc., which DO mutate and where
`from module import X` would silently capture a stale copy).
"""
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

MODEL_NAME = os.getenv("MODEL_NAME") or "gemma4:cloud"
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME") or "gemma4:cloud"
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_EMBED_API = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{{model}}:embedContent"
)
# Last-resort embedding path — only reached when Gemini's embedding call
# fails AND chat is already running on the local fallback model (see
# llm.LAST_CHAT_BACKEND). Reuses the same local Ollama instance the bot
# already talks to for chat, and the same model rag_knowledge.py uses.
LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "nomic-embed-text")
LOCAL_EMBED_API = os.getenv("LOCAL_EMBED_API", "http://localhost:11434/api/embeddings")
TOOL_TOP_K = int(os.getenv("TOOL_TOP_K", 5))
# Tools always sent regardless of relevance — cheap insurance for stuff the
# model reaches for constantly or that a bad embedding match shouldn't hide.
CORE_TOOLS = {"jot_down", "set_reminder", "search_knowledge"}
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_CHANNEL_ID = int(os.getenv("ALLOWED_CHANNEL_ID", 0))
DISCORD_USER_ID = os.getenv("DISCORD_USER_ID")  # Load default user ID from .env
LOCAL_FALLBACK_MODEL = os.getenv("LOCAL_FALLBACK_MODEL", "aliafshar/gemma3-it-qat-tools:1b")
# Sampling/context options sent ONLY to local models (never to *:cloud models).
# Low temperature makes small models far more consistent at picking tools and
# filling arguments. num_ctx must hold the system prompt + facts + history +
# tool schemas; on a 4GB GPU pair 8192 with OLLAMA_KV_CACHE_TYPE=q8_0 and
# OLLAMA_FLASH_ATTENTION=1 on the Ollama server, and drop to 4096 if
# `ollama ps` shows the model spilling onto the CPU.
LOCAL_TEMPERATURE = float(os.getenv("LOCAL_TEMPERATURE", 0.2))
LOCAL_NUM_CTX = int(os.getenv("LOCAL_NUM_CTX", 8192))

# Base directories other core/ modules build their storage paths from, e.g.
# config.DATA_DIR / "reminders.json". Centralized here so the data/debug
# layout only has to be decided once, in one file, rather than each module
# recomputing "parent.parent" relative to its own new depth under core/.
_BASE_DIR = Path(__file__).resolve().parent.parent  # core/ -> discord_bot/
DATA_DIR = _BASE_DIR / "data"
DEBUG_DIR = _BASE_DIR / "debug"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR.mkdir(parents=True, exist_ok=True)
