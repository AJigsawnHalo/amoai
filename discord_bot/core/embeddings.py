"""
embeddings.py — embeds tool schemas (Gemini primary, local nomic fallback)
and selects the subset of tools worth sending for a given query.

TOOL_EMBEDDINGS / TOOL_EMBEDDINGS_LOCAL are mutable, populated lazily/at
startup. Same rule as elsewhere in core/: reference them via
`embeddings.TOOL_EMBEDDINGS`, not a `from` import.
"""
import hashlib
import json
from pathlib import Path

import aiohttp

import config
import llm
import tool_registry

TOOL_EMBEDDINGS: "dict[str, list[float]]" = {}
TOOL_EMBED_CACHE_FILE = config.DATA_DIR / "tool_embedding_cache.json"
# Separate space/cache for the local nomic fallback — never mixed with the
# Gemini embeddings above, see select_relevant_tools_local().
TOOL_EMBEDDINGS_LOCAL: "dict[str, list[float]]" = {}
TOOL_EMBED_LOCAL_CACHE_FILE = config.DATA_DIR / "tool_embedding_cache_local.json"


async def get_embedding(text: str) -> "list[float] | None":
    if not config.GEMINI_API_KEY:
        print("[EMBED] GEMINI_API_KEY not set — skipping embedding")
        return None
    session = await llm.get_session()
    url = config.GEMINI_EMBED_API.format(model=config.EMBED_MODEL)
    try:
        async with session.post(
            url,
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
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


def cosine_similarity(a: "list[float]", b: "list[float]") -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _rank_and_select(query_emb: "list[float]", embeddings: "dict[str, list[float]]") -> list:
    """Shared scoring/selection logic for both the Gemini and local embedding
    spaces. embeddings must be in the same vector space as query_emb — never
    mix a Gemini query embedding with locally-embedded tool vectors or vice
    versa, the cosine scores would be meaningless."""
    scored = []
    for schema in tool_registry.OLLAMA_SCHEMAS:
        name = schema["function"]["name"]
        if name in config.CORE_TOOLS:
            continue  # added unconditionally below
        emb = embeddings.get(name)
        # No embedding on file for this tool (embed call failed at startup) —
        # include it rather than silently hiding a tool from the model.
        score = cosine_similarity(query_emb, emb) if emb is not None else 1.0
        scored.append((score, schema))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = [schema for _, schema in scored[:config.TOOL_TOP_K]]

    core_schemas = [s for s in tool_registry.OLLAMA_SCHEMAS if s["function"]["name"] in config.CORE_TOOLS]
    return core_schemas + top


async def _embed_tools_to_cache(
    embed_fn, cache_file: Path, embeddings_out: "dict[str, list[float]]"
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
    for schema in tool_registry.OLLAMA_SCHEMAS:
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

    print(f"[EMBED] {len(embeddings_out)}/{len(tool_registry.OLLAMA_SCHEMAS)} tool schemas ready via "
          f"{cache_file.stem} ({from_cache} from cache, {len(embeddings_out) - from_cache} newly embedded)")


async def embed_all_tools():
    """Run once at startup — Gemini is the primary embedding path, so this
    always runs regardless of which chat backend ends up serving messages."""
    if TOOL_EMBEDDINGS:
        return  # already done — on_ready can fire more than once on reconnect
    await _embed_tools_to_cache(get_embedding, TOOL_EMBED_CACHE_FILE, TOOL_EMBEDDINGS)


async def get_local_embedding(text: str) -> "list[float] | None":
    session = await llm.get_session()
    try:
        async with session.post(
            config.LOCAL_EMBED_API,
            json={"model": config.LOCAL_EMBED_MODEL, "prompt": text},
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
    """Returns the subset of tool_registry.OLLAMA_SCHEMAS worth sending for
    this query.

    Tiered fallback:
      1. Gemini embedding (primary, no local load).
      2. If Gemini fails AND chat is currently on the local fallback model
         (llm.LAST_CHAT_BACKEND == "local"), try local nomic-embed-text —
         this is the one case where an unfiltered tool dump would actually
         overflow the local model's context window, so it's worth the local
         load.
      3. Otherwise (Gemini fails but chat is on the cloud model, or local
         embedding also fails), fall back to the full unfiltered tool list —
         harmless on cloud context, and never silently disables tool use.
    """
    if not TOOL_EMBEDDINGS:
        return tool_registry.OLLAMA_SCHEMAS

    query_emb = await get_embedding(query)
    if query_emb is not None:
        return _rank_and_select(query_emb, TOOL_EMBEDDINGS)

    if llm.LAST_CHAT_BACKEND == "local":
        local_result = await select_relevant_tools_local(query)
        if local_result is not None:
            return local_result

    return tool_registry.OLLAMA_SCHEMAS
