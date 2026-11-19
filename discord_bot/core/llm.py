"""
llm.py — talks to Ollama (cloud model + local fallback model), including
the debug-dump-on-failure path.

LAST_CHAT_BACKEND is mutable, written here and read by embeddings.py and
memory_store.py. Callers elsewhere in core/ must `import llm` and reference
`llm.LAST_CHAT_BACKEND` — never `from llm import LAST_CHAT_BACKEND`, which
would capture a stale copy at import time and never see later updates.
"""
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp

import config

# Tracks which backend actually served the most recent chat response ("cloud"
# or "local"). Used as a proxy signal in embeddings.select_relevant_tools():
# if Gemini's embedding call fails AND the bot is currently running on the
# local fallback chat model, it's worth paying the local-embedding cost too,
# since context is tight there and an unfiltered 50-tool dump would blow the
# budget. If chat is still on the cloud model, an unfiltered dump is
# harmless, so there's no reason to touch a local embedding model at all.
LAST_CHAT_BACKEND = "cloud"
# Name of the model that actually served the most recent chat response.
LAST_MODEL_USED: "str | None" = None

_session: "aiohttp.ClientSession | None" = None


async def get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


def _dump_failed_payload(payload: dict, status: int, body: str):
    try:
        dump_path = config.DEBUG_DIR / f"failed_payload_{int(time.time())}.json"
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


def _is_cloud_model(name: "str | None") -> bool:
    """Ollama cloud models are tagged \"...:cloud\" or \"...-cloud\"."""
    tag = (name or "").rsplit(":", 1)[-1]
    return tag == "cloud" or tag.endswith("-cloud")


def _with_local_options(payload: dict) -> dict:
    """Adds low-temperature / fixed-context options for local models only.
    Cloud payloads are returned untouched, and an explicit \"options\" key
    already on the payload always wins."""
    if _is_cloud_model(payload.get("model")) or "options" in payload:
        return payload
    return {
        **payload,
        "options": {"temperature": config.LOCAL_TEMPERATURE, "num_ctx": config.LOCAL_NUM_CTX},
    }


async def query_ollama(payload: dict, timeout: int = 90, retries: int = 2) -> dict:
    payload = _with_local_options(payload)
    session = await get_session()
    last_err = None
    for attempt in range(retries + 1):
        try:
            async with session.post(
                config.OLLAMA_API, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
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


async def ollama_model_known(name: str) -> "bool | None":
    """Checks `name` against the Ollama instance's model list (/api/tags on the
    same host as OLLAMA_API). Returns True/False, or None if the list couldn't
    be fetched. Cloud models (e.g. "gemma4:cloud") may not appear in this list
    even though they work, so callers should treat False as a warning, not a
    hard failure."""
    parts = urlsplit(config.OLLAMA_API)
    tags_url = f"{parts.scheme}://{parts.netloc}/api/tags"
    try:
        session = await get_session()
        async with session.get(tags_url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None

    installed = {m.get("name") or m.get("model") for m in data.get("models", []) if isinstance(m, dict)}
    installed.discard(None)
    # Ollama treats a bare name as ":latest".
    wanted = name if ":" in name else f"{name}:latest"
    return name in installed or wanted in installed


# --- Gemini (OpenAI-compatible endpoint) ---------------------------------
#
# The rest of the bot speaks Ollama's chat format (messages with "images",
# assistant "tool_calls" carrying dict arguments, "tool" role messages). These
# helpers translate that to/from the OpenAI format Gemini's compatibility
# endpoint expects, so callers never need to know which backend answered.

class GeminiRateLimited(RuntimeError):
    """Gemini answered 429 (quota exhausted), or the model is cooling down."""


# model -> time.monotonic() before which we don't bother calling it again
_GEMINI_COOLDOWN: "dict[str, float]" = {}


def is_gemini_model(name: "str | None") -> bool:
    """Gemini API ids look like "gemini-3.5-flash-lite" / "gemma-4-31b-it" (no
    colon). Ollama tags always have one ("gemma3:1b", "gemma4:cloud")."""
    n = (name or "").strip().lower()
    return ":" not in n and (n.startswith("gemini-") or n.startswith("gemma-"))


def _record_served(model: "str | None"):
    global LAST_CHAT_BACKEND, LAST_MODEL_USED
    LAST_MODEL_USED = model
    LAST_CHAT_BACKEND = "cloud" if (is_gemini_model(model) or _is_cloud_model(model)) else "local"


def _note_gemini_429(model: str, body: str):
    # A per-day quota won't recover within minutes, so back off for longer.
    daily = "perday" in body.lower().replace(" ", "")
    secs = 3600 if daily else 60
    _GEMINI_COOLDOWN[model] = time.monotonic() + secs
    print(f"[GEMINI] {model} hit a {'daily' if daily else 'per-minute'} limit; "
          f"skipping it for {secs // 60} min. Body: {body[:200]!r}")


def _image_mime(b64: str) -> str:
    if b64.startswith("iVBOR"):
        return "image/png"
    if b64.startswith("R0lGOD"):
        return "image/gif"
    if b64.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


def _to_openai_messages(messages: list) -> list:
    out = []
    pending: "list[tuple[str, str]]" = []  # (tool_call_id, tool name) awaiting a result
    for i, m in enumerate(messages):
        role = m.get("role")
        content = m.get("content") or ""

        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for j, c in enumerate(m["tool_calls"]):
                fn = c.get("function", {})
                args = fn.get("arguments", {})
                if not isinstance(args, str):
                    args = json.dumps(args or {}, ensure_ascii=False)
                cid = c.get("id") or f"call_{i}_{j}"
                entry = {"id": cid, "type": "function",
                         "function": {"name": fn.get("name"), "arguments": args}}
                # Gemini 3 "thought signatures" ride along here and must be echoed back.
                if c.get("extra_content"):
                    entry["extra_content"] = c["extra_content"]
                calls.append(entry)
                pending.append((cid, fn.get("name")))
            msg = {"role": "assistant", "content": content or None, "tool_calls": calls}
            if m.get("extra_content"):
                msg["extra_content"] = m["extra_content"]
            out.append(msg)

        elif role == "tool":
            cid = m.get("tool_call_id")
            if not cid:
                idx = next((k for k, (_, n) in enumerate(pending) if n == m.get("name")), 0 if pending else None)
                cid = pending.pop(idx)[0] if idx is not None else f"call_{i}"
            else:
                pending = [p for p in pending if p[0] != cid]
            msg = {"role": "tool", "tool_call_id": cid, "content": str(content)}
            if m.get("name"):
                msg["name"] = m["name"]
            out.append(msg)

        elif m.get("images"):
            parts = [{"type": "text", "text": content}] if content else []
            for b64 in m["images"]:
                parts.append({"type": "image_url",
                              "image_url": {"url": f"data:{_image_mime(b64)};base64,{b64}"}})
            out.append({"role": role, "content": parts})

        else:
            out.append({"role": role, "content": content})
    return out


def _from_openai_response(data: dict) -> dict:
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    out = {"role": "assistant", "content": msg.get("content") or ""}
    calls = []
    for j, c in enumerate(msg.get("tool_calls") or []):
        fn = c.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            args = {}
        entry = {"id": c.get("id") or f"call_{int(time.time() * 1000)}_{j}",
                 "function": {"name": fn.get("name"), "arguments": args}}
        if c.get("extra_content"):
            entry["extra_content"] = c["extra_content"]
        calls.append(entry)
    if calls:
        out["tool_calls"] = calls
    if msg.get("extra_content"):
        out["extra_content"] = msg["extra_content"]
    return {"model": data.get("model"), "message": out, "done": True}


async def query_gemini(payload: dict, timeout: int = 90, retries: int = 1) -> dict:
    """Sends an Ollama-style payload to Gemini and returns an Ollama-style response."""
    model = payload.get("model")
    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set")
    if time.monotonic() < _GEMINI_COOLDOWN.get(model, 0):
        raise GeminiRateLimited(f"{model} is cooling down after a 429")

    body = {"model": model, "messages": _to_openai_messages(payload.get("messages", []))}
    if payload.get("tools"):
        body["tools"] = payload["tools"]
    opts = payload.get("options") or {}
    if "temperature" in opts:
        body["temperature"] = opts["temperature"]
    if "num_predict" in opts:
        body["max_tokens"] = opts["num_predict"]

    session = await get_session()
    headers = {"Authorization": f"Bearer {config.GEMINI_API_KEY}"}
    last_err = None
    for attempt in range(retries + 1):
        try:
            async with session.post(
                config.GEMINI_CHAT_API, headers=headers, json=body,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                status = resp.status
                if status == 429:
                    text = (await resp.text())[:500]
                    _note_gemini_429(model, text)
                    raise GeminiRateLimited(f"{model} returned 429")
                if status != 200:
                    text = (await resp.text())[:500]
                    _dump_failed_payload(body, status, text)
                    if 500 <= status < 600 and attempt < retries:
                        last_err = RuntimeError(f"Gemini backend returned {status}. Body: {text}")
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    raise RuntimeError(f"Gemini backend returned {status}. Body: {text}")
                return _from_openai_response(await resp.json())
        except (aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
            if attempt < retries:
                last_err = e
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise last_err


async def gemini_model_known(name: str) -> "bool | None":
    """True/False if Google says the model id exists, None if we couldn't check."""
    if not config.GEMINI_API_KEY:
        return None
    try:
        session = await get_session()
        async with session.get(
            config.GEMINI_MODELS_API.format(model=name),
            headers={"x-goog-api-key": config.GEMINI_API_KEY},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 200:
                return True
            if resp.status in (400, 404):
                return False
            return None
    except Exception:
        return None


async def model_known(name: str) -> "bool | None":
    """Backend-aware model check used by !model."""
    if is_gemini_model(name):
        return await gemini_model_known(name)
    return await ollama_model_known(name)


async def _dispatch(payload: dict, timeout: int = 90, retries: int = 2) -> dict:
    if is_gemini_model(payload.get("model")):
        return await query_gemini(payload, timeout=timeout, retries=min(retries, 1))
    return await query_ollama(payload, timeout=timeout, retries=retries)


async def _try_with_siblings(payload: dict, timeout: int, retries: int = 2) -> dict:
    """Calls payload["model"]; if that is a Gemini model and it fails, tries
    each config.GEMINI_FALLBACK_MODELS entry (separate quota) before giving up."""
    model = payload.get("model")
    candidates = [model]
    if is_gemini_model(model):
        candidates += [m for m in config.GEMINI_FALLBACK_MODELS if m != model]
    last_err = None
    for i, m in enumerate(candidates):
        try:
            result = await _dispatch(payload if i == 0 else {**payload, "model": m},
                                     timeout=timeout, retries=retries if i == 0 else 1)
            _record_served(m)
            return result
        except Exception as e:
            last_err = e
            if i + 1 < len(candidates):
                print(f"[FALLBACK] '{m}' failed ({e}); trying '{candidates[i + 1]}'.")
    raise last_err


async def query_llm(payload: dict, timeout: int = 90, channel=None, fallback_model: "str | None" = None) -> dict:
    """Sends the payload to Ollama; on failure, falls back in order:
    1. `fallback_model` (if given and different from the payload's model) — used
       by per-thread model overrides so a failing thread model drops back to the
       main model first.
    2. config.LOCAL_FALLBACK_MODEL — the existing last resort.
    """
    global LAST_CHAT_BACKEND, LAST_MODEL_USED
    # Deferred import: messaging.py needs memory_store.py, which needs this
    # function (llm.query_llm) — importing messaging at module load time
    # here would be circular. By call time (this is only ever awaited, never
    # run at import), messaging.py is fully loaded, so this is safe.
    import messaging

    failed_model = payload.get("model")
    try:
        result = await _try_with_siblings(payload, timeout=timeout)
        return result
    except Exception as cloud_err:
        if fallback_model and fallback_model != failed_model:
            print(f"[FALLBACK] Model '{failed_model}' failed ({cloud_err}); "
                  f"falling back to main model '{fallback_model}'.")
            if channel is not None:
                try:
                    await messaging.send_chunked(
                        channel,
                        f"⚠️ `{failed_model}` is unavailable right now — "
                        f"falling back to the main model `{fallback_model}`..."
                    )
                except Exception:
                    pass
            main_payload = dict(payload)
            main_payload["model"] = fallback_model
            try:
                return await _try_with_siblings(main_payload, timeout=timeout, retries=1)
            except Exception as main_err:
                cloud_err = main_err
                failed_model = fallback_model

        print(f"[FALLBACK] Cloud model '{failed_model}' failed ({cloud_err}); "
              f"falling back to local model '{config.LOCAL_FALLBACK_MODEL}'.")
        if channel is not None:
            try:
                await messaging.send_chunked(
                    channel,
                    f"⚠️ Cloud model (`{failed_model}`) is unavailable right now — "
                    f"falling back to local model `{config.LOCAL_FALLBACK_MODEL}`..."
                )
            except Exception:
                pass
        fallback_payload = dict(payload)
        fallback_payload["model"] = config.LOCAL_FALLBACK_MODEL

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
                await messaging.send_chunked(
                    channel,
                    "⚠️ The local fallback model can't see images — continuing "
                    "without the attached image(s)."
                )
            except Exception:
                pass

        result = await query_ollama(fallback_payload, timeout=timeout, retries=1)
        _record_served(config.LOCAL_FALLBACK_MODEL)
        LAST_CHAT_BACKEND = "local"
        return result
