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


async def query_ollama(payload: dict, timeout: int = 90, retries: int = 2) -> dict:
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


async def query_llm(payload: dict, timeout: int = 90, channel=None) -> dict:
    global LAST_CHAT_BACKEND
    # Deferred import: messaging.py needs memory_store.py, which needs this
    # function (llm.query_llm) — importing messaging at module load time
    # here would be circular. By call time (this is only ever awaited, never
    # run at import), messaging.py is fully loaded, so this is safe.
    import messaging

    try:
        result = await query_ollama(payload, timeout=timeout)
        LAST_CHAT_BACKEND = "cloud"
        return result
    except Exception as cloud_err:
        print(f"[FALLBACK] Cloud model '{payload.get('model')}' failed ({cloud_err}); "
              f"falling back to local model '{config.LOCAL_FALLBACK_MODEL}'.")
        if channel is not None:
            try:
                await messaging.send_chunked(
                    channel,
                    f"⚠️ Cloud model (`{payload.get('model')}`) is unavailable right now — "
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
        LAST_CHAT_BACKEND = "local"
        return result
