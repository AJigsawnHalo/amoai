"""
attachments.py — downloads and processes Discord message attachments:
image vision pre-pass, PDF/text/zip text extraction, and reply-context
resolution.

Vision is restricted to the cloud model (gemma4:cloud) — llm.query_llm's
fallback path strips "images" before ever handing a payload to the local
model, so this stays true even if the request falls back mid-flight.
"""
import asyncio
import base64
import io
import zipfile
from pathlib import Path

import discord

import config
import llm
from discord_client import bot

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

MAX_REPLY_CONTEXT_CHARS = 4000  # guard against quoting a huge message wholesale


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
        "model": config.MODEL_NAME,
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
        response = await llm.query_llm(vision_payload, timeout=90, channel=channel)
    except Exception as e:
        print(f"[VISION] Image description pass failed: {e}")
        return None

    if llm.LAST_CHAT_BACKEND != "cloud":
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
