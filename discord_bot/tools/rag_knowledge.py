"""
rag_knowledge.py

A lightweight local RAG (Retrieval-Augmented Generation) tool for the Discord bot.
Indexes markdown notes, plain text, code files, and PDFs into a SQLite-backed
vector store, and lets the bot semantically recall relevant passages later.

No external vector DB needed — just sqlite3 (stdlib) + Ollama embeddings
(reusing the same local Ollama instance the bot already talks to).

Public tool functions (auto-discovered by bot.py's register_tools()):
    - index_knowledge_base(path)
    - search_knowledge(query, top_k)
    - list_knowledge_sources()
    - forget_knowledge_source(source)

Everything prefixed with "_" is a private helper and will NOT be
registered as an LLM-callable tool.

Boundary with scratchpad_tool.py: this module only knows about files that
were explicitly indexed with index_knowledge_base. A quick thing the user
jots down in chat (e.g. "note that the wifi password changed") belongs in
scratchpad_tool.py instead, not here.
"""

import os
import json
import math
import sqlite3
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv, find_dotenv

# Same pattern as monitor.py — lets this module also be run standalone
# (outside the bot process) and still pick up the central .env file.
load_dotenv(find_dotenv())

# --- CONFIGURATION ---
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "rag_knowledge.sqlite3"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

OLLAMA_EMBED_API = os.getenv("OLLAMA_EMBED_API", "http://localhost:11434/api/embeddings")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Same containment pattern as file_ops.py — indexing is locked to this
# directory (and its subfolders) so the bot can't be tricked into reading
# and embedding files outside its intended scope. Reuses FILE_TOOL_BASE_DIR
# if you've already set one for file_ops.py; falls back to a dedicated
# RAG_TOOL_BASE_DIR, then your home dir.
BASE_DIR = Path(
    os.getenv("RAG_TOOL_BASE_DIR", os.getenv("FILE_TOOL_BASE_DIR", os.path.expanduser("~")))
).resolve()

MAX_INDEX_FILE_BYTES = 2_000_000  # skip individual files bigger than ~2MB

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 150    # characters of overlap between chunks
SUPPORTED_TEXT_EXTS = {
    ".md", ".markdown", ".txt", ".py", ".js", ".ts", ".json", ".yaml",
    ".yml", ".toml", ".cfg", ".ini", ".sh", ".html", ".css", ".sql",
}


class PathSecurityError(Exception):
    pass


def _resolve_safe(path: str) -> Path:
    """Resolves a user-supplied path and guarantees it stays inside BASE_DIR.
    Blocks '../' traversal and symlink escapes. Supports '~' shortcuts."""
    candidate = Path(path).expanduser()
    candidate = candidate if candidate.is_absolute() else BASE_DIR / candidate
    candidate = candidate.resolve()

    try:
        candidate.relative_to(BASE_DIR)
    except ValueError:
        raise PathSecurityError(
            f"Refusing — that path resolves to {candidate}, which is outside "
            f"the allowed directory ({BASE_DIR})."
        )
    return candidate


# --- DB SETUP ---
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            embedding TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            UNIQUE(source, chunk_index)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON chunks(source)")
    return conn


# --- TEXT EXTRACTION ---
def _extract_text(path: Path) -> str:
    ext = path.suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # fallback for older installs
            except ImportError:
                raise RuntimeError(
                    "PDF support requires 'pypdf' (pip install pypdf)"
                )
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if ext in SUPPORTED_TEXT_EXTS or ext == "":
        return path.read_text(encoding="utf-8", errors="ignore")

    raise ValueError(f"Unsupported file type: {ext}")


def _iter_indexable_files(root: Path):
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        if p.is_file() and (p.suffix.lower() in SUPPORTED_TEXT_EXTS or p.suffix.lower() == ".pdf"):
            # skip common junk directories
            if any(part in {".git", "node_modules", "__pycache__", ".venv", "venv"} for part in p.parts):
                continue
            yield p


# --- CHUNKING ---
def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        # try to break on a paragraph/sentence boundary near the end
        if end < length:
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start + (chunk_size // 2):
                boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + (chunk_size // 2):
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


# --- EMBEDDINGS ---
def _embed(text: str) -> list:
    resp = requests.post(
        OLLAMA_EMBED_API,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    embedding = data.get("embedding")
    if not embedding:
        raise RuntimeError(f"No embedding returned by {EMBED_MODEL}: {data}")
    return embedding


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ============================================================
# PUBLIC TOOLS (discovered + exposed to the LLM by bot.py)
# ============================================================

def index_knowledge_base(path: str) -> str:
    """
    Indexes a local file or folder (markdown notes, code, .txt, or PDFs) into
    the RAG knowledge base so its content can be searched and recalled later.
    Pass a single file path or a directory path — directories are scanned
    recursively for supported files. Re-indexing a file only re-embeds
    chunks that actually changed. Only paths inside the configured base
    directory can be indexed. Use this whenever the user asks to
    remember, learn, ingest, index, or add a document/folder/notes to memory.
    """
    target = Path(path).expanduser()
    if not target.exists():
        return f"❌ Path not found: {path}"

    try:
        target = _resolve_safe(str(target))
    except PathSecurityError as e:
        return f"❌ {e}"

    files = list(_iter_indexable_files(target))
    if not files:
        return f"⚠️ No supported files found at {path} (supported: markdown, text, code, .pdf)."

    conn = _get_conn()
    total_chunks = 0
    skipped_unchanged = 0
    skipped_too_large = 0
    files_indexed = 0
    errors = []

    for f in files:
        if f.stat().st_size > MAX_INDEX_FILE_BYTES:
            skipped_too_large += 1
            continue

        try:
            text = _extract_text(f)
        except Exception as e:
            errors.append(f"{f.name}: {e}")
            continue

        chunks = _chunk_text(text)
        if not chunks:
            continue

        source = str(f.resolve())
        now = datetime.now(timezone.utc).isoformat()

        for i, chunk in enumerate(chunks):
            chunk_hash = _hash(chunk)
            existing = conn.execute(
                "SELECT content_hash FROM chunks WHERE source = ? AND chunk_index = ?",
                (source, i),
            ).fetchone()
            if existing and existing[0] == chunk_hash:
                skipped_unchanged += 1
                continue

            embedding = _embed(chunk)
            conn.execute(
                """
                INSERT INTO chunks (source, chunk_index, content, content_hash, embedding, indexed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, chunk_index) DO UPDATE SET
                    content = excluded.content,
                    content_hash = excluded.content_hash,
                    embedding = excluded.embedding,
                    indexed_at = excluded.indexed_at
                """,
                (source, i, chunk, chunk_hash, json.dumps(embedding), now),
            )
            total_chunks += 1

        # remove stale chunks if the file shrank (fewer chunks than before)
        conn.execute(
            "DELETE FROM chunks WHERE source = ? AND chunk_index >= ?",
            (source, len(chunks)),
        )
        conn.commit()
        files_indexed += 1

    conn.close()

    msg = f"✅ Indexed {files_indexed} file(s), {total_chunks} chunk(s) embedded"
    if skipped_unchanged:
        msg += f", {skipped_unchanged} unchanged chunk(s) skipped"
    if skipped_too_large:
        msg += f", {skipped_too_large} file(s) skipped (over {MAX_INDEX_FILE_BYTES // 1_000_000}MB)"
    msg += "."
    if errors:
        msg += "\n⚠️ Errors:\n" + "\n".join(errors[:10])
    return msg


def search_knowledge(query: str, top_k: int = 5) -> str:
    """
    Searches the indexed knowledge base (markdown notes, code snippets, PDFs)
    for passages most relevant to the query and returns them with their
    source file. This is the canonical tool for "based on my notes" and
    "search my notes" — the word "notes" in this bot ALWAYS refers to this
    indexed knowledge base, never scratchpad_tool.py's entries, even if the
    user jotted something down earlier in the same conversation. Use this
    whenever the user asks to recall, find, look up, search, or reference
    something from previously indexed notes, docs, or code.
    """
    conn = _get_conn()
    rows = conn.execute("SELECT source, chunk_index, content, embedding FROM chunks").fetchall()
    conn.close()

    if not rows:
        return "⚠️ The knowledge base is empty. Index some files first with index_knowledge_base."

    try:
        query_embedding = _embed(query)
    except Exception as e:
        return f"❌ Failed to embed query: {e}"

    scored = []
    for source, chunk_index, content, embedding_json in rows:
        embedding = json.loads(embedding_json)
        score = _cosine_similarity(query_embedding, embedding)
        scored.append((score, source, chunk_index, content))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, min(top_k, 20))]

    if not top or top[0][0] < 0.15:
        return "No sufficiently relevant results found in the knowledge base for that query."

    lines = [f"🔎 Top {len(top)} result(s) for: \"{query}\"\n"]
    for score, source, chunk_index, content in top:
        name = Path(source).name
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > 800:
            snippet = snippet[:800].rsplit(" ", 1)[0] + "..."
        lines.append(f"**{name}** (chunk {chunk_index}, score {score:.2f})\n{snippet}\n")

    return "\n".join(lines)


def list_knowledge_sources() -> str:
    """
    Lists every file currently indexed in the RAG knowledge base along with
    how many chunks each one has. Use this when the user asks what's been
    indexed, what notes/docs the bot knows about, or wants an overview of
    the knowledge base.
    """
    conn = _get_conn()
    rows = conn.execute(
        "SELECT source, COUNT(*), MAX(indexed_at) FROM chunks GROUP BY source ORDER BY source"
    ).fetchall()
    conn.close()

    if not rows:
        return "The knowledge base is currently empty."

    lines = [f"📚 {len(rows)} source(s) indexed:\n"]
    for source, count, last_indexed in rows:
        lines.append(f"- {Path(source).name} — {count} chunk(s), last indexed {last_indexed}")
    return "\n".join(lines)


def forget_knowledge_source(source: str) -> str:
    """
    Removes a previously indexed file from the RAG knowledge base, matched
    by exact path or filename substring. Use this when the user asks to
    forget, remove, delete, or un-index a specific file or notes.
    """
    conn = _get_conn()
    rows = conn.execute("SELECT DISTINCT source FROM chunks").fetchall()
    matches = [r[0] for r in rows if source.lower() in r[0].lower()]

    if not matches:
        conn.close()
        return f"⚠️ No indexed source matching '{source}' was found."

    for m in matches:
        conn.execute("DELETE FROM chunks WHERE source = ?", (m,))
    conn.commit()
    conn.close()

    removed_names = ", ".join(Path(m).name for m in matches)
    return f"🗑️ Removed {len(matches)} source(s) from the knowledge base: {removed_names}"

