import os
import shutil
from datetime import datetime
from pathlib import Path

# --- CONFIGURATION ---
# All operations are locked to this directory (and its subfolders) so the bot
# can't be tricked into touching files outside its intended scope. Set
# FILE_TOOL_BASE_DIR in your .env to whatever folder you want it managing
# (e.g. /home/elskiee/managed-files). Defaults to your home dir if unset.
BASE_DIR = Path(os.getenv("FILE_TOOL_BASE_DIR", os.path.expanduser("~"))).resolve()

MAX_READ_BYTES = 200_000   # ~200KB cap so a huge file doesn't blow up the context
MAX_WRITE_BYTES = 500_000  # 500KB cap on content written in one call


class PathSecurityError(Exception):
    pass


def _resolve_safe(path: str) -> Path:
    """Resolves a user-supplied path and guarantees it stays inside BASE_DIR.
    Blocks '../' traversal and symlink escapes.
    Now supports user home shortcuts like '~'."""
    # 1. Convert to Path object and expand home shortcuts (e.g., ~ -> /home/user)
    candidate = Path(path).expanduser()
    
    # 2. If it's not absolute, resolve it relative to BASE_DIR
    candidate = candidate if candidate.is_absolute() else BASE_DIR / candidate
    candidate = candidate.resolve()
    
    # 3. Prevent directory traversal attacks
    try:
        candidate.relative_to(BASE_DIR)
    except ValueError:
        raise PathSecurityError(
            f"Refusing — that path resolves to {candidate}, which is outside "
            f"the allowed directory ({BASE_DIR})."
        )
    return candidate


def list_directory(path: str = ".") -> str:
    """Lists the contents of a folder on the server: file/folder names, sizes,
    and last-modified times. Use this whenever the user asks to browse, see
    what's in a folder, or locate a file."""
    try:
        target = _resolve_safe(path)
    except PathSecurityError as e:
        return f"❌ {e}"

    if not target.exists():
        return f"❌ Path does not exist: {target}"
    if not target.is_dir():
        return f"❌ Not a directory: {target}"

    rows = []
    for entry in sorted(target.iterdir()):
        try:
            st = entry.stat()
            kind = "DIR " if entry.is_dir() else "FILE"
            size = "-" if entry.is_dir() else f"{st.st_size}B"
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            rows.append(f"[{kind}] {entry.name:<40} {size:>10}   {mtime}")
        except OSError:
            rows.append(f"[????] {entry.name}")

    if not rows:
        return f"📂 {target} is empty."
    return f"📂 Contents of {target}:\n" + "\n".join(rows)


def read_file(path: str) -> str:
    """Reads and returns the text content of a file on the server. Use this
    whenever the user asks to view, check, or read a specific file. Refuses
    files that are binary or too large to return sensibly."""
    try:
        target = _resolve_safe(path)
    except PathSecurityError as e:
        return f"❌ {e}"

    if not target.exists():
        return f"❌ File does not exist: {target}"
    if target.is_dir():
        return f"❌ {target} is a directory — use list_directory instead."

    size = target.stat().st_size
    if size > MAX_READ_BYTES:
        return (f"❌ {target} is {size} bytes, over the {MAX_READ_BYTES}-byte "
                f"read limit. Ask for a specific section instead.")

    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"❌ {target} doesn't look like a text file (binary content)."
    except OSError as e:
        return f"❌ Failed to read {target}: {e}"

    return f"📄 {target} ({size} bytes):\n```\n{content}\n```"


def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Creates a new file with the given text content. Will NOT overwrite an
    existing file unless overwrite is explicitly set to true, to avoid silent
    data loss. Use this whenever the user asks to create, save, or write text
    to a file."""
    try:
        target = _resolve_safe(path)
    except PathSecurityError as e:
        return f"❌ {e}"

    if len(content.encode("utf-8")) > MAX_WRITE_BYTES:
        return f"❌ Content exceeds the {MAX_WRITE_BYTES}-byte write limit."

    if target.exists() and not overwrite:
        return (f"⚠️ {target} already exists. Call write_file again with "
                f"overwrite=true if you really want to replace it.")

    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as e:
        return f"❌ Failed to write {target}: {e}"

    return f"✅ Wrote {len(content.encode('utf-8'))} bytes to {target}"


def copy_file(source: str, destination: str, overwrite: bool = False) -> str:
    """Copies a file or folder to a new location, leaving the original in
    place. Use this whenever the user asks to copy, duplicate, or back up
    a file."""
    try:
        src = _resolve_safe(source)
        dst = _resolve_safe(destination)
    except PathSecurityError as e:
        return f"❌ {e}"

    if not src.exists():
        return f"❌ Source does not exist: {src}"
    if dst.exists() and not overwrite:
        return (f"⚠️ Destination {dst} already exists. Call copy_file again "
                f"with overwrite=true if you want to replace it.")

    try:
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    except OSError as e:
        return f"❌ Failed to copy {src} -> {dst}: {e}"

    return f"✅ Copied {src} -> {dst}"


def move_file(source: str, destination: str, overwrite: bool = False) -> str:
    """DESTRUCTIVE. Moves or renames a file or folder on the server — the
    source stops existing at its old location. Use this whenever the user
    asks to move, rename, or relocate a file. This tool requires user
    confirmation before it runs."""
    try:
        src = _resolve_safe(source)
        dst = _resolve_safe(destination)
    except PathSecurityError as e:
        return f"❌ {e}"

    if not src.exists():
        return f"❌ Source does not exist: {src}"
    if dst.exists() and not overwrite:
        return (f"⚠️ Destination {dst} already exists. Call move_file again "
                f"with overwrite=true if you want to replace it.")

    try:
        if dst.exists():
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    except OSError as e:
        return f"❌ Failed to move {src} -> {dst}: {e}"

    return f"✅ Moved {src} -> {dst}"


def delete_file(path: str, recursive: bool = False) -> str:
    """DESTRUCTIVE AND IRREVERSIBLE. Permanently deletes a file, or (with
    recursive=true) an entire folder and everything inside it. There is no
    trash bin — deleted data cannot be recovered. Use this whenever the user
    asks to delete or remove a file. This tool requires user confirmation
    before it runs."""
    try:
        target = _resolve_safe(path)
    except PathSecurityError as e:
        return f"❌ {e}"

    if not target.exists():
        return f"❌ Path does not exist: {target}"
    if target == BASE_DIR:
        return "❌ Refusing to delete the root managed directory itself."

    try:
        if target.is_dir():
            if not recursive:
                return (f"⚠️ {target} is a directory. Call delete_file again "
                         f"with recursive=true to confirm deleting it and "
                         f"everything inside.")
            shutil.rmtree(target)
        else:
            target.unlink()
    except OSError as e:
        return f"❌ Failed to delete {target}: {e}"

    return f"✅ Deleted {target}"
