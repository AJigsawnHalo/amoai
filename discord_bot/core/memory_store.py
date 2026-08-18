"""
memory_store.py — conversation history (per-channel rolling window +
persistent log) and long-term per-user facts (SQLite-backed), including
dedup, supersession, and consolidation.

Both conversation_log and facts go through the SAME SQLite connection
(_get_memory_conn) and the same database file — this module owns that
connection/schema; nothing else should open MEMORY_DB_PATH directly.

extract_and_store_facts() needs messaging.send_chunked to post "remembered:"
notices, but messaging.py needs get_channel_history/_append_conversation_log
from THIS module — that's a real two-way dependency. It's broken by
importing messaging lazily, inside the one function that needs it, rather
than at module load time (see the `import messaging` inside
extract_and_store_facts below). messaging.py's top-level `import
memory_store` stays a normal import; this is the module that defers.
"""
import json
import re
import sqlite3
from collections import deque
from datetime import datetime, timezone

import discord

import config
import llm
import embeddings

# --- CONVERSATION MEMORY ---
# Per-channel rolling window, kept in memory for fast access during a
# session. Threads get a larger window than the main channel — a thread is
# a dedicated, bounded conversation (like a Claude chat), so it's worth
# keeping more of it around; the main channel is shared/ambient, so a
# tighter window keeps the prompt from dragging in unrelated topics.
HISTORY_TURNS = 10
THREAD_HISTORY_TURNS = 40
# Independent of the in-memory window above — this is how much of a
# channel's/thread's history survives in SQLite across bot restarts, so
# reopening an old thread days later doesn't come back empty.
CONVERSATION_LOG_MAX_PER_CHANNEL = 500

CHANNEL_HISTORY: "dict[int, deque]" = {}
_HYDRATED_CHANNELS: "set[int]" = set()


def _history_cap_messages(channel) -> int:
    turns = THREAD_HISTORY_TURNS if isinstance(channel, discord.Thread) else HISTORY_TURNS
    return turns * 2


# --- PERSISTENT USER MEMORY (SQLite-backed) ---
# Was a flat memory_store.json capped at 40 facts/user. Moved to SQLite
# (same pattern as tools/rag_knowledge.py's vector store) so raising the cap
# is just a number, not a rewrite, and per-user lookups don't require
# loading every other user's facts into memory first.
MEMORY_DB_PATH = config.DATA_DIR / "memory_store.sqlite3"
# The legacy pre-SQLite file is NOT part of the new data/ layout — it's only
# ever read once, by the migration below, and deliberately left wherever it
# already sits (see migrate_data_layout.py, which does not move it either).
LEGACY_MEMORY_FILE = config.DATA_DIR.parent / "memory_store.json"
MAX_FACTS_PER_USER = 200  # was 40 on the old JSON store

# Two facts whose embeddings score at or above this are close enough to be
# treated as verbatim the same and skipped outright — no need to even ask
# about it. In practice most real near-duplicates ("likes coffee" vs "really
# likes coffee in the morning") land a bit below this, which is why there's
# a second, lower band below for those.
FACT_DEDUP_THRESHOLD = 0.93
# Two facts whose embeddings score in [FACT_SUPERSESSION_BAND_MIN,
# FACT_DEDUP_THRESHOLD) are similar enough that they're probably about the
# same underlying thing but not similar enough to safely auto-skip — that's
# exactly the "slightly different phrasing" case that was slipping through
# before. Rather than trust the original broad extraction call to have
# already caught this (it often doesn't, especially on smaller models), a
# candidate in this band gets one focused, single-pair LLM comparison before
# being stored — see _find_supersession_candidate / _llm_decides_replace.
# Start conservative; if it's still merging facts that were actually meant
# to coexist, raise this number, and if near-duplicates keep slipping
# through as separate entries, lower it.
FACT_SUPERSESSION_BAND_MIN = 0.78
# Fact lists at or under this size are injected into the system prompt
# whole — not worth the embedding/ranking overhead. Above it, only the
# FACT_RELEVANCE_TOP_K most relevant facts to the current message go in, so
# the prompt doesn't grow unbounded as a user's fact count approaches
# MAX_FACTS_PER_USER.
FACT_INJECT_ALWAYS_UNDER = 8
FACT_RELEVANCE_TOP_K = 8

# Two facts whose embeddings score at or above this are treated as "about
# the same underlying topic" for consolidation purposes — deliberately a
# much looser band than FACT_SUPERSESSION_BAND_MIN above. Supersession is
# about two facts being restatements of the SAME fact (one should replace
# the other); this is about facts that are genuinely distinct but all
# orbiting one subject (e.g. five separate facts about one worldbuilding
# nation's government, motto, symbols, and history) — the case that
# generates a wall of individual "remembered:" lines from one big source
# dump without ever tripping dedup or supersession, since each is a real,
# non-duplicate fact.
CONSOLIDATION_TOPIC_THRESHOLD = 0.60
# A cluster only gets merged once it has at least this many facts — small
# clusters aren't worth the LLM call or the risk of losing detail.
CONSOLIDATION_MIN_CLUSTER_SIZE = 3
# Bounds worst-case GPU-time per consolidation pass regardless of how many
# eligible clusters exist — each cluster is still its own isolated LLM
# call (deliberately NOT batched into one call across clusters: a smaller
# cloud model is meaningfully more prone to bleeding a detail from one
# cluster into another's merged output when they're all in the same
# prompt, and one bad/truncated response would take the whole pass down
# instead of just one cluster). Only the largest N eligible clusters get
# merged in a given pass; anything left over just persists and gets
# caught on a later trigger — nothing is lost, only deferred.
CONSOLIDATION_MAX_CLUSTERS_PER_PASS = 5
# Consolidation isn't scheduled — it runs reactively, checked right before
# a new fact would be added, so it fires exactly when a user is about to
# hit or has hit MAX_FACTS_PER_USER, instead of on a clock. The margin
# gives it room to run BEFORE add_user_fact's own oldest-first trim would
# otherwise silently delete anything.
CONSOLIDATION_NEAR_CAP_MARGIN = 15
CONSOLIDATION_TRIGGER_COUNT = MAX_FACTS_PER_USER - CONSOLIDATION_NEAR_CAP_MARGIN

# Cheap pre-filter so extract_and_store_facts doesn't burn an LLM call on
# every single message (most messages are acknowledgments or one-off
# requests with nothing durable in them). False negatives here just mean a
# fact-bearing message got skipped — it almost always resurfaces in a later,
# more substantive message, so skipping is low-risk; the alternative is
# paying for an extraction call on every "ok thanks".
_LOW_SIGNAL_PATTERN = re.compile(
    r"^(ok(ay)?|k+|thanks?( you)?|thx|ty|cool|nice|lol+|lmao+|yes|yep|yeah|no|nope|"
    r"sure|got ?it|alright|sounds good|np|welcome)[.!?]*$",
    re.IGNORECASE,
)


def _looks_low_signal(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 12:
        return True
    return bool(_LOW_SIGNAL_PATTERN.match(stripped))


def _get_memory_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            fact TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(user_id, fact)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            channel_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_convlog_channel ON conversation_log(channel_id)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_suggestion_state (
            channel_id TEXT PRIMARY KEY,
            last_suggested_at INTEGER NOT NULL
        )
        """
    )
    # Additive migration for DBs created before updated_at/embedding existed.
    # ADD COLUMN has no "IF NOT EXISTS" in SQLite, so this just no-ops with
    # an OperationalError ("duplicate column") on every run after the first.
    for ddl in (
        "ALTER TABLE facts ADD COLUMN updated_at TEXT",
        "ALTER TABLE facts ADD COLUMN embedding TEXT",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass
    return conn


# --- conversation log / rolling history ---

def _load_conversation_log(channel_id, limit_messages: int) -> list:
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT role, content FROM conversation_log WHERE channel_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (str(channel_id), limit_messages),
        ).fetchall()
    finally:
        conn.close()
    return [{"role": role, "content": content} for role, content in reversed(rows)]


def _append_conversation_log(channel_id, role: str, content: str):
    conn = _get_memory_conn()
    try:
        conn.execute(
            "INSERT INTO conversation_log (channel_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (str(channel_id), role, content, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM conversation_log WHERE channel_id = ?", (str(channel_id),)
        ).fetchone()[0]
        if count > CONVERSATION_LOG_MAX_PER_CHANNEL:
            overflow = count - CONVERSATION_LOG_MAX_PER_CHANNEL
            conn.execute(
                """
                DELETE FROM conversation_log WHERE id IN (
                    SELECT id FROM conversation_log WHERE channel_id = ? ORDER BY id ASC LIMIT ?
                )
                """,
                (str(channel_id), overflow),
            )
            conn.commit()
    finally:
        conn.close()


def get_channel_history(channel) -> deque:
    """Returns the in-memory rolling-history deque for this channel/thread,
    hydrating it from the persistent conversation_log on first touch since
    process start so a bot restart doesn't blank out an in-progress thread."""
    cid = channel.id
    if cid not in CHANNEL_HISTORY:
        cap = _history_cap_messages(channel)
        history = deque(maxlen=cap)
        if cid not in _HYDRATED_CHANNELS:
            history.extend(_load_conversation_log(cid, cap))
            _HYDRATED_CHANNELS.add(cid)
        CHANNEL_HISTORY[cid] = history
    return CHANNEL_HISTORY[cid]


def record_turn(channel, user_query: str, response_text: str):
    """Appends a user/assistant exchange to both the in-memory window and
    the persistent log — call this everywhere a turn currently gets pushed
    onto CHANNEL_HISTORY."""
    history = get_channel_history(channel)
    history.append({"role": "user", "content": user_query})
    history.append({"role": "assistant", "content": response_text})
    _append_conversation_log(channel.id, "user", user_query)
    _append_conversation_log(channel.id, "assistant", response_text)


# --- thread-suggestion state (persisted here since it shares the DB/connection) ---

def _get_last_thread_suggestion(channel_id) -> int:
    conn = _get_memory_conn()
    try:
        row = conn.execute(
            "SELECT last_suggested_at FROM thread_suggestion_state WHERE channel_id = ?",
            (str(channel_id),),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else 0


def _get_conversation_message_count(channel_id) -> int:
    """True, unbounded count of messages logged for this channel so far.

    Deliberately NOT len(get_channel_history(channel)) — that deque is capped
    at HISTORY_TURNS*2 / THREAD_HISTORY_TURNS*2 messages, so its len() stops
    growing once a conversation passes that cap and just sits pinned at the
    ceiling forever. Used as the cooldown math's input, that plateau makes
    "history_length - last_suggested_at >= COOLDOWN" permanently unsatisfiable
    after the first couple of suggestions, silently disabling the whole
    feature for that channel for good. conversation_log isn't capped until
    500 messages/channel, comfortably above where this feature operates.
    """
    conn = _get_memory_conn()
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM conversation_log WHERE channel_id = ?",
            (str(channel_id),),
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else 0


def _set_last_thread_suggestion(channel_id, history_length: int):
    conn = _get_memory_conn()
    try:
        conn.execute(
            "INSERT INTO thread_suggestion_state (channel_id, last_suggested_at) VALUES (?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET last_suggested_at = excluded.last_suggested_at",
            (str(channel_id), history_length),
        )
        conn.commit()
    finally:
        conn.close()


# --- fact embedding helpers ---

def _encode_embedding(embedding: "tuple[str, list[float]] | None") -> "str | None":
    if not embedding:
        return None
    space, vector = embedding
    if not vector:
        return None
    return json.dumps({"space": space, "vector": vector})


def _parse_embedding(raw: "str | None") -> "tuple[str, list[float]] | None":
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        vector = obj.get("vector")
        return (obj.get("space"), vector) if vector else None
    except (json.JSONDecodeError, AttributeError, TypeError):
        return None


async def _embed_fact(text: str) -> "tuple[str, list[float]] | None":
    """Tiered embedding for a fact or a query string — same fallback order
    as embeddings.select_relevant_tools(): Gemini primary, local nomic only
    when chat is currently on the local backend. Returns (space, vector)
    rather than a bare vector so callers only ever compare vectors embedded
    in the same space — Gemini and local vectors are never comparable to
    each other (see the note on embeddings.TOOL_EMBEDDINGS_LOCAL)."""
    emb = await embeddings.get_embedding(text)
    if emb is not None:
        return "gemini", emb
    if llm.LAST_CHAT_BACKEND == "local":
        emb = await embeddings.get_local_embedding(text)
        if emb is not None:
            return "local", emb
    return None


def _migrate_legacy_json_memory():
    """One-time migration from the old memory_store.json into SQLite. Runs
    at import time; no-ops once the JSON file is gone or already migrated
    (renamed to .json.migrated on success, so this only ever runs once)."""
    if not LEGACY_MEMORY_FILE.exists():
        return
    try:
        with open(LEGACY_MEMORY_FILE, "r", encoding="utf-8") as f:
            legacy_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[MEMORY] Couldn't read legacy {LEGACY_MEMORY_FILE.name}, leaving it in place: {e}")
        return

    conn = _get_memory_conn()
    migrated_any = False
    try:
        for user_id, facts in legacy_data.items():
            for fact in facts:
                fact = fact.strip()
                if not fact:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO facts (user_id, fact, created_at) VALUES (?, ?, ?)",
                    (str(user_id), fact, datetime.now(timezone.utc).isoformat()),
                )
                migrated_any = True
        conn.commit()
    finally:
        conn.close()

    if migrated_any:
        backup_path = LEGACY_MEMORY_FILE.with_suffix(".json.migrated")
        try:
            LEGACY_MEMORY_FILE.rename(backup_path)
            print(f"[MEMORY] Migrated {LEGACY_MEMORY_FILE.name} into SQLite -> {backup_path.name}")
        except OSError as e:
            print(f"[MEMORY] Migrated to SQLite but couldn't rename old file: {e}")


_migrate_legacy_json_memory()


def get_user_facts(user_id: str) -> list:
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT fact FROM facts WHERE user_id = ? ORDER BY id ASC",
            (str(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def get_user_facts_with_embeddings(user_id: str) -> list:
    """Returns [(fact, (space, vector) | None), ...] for this user — used for
    semantic dedup and relevance ranking. Facts stored before embeddings
    existed, or whose embedding call failed at the time, come back with
    None; callers treat those as always-relevant/always-distinct rather than
    hiding them, same philosophy as embeddings.select_relevant_tools() for
    tools with no cached embedding."""
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT fact, embedding FROM facts WHERE user_id = ? ORDER BY id ASC",
            (str(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return [(fact, _parse_embedding(raw)) for fact, raw in rows]


def add_user_fact(user_id: str, fact: str, embedding: "tuple[str, list[float]] | None" = None) -> bool:
    """Inserts a single fact. Returns False if it already exists verbatim for
    this user (UNIQUE constraint) or the fact is blank — semantic near-dupes
    should be filtered by the caller with _is_semantic_duplicate before this
    is ever reached."""
    fact = fact.strip()
    if not fact:
        return False
    conn = _get_memory_conn()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT OR IGNORE INTO facts (user_id, fact, created_at, updated_at, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(user_id), fact, now, now, _encode_embedding(embedding)),
        )
        inserted = bool(cur.rowcount)
        conn.commit()

        if inserted:
            # Enforce the per-user cap by trimming the oldest rows beyond it,
            # same "keep the newest N" behavior the old JSON store had.
            count = conn.execute(
                "SELECT COUNT(*) FROM facts WHERE user_id = ?", (str(user_id),)
            ).fetchone()[0]
            if count > MAX_FACTS_PER_USER:
                overflow = count - MAX_FACTS_PER_USER
                conn.execute(
                    """
                    DELETE FROM facts WHERE id IN (
                        SELECT id FROM facts WHERE user_id = ? ORDER BY id ASC LIMIT ?
                    )
                    """,
                    (str(user_id), overflow),
                )
                conn.commit()
        return inserted
    finally:
        conn.close()


def update_user_fact(
    user_id: str, old_fact: str, new_fact: str, embedding: "tuple[str, list[float]] | None" = None
) -> "tuple[str, str] | None":
    """Finds old_fact by case-insensitive exact match and rewrites its text
    and embedding in place, preserving its row rather than appending — this
    is what lets a revised fact ("moved to Manila") supersede the old one
    ("lives in Calumpit") instead of both persisting side by side forever.
    Returns (old_text, new_text) on success, or None if old_fact wasn't
    found (the caller should then treat it as a plain add)."""
    new_fact = new_fact.strip()
    if not new_fact:
        return None
    conn = _get_memory_conn()
    try:
        row = conn.execute(
            "SELECT id, fact FROM facts WHERE user_id = ? AND lower(fact) = lower(?)",
            (str(user_id), old_fact.strip()),
        ).fetchone()
        if not row:
            return None
        fact_id, old_text = row
        conn.execute(
            "UPDATE facts SET fact = ?, updated_at = ?, embedding = ? WHERE id = ?",
            (new_fact, datetime.now(timezone.utc).isoformat(), _encode_embedding(embedding), fact_id),
        )
        conn.commit()
        return old_text, new_fact
    finally:
        conn.close()


def _is_semantic_duplicate(
    embedding: "tuple[str, list[float]] | None", existing: list
) -> bool:
    """True if `embedding` is close enough to something already stored that
    a new row isn't worth adding. Only ever compares within the same
    embedding space (Gemini vs local vectors are not comparable — see
    _embed_fact); if either side has no embedding this returns False rather
    than silently dropping a fact it can't judge."""
    if not embedding:
        return False
    space, vector = embedding
    for _, existing_embedding in existing:
        if not existing_embedding:
            continue
        existing_space, existing_vector = existing_embedding
        if existing_space != space:
            continue
        if embeddings.cosine_similarity(vector, existing_vector) >= FACT_DEDUP_THRESHOLD:
            return True
    return False


def _find_supersession_candidate(
    embedding: "tuple[str, list[float]] | None", existing: list
) -> "tuple[str, float] | None":
    """Finds the existing fact most similar to `embedding`, if its score
    falls in [FACT_SUPERSESSION_BAND_MIN, FACT_DEDUP_THRESHOLD) — similar
    enough to plausibly be the same underlying fact restated, but not so
    similar that _is_semantic_duplicate already caught it. Returns
    (existing_fact_text, score) or None. Same-space-only, same reasoning as
    _is_semantic_duplicate."""
    if not embedding:
        return None
    space, vector = embedding
    best = None
    for fact_text, existing_embedding in existing:
        if not existing_embedding:
            continue
        existing_space, existing_vector = existing_embedding
        if existing_space != space:
            continue
        score = embeddings.cosine_similarity(vector, existing_vector)
        if FACT_SUPERSESSION_BAND_MIN <= score < FACT_DEDUP_THRESHOLD:
            if best is None or score > best[1]:
                best = (fact_text, score)
    return best


async def _llm_decides_replace(existing_fact: str, new_fact: str) -> bool:
    """Focused single-pair comparison — deliberately a much narrower ask than
    the original extraction prompt (which has to scan the whole fact list
    and produce structured JSON in one pass, and in practice often misses
    this). A yes/no call on exactly two sentences is a task small/local
    models handle far more reliably. Defaults to False (keep both as
    separate facts) if the call fails or comes back unparseable — the worse
    outcome from a false negative here is one extra stored fact, which is
    far less damaging than wrongly erasing one the user still meant."""
    prompt = (
        "Two statements about the same person, from different times:\n"
        f"Earlier: {existing_fact}\n"
        f"Just now: {new_fact}\n\n"
        "Should \"Just now\" REPLACE \"Earlier\" (same underlying fact — a "
        "preference changed, a detail got more specific, a status changed)? "
        "Or are they two separate facts that can both stay true at the same "
        "time (e.g. two different hobbies, two different routines)?\n"
        "Reply with exactly one word: REPLACE or SEPARATE."
    )
    try:
        response = await llm.query_llm(
            {"model": config.MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=20,
        )
        verdict = response.get("message", {}).get("content", "").strip().upper()
        return verdict.startswith("REPLACE")
    except Exception as e:
        print(f"[MEMORY] Supersession check skipped (non-fatal): {e}")
        return False


def remove_user_fact(user_id: str, identifier: str):
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT id, fact FROM facts WHERE user_id = ? ORDER BY id ASC",
            (str(user_id),),
        ).fetchall()
        if not rows:
            return None

        target_id, target_fact = None, None
        identifier = identifier.strip()
        if identifier.isdigit():
            idx = int(identifier) - 1
            if 0 <= idx < len(rows):
                target_id, target_fact = rows[idx]
        else:
            lowered = identifier.lower()
            # Exact match first, so a short fact that also happens to be a
            # substring of a longer one doesn't get shadowed by it.
            for row_id, fact in rows:
                if fact.lower() == lowered:
                    target_id, target_fact = row_id, fact
                    break
            if target_id is None:
                for row_id, fact in rows:
                    if lowered in fact.lower():
                        target_id, target_fact = row_id, fact
                        break

        if target_id is None:
            return None

        conn.execute("DELETE FROM facts WHERE id = ?", (target_id,))
        conn.commit()
        return target_fact
    finally:
        conn.close()


def _parse_fact_indices(spec: str) -> "list[int] | None":
    """Parses a comma/space-separated list of 1-based !recall positions and
    ranges ('1,3,5', '1 3 5', '2-4', or any mix) into a sorted, de-duplicated
    list of ints. Returns None if `spec` doesn't look like an index list at
    all, so the caller can fall back to single-fact text matching — this
    keeps plain '!forget <text>' working exactly as before."""
    spec = spec.strip()
    if not spec:
        return None
    tokens = [t for t in re.split(r"[,\s]+", spec) if t]
    if not tokens:
        return None
    indices = set()
    for tok in tokens:
        if tok.isdigit():
            indices.add(int(tok))
        elif re.fullmatch(r"\d+-\d+", tok):
            a, b = (int(x) for x in tok.split("-"))
            if a > b:
                a, b = b, a
            indices.update(range(a, b + 1))
        else:
            return None  # contains something that isn't a number or range
    return sorted(i for i in indices if i > 0) or None


def remove_user_facts(user_id: str, indices: list) -> tuple:
    """Removes multiple facts at once by their 1-based !recall position.
    All positions are resolved against a single snapshot of the current
    list before anything is deleted, so removing e.g. both 1 and 3 in the
    same call is safe — an earlier deletion never shifts what a later index
    refers to. Returns (removed_texts, invalid_positions); invalid_positions
    covers anything out of range so the caller can report it back."""
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT id, fact FROM facts WHERE user_id = ? ORDER BY id ASC",
            (str(user_id),),
        ).fetchall()

        removed_texts, invalid, to_delete_ids = [], [], []
        for idx in indices:
            pos = idx - 1
            if 0 <= pos < len(rows):
                row_id, fact = rows[pos]
                to_delete_ids.append(row_id)
                removed_texts.append(fact)
            else:
                invalid.append(idx)

        if to_delete_ids:
            conn.executemany("DELETE FROM facts WHERE id = ?", [(i,) for i in to_delete_ids])
            conn.commit()
        return removed_texts, invalid
    finally:
        conn.close()


def clear_user_facts(user_id: str):
    conn = _get_memory_conn()
    try:
        conn.execute("DELETE FROM facts WHERE user_id = ?", (str(user_id),))
        conn.commit()
    finally:
        conn.close()


def _cluster_facts_by_topic(facts_with_emb: list) -> list:
    """Groups facts into topic clusters via union-find over pairwise cosine
    similarity at CONSOLIDATION_TOPIC_THRESHOLD. Same same-space-only
    reasoning as _is_semantic_duplicate/_find_supersession_candidate — a
    fact with no embedding, or in a different space than its neighbors,
    just ends up alone in its own singleton cluster rather than being
    force-grouped or dropped."""
    n = len(facts_with_emb)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        emb_i = facts_with_emb[i][1]
        if not emb_i:
            continue
        space_i, vec_i = emb_i
        for j in range(i + 1, n):
            emb_j = facts_with_emb[j][1]
            if not emb_j or emb_j[0] != space_i:
                continue
            if embeddings.cosine_similarity(vec_i, emb_j[1]) >= CONSOLIDATION_TOPIC_THRESHOLD:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(facts_with_emb[i])
    return list(groups.values())


async def _llm_consolidate_cluster(facts: list) -> "str | None":
    """Merges a cluster of topically-related facts into one dense entry.
    This is real (lossy) summarization, not just compaction: the model is
    told to keep only what's most important/distinctive and may drop minor
    or lower-value specifics to actually shrink token footprint, not just
    row count — the explicit tradeoff is fewer tokens in the prompt when
    this entry gets injected, at the cost of losing the least-important
    detail from the cluster. Returns None (leave the cluster alone) if the
    call fails or comes back empty, same fail-safe posture as
    _llm_decides_replace: a missed consolidation just means a few extra
    rows survive, which is far less damaging than a bad/empty summary
    silently replacing real facts."""
    facts_block = "\n".join(f"- {f}" for f in facts)
    prompt = (
        "These are separate memory entries stored about the same user, all "
        "about the same underlying topic (likely captured one at a time "
        "from one longer conversation or document). Summarize them into "
        "ONE consolidated entry that is noticeably shorter than the "
        "combined originals — a sentence or two, not a full paragraph. "
        "Keep the most important and distinctive specifics (names, "
        "numbers, key decisions); it's fine to drop minor or redundant "
        "details to keep it short. Do not add anything that isn't already "
        "stated below.\n\n"
        f"{facts_block}\n\n"
        "Reply with ONLY the summarized entry, no preamble, no markdown."
    )
    try:
        response = await llm.query_llm(
            {"model": config.MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=60,
        )
        merged = response.get("message", {}).get("content", "").strip()
        return merged or None
    except Exception as e:
        print(f"[MEMORY] Consolidation skipped for one cluster (non-fatal): {e}")
        return None


def _eligible_clusters_for_consolidation(facts_with_emb: list) -> list:
    """Shared selection logic for both consolidate_user_facts and
    preview_consolidation: cluster by topic, keep clusters big enough to be
    worth merging, largest first, capped at
    CONSOLIDATION_MAX_CLUSTERS_PER_PASS — so a preview always shows exactly
    what a real run would touch, and vice versa."""
    clusters = _cluster_facts_by_topic(facts_with_emb)
    eligible = [c for c in clusters if len(c) >= CONSOLIDATION_MIN_CLUSTER_SIZE]
    # Largest clusters first — they free up the most rows per LLM call,
    # which matters most since only the top N get processed this pass.
    eligible.sort(key=len, reverse=True)
    return eligible[:CONSOLIDATION_MAX_CLUSTERS_PER_PASS]


async def preview_consolidation(user_id: str, force: bool = False) -> "list[tuple[list[str], str | None]] | None":
    """Dry-run counterpart to consolidate_user_facts — same clustering and
    selection, and the same LLM merge call per cluster, but nothing is ever
    written to the database. Powers `!consolidate preview` so a user can see
    what a real `!consolidate` would do before committing to it.

    Returns a list of (original_fact_texts, merged_text_or_None) tuples, one
    per eligible cluster (merged is None if that cluster's LLM call failed —
    the real run would leave that cluster untouched too), or None if there's
    nothing eligible to show (mirrors consolidate_user_facts's None return)."""
    facts_with_emb = get_user_facts_with_embeddings(user_id)
    if not force and len(facts_with_emb) < CONSOLIDATION_TRIGGER_COUNT:
        return None

    eligible = _eligible_clusters_for_consolidation(facts_with_emb)
    if not eligible:
        return None

    results = []
    for cluster in eligible:
        texts = [f for f, _ in cluster]
        merged = await _llm_consolidate_cluster(texts)
        results.append((texts, merged))

    return results


async def consolidate_user_facts(user_id: str, force: bool = False) -> "tuple[int, int] | None":
    """Sweeps one user's facts for topic clusters and summarizes up to
    CONSOLIDATION_MAX_CLUSTERS_PER_PASS of the largest clusters (each
    CONSOLIDATION_MIN_CLUSTER_SIZE+ facts) into single, shorter entries.
    This is lossy — minor/redundant details from the cluster may not
    survive — trading detail for a real reduction in both row count and
    prompt-injection tokens. Only runs once the user's fact count reaches
    CONSOLIDATION_TRIGGER_COUNT, unless force=True (used by the manual
    !consolidate command to bypass that gate and sweep regardless of how
    many facts are currently stored).
    Returns (count_before, count_after) if anything was merged, else None."""
    facts_with_emb = get_user_facts_with_embeddings(user_id)
    before = len(facts_with_emb)
    if not force and before < CONSOLIDATION_TRIGGER_COUNT:
        return None

    eligible = _eligible_clusters_for_consolidation(facts_with_emb)
    merged_any = False
    for cluster in eligible:
        texts = [f for f, _ in cluster]
        merged = await _llm_consolidate_cluster(texts)
        if not merged:
            continue  # leave this cluster untouched rather than risk losing detail
        embedding = await _embed_fact(merged)
        conn = _get_memory_conn()
        try:
            placeholders = ",".join("?" * len(texts))
            conn.execute(
                f"DELETE FROM facts WHERE user_id = ? AND fact IN ({placeholders})",
                (str(user_id), *texts),
            )
            conn.commit()
        finally:
            conn.close()
        add_user_fact(user_id, merged, embedding)
        merged_any = True

    if not merged_any:
        return None
    return before, len(get_user_facts(user_id))


async def get_relevant_facts_block(user_id: str, query: str) -> str:
    """Builds the "What you remember about this user" block for the system
    prompt. Small fact lists are sent whole; larger ones are trimmed to the
    FACT_RELEVANCE_TOP_K facts most relevant to the current message, so the
    prompt doesn't grow unbounded as a user's fact count climbs toward
    MAX_FACTS_PER_USER."""
    facts_with_emb = get_user_facts_with_embeddings(user_id)
    if not facts_with_emb:
        return ""

    if len(facts_with_emb) <= FACT_INJECT_ALWAYS_UNDER:
        chosen = [f for f, _ in facts_with_emb]
    else:
        query_emb = await _embed_fact(query)
        if query_emb is None:
            # Can't rank without a query embedding — most-recent facts are a
            # safer bet than an arbitrary truncation.
            chosen = [f for f, _ in facts_with_emb[-FACT_RELEVANCE_TOP_K:]]
        else:
            q_space, q_vector = query_emb
            scored = []
            for fact, emb in facts_with_emb:
                if emb and emb[0] == q_space:
                    score = embeddings.cosine_similarity(q_vector, emb[1])
                else:
                    score = 1.0  # no comparable embedding — never silently hide it
                scored.append((score, fact))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            chosen = [fact for _, fact in scored[:FACT_RELEVANCE_TOP_K]]

    return "\n\nWhat you remember about this user:\n" + "\n".join(f"- {f}" for f in chosen)


async def extract_and_store_facts(user_id: str, user_query: str, channel=None):
    # Deferred import: messaging.py needs get_channel_history/
    # _append_conversation_log from this module at top level, so this module
    # can't import messaging at top level too without a cycle. By the time
    # this function actually runs, messaging.py is already fully loaded.
    import messaging

    if _looks_low_signal(user_query):
        return

    existing_facts = get_user_facts(user_id)
    existing_block = "\n".join(f"- {f}" for f in existing_facts) if existing_facts else "(none yet)"

    extraction_prompt = (
        "Below is a single message a user sent to a Discord bot, plus the facts "
        "already remembered about this user. Decide whether the message contains "
        "any NEW durable fact worth remembering long-term (name, role, "
        "preferences, ongoing projects, recurring routines, etc), or whether it "
        "REVISES/contradicts one of the existing facts (e.g. moved cities, "
        "changed jobs, switched a preference). Ignore one-off requests, "
        "questions, or temporary details.\n\n"
        "Reply with ONLY a JSON object of this exact shape (no markdown, no "
        "preamble):\n"
        '{"add": ["new fact 1", ...], "update": [{"replaces": "<verbatim existing '
        'fact text>", "with": "<revised fact text>"}]}\n'
        "Only use \"update\" when \"replaces\" is copied verbatim from the "
        "existing facts list below — never paraphrase it. If nothing applies, "
        'reply with exactly {"add": [], "update": []}\n\n'
        f"Existing facts:\n{existing_block}\n\nMessage: {user_query}"
    )
    payload = {
        "model": config.MODEL_NAME,
        "messages": [{"role": "user", "content": extraction_prompt}],
        "stream": False
    }
    try:
        response = await llm.query_llm(payload, timeout=60)
        raw = response.get("message", {}).get("content", "{}").strip()
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return
        to_add = [str(f).strip() for f in parsed.get("add", []) if str(f).strip()]
        to_update = [
            u for u in parsed.get("update", [])
            if isinstance(u, dict) and str(u.get("replaces", "")).strip() and str(u.get("with", "")).strip()
        ]
    except Exception as e:
        print(f"[MEMORY] Extraction skipped (non-fatal): {e}")
        return

    updated_summaries = []
    for u in to_update:
        new_text = str(u["with"]).strip()
        embedding = await _embed_fact(new_text)
        result = update_user_fact(user_id, str(u["replaces"]), new_text, embedding)
        if result:
            updated_summaries.append(f"{result[0]} → {result[1]}")
        else:
            # "replaces" didn't match anything on file — treat as a fresh
            # fact rather than silently dropping it.
            to_add.append(new_text)

    added = []
    consolidation_summaries = []
    consolidation_exhausted = False
    if to_add:
        existing_for_dedup = get_user_facts_with_embeddings(user_id)
        for fact in to_add:
            if not fact:
                continue
            embedding = await _embed_fact(fact)
            if _is_semantic_duplicate(embedding, existing_for_dedup):
                continue

            candidate = _find_supersession_candidate(embedding, existing_for_dedup)
            if candidate is not None:
                old_text, _score = candidate
                if await _llm_decides_replace(old_text, fact):
                    result = update_user_fact(user_id, old_text, fact, embedding)
                    if result:
                        updated_summaries.append(f"{result[0]} → {result[1]}")
                        existing_for_dedup = [
                            (f, e) for f, e in existing_for_dedup if f != old_text
                        ]
                        existing_for_dedup.append((fact, embedding))
                    continue

            # Checked per-fact, not just once before the loop — a single
            # message can extract many new facts at once (a big paste), and
            # a single up-front check would let that whole batch sail past
            # the cap before the sweep got a second look. consolidation_exhausted
            # stops it from retrying every remaining iteration once a pass
            # has genuinely found nothing left to merge — nothing changed,
            # so an immediate retry can't help; add_user_fact's own trim
            # remains the safety net for whatever's left in that case.
            if not consolidation_exhausted and len(existing_for_dedup) >= CONSOLIDATION_TRIGGER_COUNT:
                result = await consolidate_user_facts(user_id)
                if result:
                    before, after = result
                    consolidation_summaries.append(f"{before} → {after} facts")
                    existing_for_dedup = get_user_facts_with_embeddings(user_id)
                else:
                    consolidation_exhausted = True

            if add_user_fact(user_id, fact, embedding):
                added.append(fact)
                existing_for_dedup.append((fact, embedding))

    consolidation_summary = "; ".join(consolidation_summaries) if consolidation_summaries else None

    if channel is not None and (added or updated_summaries or consolidation_summary):
        lines = [f"-# 🧠 remembered: {f}" for f in added]
        lines += [f"-# 🧠 updated: {s}" for s in updated_summaries]
        if consolidation_summary:
            lines.append(f"-# 🧠 consolidated: {consolidation_summary}")
        await messaging.send_chunked(channel, "\n".join(lines))
