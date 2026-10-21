"""
memory_store.py — conversation history (per-channel rolling window +
persistent log) and long-term per-user facts, now organized into topics
(SQLite-backed), including dedup, supersession, per-topic consolidation,
and one-time migration of pre-topic facts.

Both conversation_log and facts/topics go through the SAME SQLite
connection (_get_memory_conn) and the same database file — this module owns
that connection/schema; nothing else should open MEMORY_DB_PATH directly.

extract_and_store_facts() needs messaging.send_chunked to post "remembered:"
notices, but messaging.py needs get_channel_history/_append_conversation_log
from THIS module — that's a real two-way dependency. It's broken by
importing messaging lazily, inside the one function that needs it, rather
than at module load time (see the `import messaging` inside
extract_and_store_facts below). messaging.py's top-level `import
memory_store` stays a normal import; this is the module that defers.

TOPIC MODEL: every fact belongs to exactly one topic (a short label like
"Rutrinda worldbuilding", or the standing "General" bucket for genuine
one-offs). New facts get a topic proposed by the extraction LLM call, then
canonicalized against the user's existing topics (exact match, then
embedding similarity) so slightly different wording doesn't spawn
near-duplicate topics. Cross-topic fact text stays globally unique per user
(facts.UNIQUE(user_id, fact) has no topic scoping) — this also means
merge_topics() can safely reassign topic_id without ever risking a
collision on fact text.

Nothing here relies on SQLite's ON DELETE CASCADE (whose enforcement
depends on a per-connection PRAGMA this codebase doesn't set) — anywhere a
topic is deleted, its facts are moved or deleted explicitly first.
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


# --- PERSISTENT USER MEMORY (SQLite-backed, topic-organized) ---
# Was a flat memory_store.json capped at 40 facts/user. Moved to SQLite
# (same pattern as tools/rag_knowledge.py's vector store) so raising the cap
# is just a number, not a rewrite, and per-user lookups don't require
# loading every other user's facts into memory first.
MEMORY_DB_PATH = config.DATA_DIR / "memory_store.sqlite3"
# The legacy pre-SQLite file is NOT part of the new data/ layout — it's only
# ever read once, by the migration below, and deliberately left wherever it
# already sits (see migrate_data_layout.py, which does not move it either).
LEGACY_MEMORY_FILE = config.DATA_DIR.parent / "memory_store.json"
MAX_FACTS_PER_USER = 200  # global safety cap across all of a user's topics combined

# Every fact belongs to a topic. Genuine one-offs with no ongoing subject
# land here rather than spawning a single-fact topic of their own.
DEFAULT_TOPIC_LABEL = "General"
# Standing topic for durable facts about the user's own identity,
# personality, habits, and preferences — as opposed to facts that belong to
# a specific ongoing project, world, or subject (which get their own named
# topic instead). Without this as an explicit, always-available option,
# personal facts tend to get glued onto whatever project topic happens to
# be active in the conversation at the time, or scattered into General.
USER_TOPIC_LABEL = "About You"
# Two topic labels whose embeddings score at or above this are treated as
# the same underlying topic — this is what lets "Rutrinda" and "The world
# of Rutrinda" resolve to one topic instead of two, even when the
# extraction model doesn't reuse the existing label verbatim. Looser than
# fact-level dedup since topic labels are short and paraphrase more freely
# than full sentences.
TOPIC_MATCH_THRESHOLD = 0.75

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
# Dedup/supersession stay CROSS-TOPIC deliberately — a fact restated while
# talking about a different topic should still be recognized as the same
# fact, not silently duplicated because it landed in a different bucket.
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
# orbiting one subject — used both for sub-clustering WITHIN a topic during
# consolidation, and for the one-time migration that groups pre-topic facts
# into topics in the first place.
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
# Per-topic reactive trigger — checked against a single topic's OWN fact
# count right after a new fact is filed under it, not the user's total
# across all topics. This is what actually solves topic fragmentation
# during a single session: a topic like "Rutrinda worldbuilding" gets
# tidied on its own once it individually grows large, instead of waiting
# for the user's entire fact count to approach MAX_FACTS_PER_USER.
CONSOLIDATION_TOPIC_TRIGGER_COUNT = 20
# Migration-time clustering uses a lower bar than consolidation's
# CONSOLIDATION_MIN_CLUSTER_SIZE — this is about INITIAL categorization
# (does this fact belong with others, or is it a one-off for General), not
# about whether a cluster is big enough to be worth an LLM summarization
# call. Two related pre-topic facts already deserve their own topic even if
# they're not yet big enough to warrant condensing.
MIGRATION_MIN_CLUSTER_SIZE = 2

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
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            label TEXT NOT NULL,
            summary TEXT,
            embedding TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, label COLLATE NOCASE)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_topics_user ON topics(user_id)")
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_models (
            channel_id TEXT PRIMARY KEY,
            model TEXT NOT NULL
        )
        """
    )
    # Additive migration for DBs created before these columns existed.
    # ADD COLUMN has no "IF NOT EXISTS" in SQLite, so this just no-ops with
    # an OperationalError ("duplicate column") on every run after the first.
    for ddl in (
        "ALTER TABLE facts ADD COLUMN updated_at TEXT",
        "ALTER TABLE facts ADD COLUMN embedding TEXT",
        # Nullable, NOT "NOT NULL" — existing rows predate topics and start
        # out unassigned; migrate_facts_to_topics() backfills them at
        # startup. Every NEW fact from add_user_fact() onward always
        # receives a real topic_id, so in practice this is non-null for
        # anything written going forward, even though the column itself
        # can't enforce that retroactively on rows that already existed.
        "ALTER TABLE facts ADD COLUMN topic_id INTEGER REFERENCES topics(id)",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_facts_topic ON facts(topic_id)")
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


# --- per-thread model override (set manually with !model, thread-only) ---
# Only ever consulted for discord.Thread channels. The main channel has no
# row here and always resolves to config.MODEL_NAME. New threads (including
# ones created by maybe_suggest_thread) never inherit a row — an override is
# created only by an explicit !model in that thread.

def get_thread_model_override(channel_id) -> "str | None":
    conn = _get_memory_conn()
    try:
        row = conn.execute(
            "SELECT model FROM thread_models WHERE channel_id = ?", (str(channel_id),)
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def set_thread_model_override(channel_id, model: str):
    conn = _get_memory_conn()
    try:
        conn.execute(
            "INSERT INTO thread_models (channel_id, model) VALUES (?, ?) "
            "ON CONFLICT(channel_id) DO UPDATE SET model = excluded.model",
            (str(channel_id), model),
        )
        conn.commit()
    finally:
        conn.close()


def clear_thread_model_override(channel_id) -> bool:
    """Returns True if there was an override to remove."""
    conn = _get_memory_conn()
    try:
        cur = conn.execute("DELETE FROM thread_models WHERE channel_id = ?", (str(channel_id),))
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def get_channel_model(channel) -> str:
    """The chat model to use for this channel: a thread's !model override if it
    has one, otherwise config.MODEL_NAME (always the case for the main channel)."""
    if isinstance(channel, discord.Thread):
        override = get_thread_model_override(channel.id)
        if override:
            return override
    return config.MODEL_NAME


# --- embedding helpers (shared by facts AND topic labels) ---

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


async def _embed_text(text: str) -> "tuple[str, list[float]] | None":
    """Tiered embedding for any piece of text — a fact, a query string, or a
    topic label. Same fallback order as embeddings.select_relevant_tools():
    Gemini primary, local nomic only when chat is currently on the local
    backend. Returns (space, vector) rather than a bare vector so callers
    only ever compare vectors embedded in the same space — Gemini and local
    vectors are never comparable to each other (see the note on
    embeddings.TOOL_EMBEDDINGS_LOCAL)."""
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
    (renamed to .json.migrated on success, so this only ever runs once).
    Facts land with topic_id NULL, same as any other pre-topic fact —
    migrate_facts_to_topics() (run at startup, see bot.py's on_ready) picks
    them up from there."""
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


# --- topics ---

def _get_user_topics_with_embeddings(user_id: str) -> list:
    """[(topic_id, label, (space, vector) | None), ...]"""
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT id, label, embedding FROM topics WHERE user_id = ?",
            (str(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return [(tid, label, _parse_embedding(raw)) for tid, label, raw in rows]


async def get_or_create_topic(user_id: str, proposed_label: str) -> int:
    """Resolves a proposed topic label to an existing topic — exact
    case-insensitive match first (cheap, no embedding needed), then
    embedding similarity >= TOPIC_MATCH_THRESHOLD — or creates a new one.
    This is what keeps "Rutrinda" and "The world of Rutrinda" from becoming
    two separate topics just because the extraction model phrased its
    proposal differently across calls."""
    proposed_label = proposed_label.strip() or DEFAULT_TOPIC_LABEL

    conn = _get_memory_conn()
    try:
        row = conn.execute(
            "SELECT id FROM topics WHERE user_id = ? AND lower(label) = lower(?)",
            (str(user_id), proposed_label),
        ).fetchone()
        if row:
            return row[0]
    finally:
        conn.close()

    existing_topics = _get_user_topics_with_embeddings(user_id)
    embedding = await _embed_text(proposed_label)
    if embedding is not None and existing_topics:
        space, vector = embedding
        best_id, best_score = None, 0.0
        for tid, _label, existing_emb in existing_topics:
            if not existing_emb or existing_emb[0] != space:
                continue
            score = embeddings.cosine_similarity(vector, existing_emb[1])
            if score > best_score:
                best_id, best_score = tid, score
        if best_id is not None and best_score >= TOPIC_MATCH_THRESHOLD:
            return best_id

    conn = _get_memory_conn()
    try:
        now = datetime.now(timezone.utc).isoformat()
        try:
            cur = conn.execute(
                "INSERT INTO topics (user_id, label, embedding, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (str(user_id), proposed_label, _encode_embedding(embedding), now, now),
            )
            conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            # Another concurrent message for this user created this exact
            # label between our check above and this insert — use theirs.
            row = conn.execute(
                "SELECT id FROM topics WHERE user_id = ? AND lower(label) = lower(?)",
                (str(user_id), proposed_label),
            ).fetchone()
            return row[0] if row else None
    finally:
        conn.close()


async def get_default_topic_id(user_id: str) -> int:
    return await get_or_create_topic(user_id, DEFAULT_TOPIC_LABEL)


async def get_user_topic_id(user_id: str) -> int:
    return await get_or_create_topic(user_id, USER_TOPIC_LABEL)


def get_user_topics(user_id: str) -> list:
    """Returns [(label, fact_count), ...] for !topics — regular topics in
    creation order, then "About You", then "General" always last."""
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            """
            SELECT t.label, COUNT(f.id) as cnt
            FROM topics t LEFT JOIN facts f ON f.topic_id = t.id
            WHERE t.user_id = ?
            GROUP BY t.id
            ORDER BY
                CASE
                    WHEN lower(t.label) = lower(?) THEN 2
                    WHEN lower(t.label) = lower(?) THEN 1
                    ELSE 0
                END ASC,
                t.id ASC
            """,
            (str(user_id), DEFAULT_TOPIC_LABEL, USER_TOPIC_LABEL),
        ).fetchall()
    finally:
        conn.close()
    return rows


def rename_topic_sync(user_id: str, old_label: str, new_label: str) -> "tuple[str, str] | None":
    """Case-insensitive match on old_label; renames in place (same row,
    same facts, same topic_id — nothing about the linked facts changes).
    Returns (matched_old_label, new_label) on success, None if old_label
    didn't match anything, or if new_label collides with a DIFFERENT
    existing topic (merge_topics is the tool for combining two topics, not
    a rename that happens to collide)."""
    new_label = new_label.strip()
    if not new_label:
        return None
    conn = _get_memory_conn()
    try:
        row = conn.execute(
            "SELECT id, label FROM topics WHERE user_id = ? AND lower(label) = lower(?)",
            (str(user_id), old_label.strip()),
        ).fetchone()
        if not row:
            return None
        topic_id, matched_label = row

        collision = conn.execute(
            "SELECT id FROM topics WHERE user_id = ? AND lower(label) = lower(?) AND id != ?",
            (str(user_id), new_label, topic_id),
        ).fetchone()
        if collision:
            return None

        conn.execute(
            "UPDATE topics SET label = ?, updated_at = ? WHERE id = ?",
            (new_label, datetime.now(timezone.utc).isoformat(), topic_id),
        )
        conn.commit()
        return matched_label, new_label
    finally:
        conn.close()


async def rename_topic(user_id: str, old_label: str, new_label: str) -> "tuple[str, str] | None":
    """Same as rename_topic_sync, but also re-embeds the new label so future
    get_or_create_topic() calls canonicalize against the renamed text, not
    the stale one."""
    result = rename_topic_sync(user_id, old_label, new_label)
    if result is None:
        return None
    embedding = await _embed_text(new_label)
    if embedding is not None:
        conn = _get_memory_conn()
        try:
            conn.execute(
                "UPDATE topics SET embedding = ? WHERE user_id = ? AND lower(label) = lower(?)",
                (_encode_embedding(embedding), str(user_id), new_label),
            )
            conn.commit()
        finally:
            conn.close()
    return result


def merge_topics(user_id: str, keep_label: str, remove_label: str) -> "tuple[str, str, int] | None":
    """Moves every fact from remove_label's topic into keep_label's, then
    deletes the now-empty remove_label topic row. Safe against a fact-text
    collision — facts.UNIQUE(user_id, fact) is global per user, not scoped
    per topic, so the same fact text can never already exist in both. Moving
    is just an UPDATE of topic_id, never touches fact text.
    Returns (matched_keep_label, matched_remove_label, moved_count) on
    success, or None if either label doesn't match, or both resolve to the
    same topic."""
    conn = _get_memory_conn()
    try:
        keep_row = conn.execute(
            "SELECT id, label FROM topics WHERE user_id = ? AND lower(label) = lower(?)",
            (str(user_id), keep_label.strip()),
        ).fetchone()
        remove_row = conn.execute(
            "SELECT id, label FROM topics WHERE user_id = ? AND lower(label) = lower(?)",
            (str(user_id), remove_label.strip()),
        ).fetchone()
        if not keep_row or not remove_row:
            return None
        keep_id, keep_matched = keep_row
        remove_id, remove_matched = remove_row
        if keep_id == remove_id:
            return None

        moved = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE user_id = ? AND topic_id = ?",
            (str(user_id), remove_id),
        ).fetchone()[0]
        conn.execute(
            "UPDATE facts SET topic_id = ? WHERE user_id = ? AND topic_id = ?",
            (keep_id, str(user_id), remove_id),
        )
        conn.execute("DELETE FROM topics WHERE id = ?", (remove_id,))
        conn.commit()
        return keep_matched, remove_matched, moved
    finally:
        conn.close()


# Facts per LLM call when reclassifying a topic — bounds prompt size and
# limits how much one bad/truncated response can affect at once, same
# reasoning as CONSOLIDATION_MAX_CLUSTERS_PER_PASS.
RECLASSIFY_BATCH_SIZE = 40


async def _llm_propose_topics_for_facts(facts: list, existing_topic_labels: list) -> "list[str] | None":
    """Batch LLM call: given up to RECLASSIFY_BATCH_SIZE facts, proposes a
    topic label PER fact — the same per-fact judgment extract_and_store_facts
    uses for a fact as it arrives, just applied retroactively to a batch of
    already-stored facts in one call instead of one message at a time.

    This exists because embedding-similarity clustering (used by the
    one-time migrate_facts_to_topics()) can under-connect a large,
    thematically diffuse set of facts that a reader would recognize as
    clearly related but whose individual embeddings don't score close
    enough pairwise — a worldbuilding project with many distinct named
    entities (nations, mottos, technology, history) is exactly this case.
    An LLM reading the fact directly doesn't have that failure mode.

    Returns a list of labels, same length/order as `facts`, or None if the
    call fails or the response can't be parsed into exactly that many
    labels — caller leaves that batch untouched rather than guess."""
    facts_block = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts))
    topics_block = ", ".join(existing_topic_labels) if existing_topic_labels else "(none yet)"
    prompt = (
        "Below is a numbered list of facts stored about a user. For EACH "
        "fact, propose a short topic label (2-5 words) naming the ongoing "
        "subject it belongs to. Decide which of these three cases applies:\n"
        "1. The fact is about the user's OWN identity, personality, "
        "habits, tools they use, or preferences that hold true regardless "
        "of which project or topic they're discussing (who they are, not "
        'what they\'re working on) — use exactly "About You" for these.\n'
        "2. The fact is part of a specific ongoing project, world, or "
        "subject — reuse an existing topic label below if it clearly "
        "continues that subject. Only propose a new label for a genuinely "
        "different ongoing subject — and reuse the SAME new label across "
        "facts that share that subject, rather than inventing a slightly "
        "different wording for each one. This includes an opinion, "
        "struggle, or preference that's SPECIFIC to that project (e.g. "
        '"finds writing a particular character difficult", "wants the '
        'story to end differently") — these stay with the project\'s '
        "topic even though they're phrased as being about the user. "
        '"About You" is only for things true about the user independent '
        "of which project is being discussed.\n"
        "3. A true one-off with no lasting category and not about the "
        'user\'s own identity — use exactly "General".\n\n'
        f"Existing topics: {topics_block}\n\n"
        f"Facts:\n{facts_block}\n\n"
        "Reply with ONLY a JSON array of strings, one topic label per fact, "
        "in the same order and the same length as the numbered list above. "
        'Example: ["About You", "Label A", "General"]'
    )
    try:
        response = await llm.query_llm(
            {"model": config.MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=60,
        )
        raw = response.get("message", {}).get("content", "[]").strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)
        if isinstance(parsed, list) and len(parsed) == len(facts):
            return [str(label).strip() or DEFAULT_TOPIC_LABEL for label in parsed]
        return None
    except Exception as e:
        print(f"[MEMORY] Reclassification batch skipped (non-fatal): {e}")
        return None


async def reclassify_topic(user_id: str, topic_label: str) -> "tuple[int, int] | None":
    """Manual remediation for a topic — typically General — that ended up
    too broad. Re-runs classification on every fact currently filed under
    `topic_label` using _llm_propose_topics_for_facts (LLM judgment) instead
    of migration's embedding-distance clustering, and moves each fact to
    whatever topic the LLM proposes (via the same get_or_create_topic
    canonicalization new facts already go through). The topic being
    reclassified is deliberately excluded from the "existing topics" list
    shown to the LLM, so it's not just handed its own label as the easy
    answer — "General" is still available as a genuine fallback, but only
    via the prompt's explicit instruction, not because it's sitting in the
    reuse-this-list.
    Returns (total_facts_in_topic, facts_moved_to_a_different_topic) on
    success, or None if topic_label doesn't match an existing topic or it
    currently has no facts."""
    conn = _get_memory_conn()
    try:
        topic_row = conn.execute(
            "SELECT id FROM topics WHERE user_id = ? AND lower(label) = lower(?)",
            (str(user_id), topic_label.strip()),
        ).fetchone()
    finally:
        conn.close()
    if not topic_row:
        return None
    source_topic_id = topic_row[0]

    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT id, fact FROM facts WHERE user_id = ? AND topic_id = ? ORDER BY id ASC",
            (str(user_id), source_topic_id),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None

    moved = 0
    for start in range(0, len(rows), RECLASSIFY_BATCH_SIZE):
        batch = rows[start:start + RECLASSIFY_BATCH_SIZE]
        existing_labels = [
            label for label, _c in get_user_topics(user_id)
            if label.lower() != topic_label.strip().lower()
        ]
        proposed = await _llm_propose_topics_for_facts([fact for _fid, fact in batch], existing_labels)
        if proposed is None:
            continue  # leave this batch untouched rather than guess

        for (fact_id, _fact_text), new_label in zip(batch, proposed):
            new_topic_id = await get_or_create_topic(user_id, new_label)
            if new_topic_id != source_topic_id:
                conn = _get_memory_conn()
                try:
                    conn.execute("UPDATE facts SET topic_id = ? WHERE id = ?", (new_topic_id, fact_id))
                    conn.commit()
                finally:
                    conn.close()
                moved += 1

    return len(rows), moved


async def move_fact_to_topic(user_id: str, position: int, topic_label: str) -> "tuple[str, str] | None":
    """Manual override for a single fact — moves the fact at !recall
    position `position` (1-based, same continuous cross-topic ordering as
    get_user_facts_ordered_with_topics/remove_user_facts) to `topic_label`
    (an existing topic reused via the same canonicalization new facts
    already go through, or a brand new one). This exists because LLM
    classification — however good the prompt — will occasionally get a
    genuinely ambiguous case wrong (a project-specific opinion phrased as
    being about the user, for instance), and the person who actually knows
    where a fact belongs should always be able to just say so directly
    rather than being stuck re-running reclassify and hoping it lands
    differently.
    Returns (fact_text, resolved_topic_label) on success — resolved_topic_label
    is what the fact actually got filed under, which may differ from
    `topic_label` if it canonicalized to an existing topic with slightly
    different wording. Returns None if `position` is out of range."""
    rows = get_user_facts_ordered_with_topics(user_id)
    pos = position - 1
    if not (0 <= pos < len(rows)):
        return None
    fact_id, _old_topic_label, fact_text = rows[pos]

    topic_id = await get_or_create_topic(user_id, topic_label)
    conn = _get_memory_conn()
    try:
        conn.execute("UPDATE facts SET topic_id = ? WHERE id = ?", (topic_id, fact_id))
        conn.commit()
        resolved_row = conn.execute("SELECT label FROM topics WHERE id = ?", (topic_id,)).fetchone()
    finally:
        conn.close()

    resolved_label = resolved_row[0] if resolved_row else topic_label
    return fact_text, resolved_label


# --- facts ---

def get_user_facts(user_id: str) -> list:
    """Flat, cross-topic list of fact text only — used where topic grouping
    doesn't matter (the extraction prompt's existing-facts block for
    add/update decisions, which needs to see everything regardless of
    which topic it's filed under)."""
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
    """Flat, cross-topic [(fact, (space, vector) | None), ...] — used for
    dedup/supersession, which are deliberately cross-topic (see the module
    docstring). Facts stored before embeddings existed, or whose embedding
    call failed at the time, come back with None; callers treat those as
    always-relevant/always-distinct rather than hiding them, same
    philosophy as embeddings.select_relevant_tools() for tools with no
    cached embedding."""
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT fact, embedding FROM facts WHERE user_id = ? ORDER BY id ASC",
            (str(user_id),),
        ).fetchall()
    finally:
        conn.close()
    return [(fact, _parse_embedding(raw)) for fact, raw in rows]


def get_user_facts_ordered_with_topics(user_id: str) -> list:
    """Returns [(fact_id, topic_label, fact_text), ...] in the SAME stable
    order !recall displays and !forget's numeric positions refer to —
    regular topics in creation order, then "About You", then "General"
    always last, facts within a topic in creation order. Both !recall's
    display and remove_user_facts() read this exact ordering so a position
    never means two different things between what's shown and what gets
    removed."""
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            """
            SELECT f.id, t.label, f.fact
            FROM facts f JOIN topics t ON f.topic_id = t.id
            WHERE f.user_id = ?
            ORDER BY
                CASE
                    WHEN lower(t.label) = lower(?) THEN 2
                    WHEN lower(t.label) = lower(?) THEN 1
                    ELSE 0
                END ASC,
                t.id ASC, f.id ASC
            """,
            (str(user_id), DEFAULT_TOPIC_LABEL, USER_TOPIC_LABEL),
        ).fetchall()
    finally:
        conn.close()
    return rows


def _get_topic_facts_with_embeddings(user_id: str, topic_id: int) -> list:
    conn = _get_memory_conn()
    try:
        rows = conn.execute(
            "SELECT fact, embedding FROM facts WHERE user_id = ? AND topic_id = ? ORDER BY id ASC",
            (str(user_id), topic_id),
        ).fetchall()
    finally:
        conn.close()
    return [(fact, _parse_embedding(raw)) for fact, raw in rows]


def _count_topic_facts(user_id: str, topic_id: int) -> int:
    conn = _get_memory_conn()
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM facts WHERE user_id = ? AND topic_id = ?",
            (str(user_id), topic_id),
        ).fetchone()[0]
    finally:
        conn.close()


def add_user_fact(
    user_id: str, fact: str, topic_id: int, embedding: "tuple[str, list[float]] | None" = None
) -> bool:
    """Inserts a single fact under the given topic. Returns False if it
    already exists verbatim for this user (UNIQUE constraint — global per
    user, not scoped by topic) or the fact is blank — semantic near-dupes
    should be filtered by the caller with _is_semantic_duplicate before this
    is ever reached."""
    fact = fact.strip()
    if not fact:
        return False
    conn = _get_memory_conn()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cur = conn.execute(
            "INSERT OR IGNORE INTO facts (user_id, topic_id, fact, created_at, updated_at, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(user_id), topic_id, fact, now, now, _encode_embedding(embedding)),
        )
        inserted = bool(cur.rowcount)
        conn.commit()

        if inserted:
            # Enforce the per-user cap (across ALL topics combined) by
            # trimming the oldest rows beyond it, same "keep the newest N"
            # behavior the old JSON store had. This is a blunt last-resort
            # safety valve — per-topic consolidation (see
            # CONSOLIDATION_TOPIC_TRIGGER_COUNT) is what's meant to keep
            # things tidy well before this ever has to trim anything.
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
    and embedding in place, preserving its row (and therefore its existing
    topic_id — a revised fact stays under whatever topic it was already
    filed in) rather than appending. This is what lets a revised fact
    ("moved to Manila") supersede the old one ("lives in Calumpit") instead
    of both persisting side by side forever. Returns (old_text, new_text) on
    success, or None if old_fact wasn't found (the caller should then treat
    it as a plain add)."""
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
    _embed_text); if either side has no embedding this returns False rather
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
    """Text-based /!forget <text> lookup — stays cross-topic, unaffected by
    grouping (a fact is found by its own text/position regardless of which
    topic it's filed under)."""
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
    """Removes multiple facts at once by their 1-based !recall position —
    positions refer to get_user_facts_ordered_with_topics()'s ordering
    (continuous across all topics), the SAME ordering !recall displays, so
    a number always means the same fact in both places. All positions are
    resolved against a single snapshot before anything is deleted, so
    removing e.g. both 1 and 3 in the same call is safe — an earlier
    deletion never shifts what a later index refers to. Returns
    (removed_descriptions, invalid_positions); invalid_positions covers
    anything out of range so the caller can report it back."""
    rows = get_user_facts_ordered_with_topics(user_id)

    removed, invalid, to_delete_ids = [], [], []
    for idx in indices:
        pos = idx - 1
        if 0 <= pos < len(rows):
            fact_id, topic_label, fact_text = rows[pos]
            to_delete_ids.append(fact_id)
            removed.append(f"{fact_text} [{topic_label}]")
        else:
            invalid.append(idx)

    if to_delete_ids:
        conn = _get_memory_conn()
        try:
            conn.executemany("DELETE FROM facts WHERE id = ?", [(i,) for i in to_delete_ids])
            conn.commit()
        finally:
            conn.close()
    return removed, invalid


def clear_user_facts(user_id: str):
    conn = _get_memory_conn()
    try:
        conn.execute("DELETE FROM facts WHERE user_id = ?", (str(user_id),))
        conn.execute("DELETE FROM topics WHERE user_id = ?", (str(user_id),))
        conn.commit()
    finally:
        conn.close()


def _cluster_facts_by_topic(facts_with_emb: list) -> list:
    """Groups facts into clusters via union-find over pairwise cosine
    similarity at CONSOLIDATION_TOPIC_THRESHOLD. Used both for sub-
    clustering WITHIN a topic during consolidation, and for grouping
    pre-topic facts during the one-time migration. Same same-space-only
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
    """Shared selection logic for both consolidate_topic and
    preview_topic_consolidation: cluster by topic-similarity, keep clusters
    big enough to be worth merging, largest first, capped at
    CONSOLIDATION_MAX_CLUSTERS_PER_PASS — so a preview always shows exactly
    what a real run would touch, and vice versa."""
    clusters = _cluster_facts_by_topic(facts_with_emb)
    eligible = [c for c in clusters if len(c) >= CONSOLIDATION_MIN_CLUSTER_SIZE]
    # Largest clusters first — they free up the most rows per LLM call,
    # which matters most since only the top N get processed this pass.
    eligible.sort(key=len, reverse=True)
    return eligible[:CONSOLIDATION_MAX_CLUSTERS_PER_PASS]


async def preview_topic_consolidation(
    user_id: str, topic_id: int, force: bool = False
) -> "list[tuple[list[str], str | None]] | None":
    """Dry-run counterpart to consolidate_topic — same clustering and
    selection, and the same LLM merge call per cluster, but nothing is ever
    written to the database. Returns a list of (original_fact_texts,
    merged_text_or_None) tuples, one per eligible cluster, or None if
    there's nothing eligible to show."""
    facts_with_emb = _get_topic_facts_with_embeddings(user_id, topic_id)
    if not force and len(facts_with_emb) < CONSOLIDATION_TOPIC_TRIGGER_COUNT:
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


async def consolidate_topic(user_id: str, topic_id: int, force: bool = False) -> "tuple[int, int] | None":
    """Sweeps a SINGLE topic's facts for sub-clusters and summarizes up to
    CONSOLIDATION_MAX_CLUSTERS_PER_PASS of the largest ones (each
    CONSOLIDATION_MIN_CLUSTER_SIZE+ facts) into single, shorter entries.
    This is lossy — minor/redundant details from a cluster may not survive
    — trading detail for a real reduction in both row count and
    prompt-injection tokens. Only runs once this topic's fact count reaches
    CONSOLIDATION_TOPIC_TRIGGER_COUNT, unless force=True.
    Returns (count_before, count_after) if anything was merged, else None."""
    facts_with_emb = _get_topic_facts_with_embeddings(user_id, topic_id)
    before = len(facts_with_emb)
    if not force and before < CONSOLIDATION_TOPIC_TRIGGER_COUNT:
        return None

    eligible = _eligible_clusters_for_consolidation(facts_with_emb)
    merged_any = False
    for cluster in eligible:
        texts = [f for f, _ in cluster]
        merged = await _llm_consolidate_cluster(texts)
        if not merged:
            continue  # leave this cluster untouched rather than risk losing detail
        embedding = await _embed_text(merged)
        conn = _get_memory_conn()
        try:
            placeholders = ",".join("?" * len(texts))
            conn.execute(
                f"DELETE FROM facts WHERE user_id = ? AND topic_id = ? AND fact IN ({placeholders})",
                (str(user_id), topic_id, *texts),
            )
            conn.commit()
        finally:
            conn.close()
        add_user_fact(user_id, merged, topic_id, embedding)
        merged_any = True

    if not merged_any:
        return None
    return before, _count_topic_facts(user_id, topic_id)


async def consolidate_all_topics(user_id: str, force: bool = False) -> list:
    """Loops consolidate_topic across every one of a user's topics — powers
    the bare `!consolidate` command. Returns [(topic_label, before, after), ...]
    for topics where something was actually merged."""
    results = []
    for label, _count in get_user_topics(user_id):
        conn = _get_memory_conn()
        try:
            row = conn.execute(
                "SELECT id FROM topics WHERE user_id = ? AND label = ?",
                (str(user_id), label),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            continue
        result = await consolidate_topic(user_id, row[0], force=force)
        if result:
            before, after = result
            results.append((label, before, after))
    return results


async def preview_all_topics_consolidation(user_id: str, force: bool = False) -> list:
    """Loops preview_topic_consolidation across every topic — powers
    `!consolidate preview`. Returns [(topic_label, cluster_results), ...]
    for topics with at least one eligible cluster."""
    results = []
    for label, _count in get_user_topics(user_id):
        conn = _get_memory_conn()
        try:
            row = conn.execute(
                "SELECT id FROM topics WHERE user_id = ? AND label = ?",
                (str(user_id), label),
            ).fetchone()
        finally:
            conn.close()
        if not row:
            continue
        preview = await preview_topic_consolidation(user_id, row[0], force=force)
        if preview:
            results.append((label, preview))
    return results


async def get_relevant_facts_block(user_id: str, query: str) -> str:
    """Builds the "What you remember about this user" block for the system
    prompt. Small fact lists are sent whole; larger ones are trimmed to the
    FACT_RELEVANCE_TOP_K facts most relevant to the current message, so the
    prompt doesn't grow unbounded as a user's fact count climbs toward
    MAX_FACTS_PER_USER. Stays flat/cross-topic — this is about relevance to
    the current message, not about organizing a full listing (that's
    !recall's job)."""
    facts_with_emb = get_user_facts_with_embeddings(user_id)
    if not facts_with_emb:
        return ""

    if len(facts_with_emb) <= FACT_INJECT_ALWAYS_UNDER:
        chosen = [f for f, _ in facts_with_emb]
    else:
        query_emb = await _embed_text(query)
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


# --- one-time migration of pre-topic facts ---

async def _llm_name_topic(facts: list) -> "tuple[str, str] | None":
    """One LLM call to propose a short topic label + one-line summary for a
    cluster of facts discovered during the one-time migration into topics
    (these facts predate the topic concept, so nothing tagged them at write
    time). Returns None (caller falls back to General) if the call fails or
    comes back unparseable."""
    facts_block = "\n".join(f"- {f}" for f in facts)
    prompt = (
        "These facts were all stored about the same user and turned out to be "
        "about the same underlying subject. If they're about the user's OWN "
        "identity, personality, habits, tools they use, or preferences that "
        "hold true regardless of which project they're discussing (who they "
        'are, not what they\'re working on), reply with exactly "About You" '
        "as the label. An opinion, struggle, or preference SPECIFIC to one "
        'project (e.g. "finds writing a particular character difficult") '
        "belongs to that project instead, even though it's phrased as being "
        "about the user. Otherwise, propose a short topic label (2-5 words) "
        "naming the specific ongoing project, world, or subject they share, "
        "and a one-sentence summary.\n\n"
        f"{facts_block}\n\n"
        'Reply with ONLY a JSON object of this exact shape (no markdown, no '
        'preamble): {"label": "...", "summary": "..."}'
    )
    try:
        response = await llm.query_llm(
            {"model": config.MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "stream": False},
            timeout=30,
        )
        raw = response.get("message", {}).get("content", "{}").strip().strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
        parsed = json.loads(raw)
        label = str(parsed.get("label", "")).strip()
        summary = str(parsed.get("summary", "")).strip()
        return (label, summary) if label else None
    except Exception as e:
        print(f"[MEMORY] Topic naming skipped for one cluster during migration (non-fatal): {e}")
        return None


async def migrate_facts_to_topics():
    """One-time migration for facts stored before the topics table existed
    (topic_id IS NULL — this covers both facts from before this feature
    shipped, and anything that slipped in from the legacy JSON-file
    migration above). Clusters each affected user's orphaned facts by
    embedding similarity (reusing the same clustering consolidation already
    uses) and names each real cluster via one LLM call. Facts that didn't
    cluster with anything (embedding-similarity singletons) are batched
    together and classified via the same per-fact LLM judgment
    reclassify_topic() uses, rather than defaulting straight to General —
    a single isolated personal fact should still land in "About You", not
    General, just because nothing else happened to embed near it.

    Call this once at startup (see bot.py's on_ready) — safe to call every
    time the bot starts: once every fact has a topic_id, the initial query
    returns nothing and this is a no-op.
    """
    conn = _get_memory_conn()
    try:
        user_ids = [r[0] for r in conn.execute(
            "SELECT DISTINCT user_id FROM facts WHERE topic_id IS NULL"
        ).fetchall()]
    finally:
        conn.close()

    if not user_ids:
        return

    print(f"[MEMORY] Migrating orphaned facts into topics for {len(user_ids)} user(s)...")
    for user_id in user_ids:
        conn = _get_memory_conn()
        try:
            rows = conn.execute(
                "SELECT id, fact, embedding FROM facts WHERE user_id = ? AND topic_id IS NULL",
                (user_id,),
            ).fetchall()
        finally:
            conn.close()

        facts_with_emb = [(fid, fact, _parse_embedding(raw)) for fid, fact, raw in rows]
        # _cluster_facts_by_topic works on (fact_text, embedding) pairs —
        # adapt, then map the clustered text back to its fact_id after.
        clusters = _cluster_facts_by_topic([(fact, emb) for _fid, fact, emb in facts_with_emb])
        text_to_id = {fact: fid for fid, fact, _emb in facts_with_emb}

        singleton_rows = []  # (fact_id, fact_text) — handled in a batch after real clusters
        for cluster in clusters:
            cluster_texts = [f for f, _ in cluster]
            fact_ids = [text_to_id[f] for f in cluster_texts]

            if len(cluster) < MIGRATION_MIN_CLUSTER_SIZE:
                # Too small to be worth an LLM call per-cluster — batched
                # together below via _llm_propose_topics_for_facts instead
                # of defaulting straight to General. A lone fact with no
                # embedding neighbor isn't necessarily unrelated to
                # everything else; it just didn't score close enough to
                # anything to cluster automatically. This is what keeps a
                # single isolated personal fact ("uses TickTick") from
                # defaulting to General instead of "About You" just because
                # nothing else happened to embed near it.
                singleton_rows.extend((fid, text) for fid, text in zip(fact_ids, cluster_texts))
                continue

            named = await _llm_name_topic(cluster_texts)
            topic_id = await get_or_create_topic(user_id, named[0] if named else DEFAULT_TOPIC_LABEL)
            if named and named[1]:
                conn = _get_memory_conn()
                try:
                    conn.execute("UPDATE topics SET summary = ? WHERE id = ?", (named[1], topic_id))
                    conn.commit()
                finally:
                    conn.close()

            conn = _get_memory_conn()
            try:
                conn.executemany(
                    "UPDATE facts SET topic_id = ? WHERE id = ?",
                    [(topic_id, fid) for fid in fact_ids],
                )
                conn.commit()
            finally:
                conn.close()

        for start in range(0, len(singleton_rows), RECLASSIFY_BATCH_SIZE):
            batch = singleton_rows[start:start + RECLASSIFY_BATCH_SIZE]
            existing_labels = [label for label, _c in get_user_topics(user_id)]
            proposed = await _llm_propose_topics_for_facts([text for _fid, text in batch], existing_labels)
            if proposed is None:
                # Call failed — fall back to General rather than leaving
                # these facts orphaned (topic_id NULL) forever, which would
                # otherwise make them invisible to !recall entirely.
                proposed = [DEFAULT_TOPIC_LABEL] * len(batch)

            for (fact_id, _fact_text), label in zip(batch, proposed):
                topic_id = await get_or_create_topic(user_id, label)
                conn = _get_memory_conn()
                try:
                    conn.execute("UPDATE facts SET topic_id = ? WHERE id = ?", (topic_id, fact_id))
                    conn.commit()
                finally:
                    conn.close()

    print("[MEMORY] Topic migration complete.")


# --- extraction ---

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
    existing_topic_labels = [label for label, _count in get_user_topics(user_id)]
    topics_block = ", ".join(existing_topic_labels) if existing_topic_labels else "(none yet)"

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
        '{"add": [{"fact": "new fact 1", "topic": "short topic label"}, ...], '
        '"update": [{"replaces": "<verbatim existing fact text>", "with": '
        '"<revised fact text>"}]}\n'
        "Only use \"update\" when \"replaces\" is copied verbatim from the "
        "existing facts list below — never paraphrase it.\n\n"
        "For \"topic\": a short label (2-5 words) naming the ongoing subject "
        "this fact belongs to. Decide which of these three cases applies:\n"
        "1. The fact is about the user's OWN identity, personality, habits, "
        "tools they use, or preferences that hold true regardless of which "
        "project or topic they're discussing (who they are, not what "
        "they're working on) — use exactly \"About You\" for these, even if "
        "the message also mentions a project. Don't blend a personal fact "
        "into a project topic just because both came up in the same "
        "message.\n"
        "2. The fact is part of a specific ongoing project, world, or "
        "subject (a story, a piece of software, a hobby with its own "
        "recurring details) — if it continues one of the EXISTING TOPICS "
        "below, reuse that exact label; only propose a new label for a "
        "genuinely different ongoing subject. This includes an opinion, "
        "struggle, or preference that's SPECIFIC to that project (e.g. "
        "\"finds writing a particular character difficult\", \"wants the "
        "story to end differently\") — these stay with the project's topic "
        "even though they're phrased as being about the user. \"About You\" "
        "is only for things true about the user independent of which "
        "project is being discussed.\n"
        "3. A true one-off with no lasting category and not about the "
        'user\'s own identity — use exactly "General".\n\n'
        f"Existing topics: {topics_block}\n\n"
        "If nothing applies, reply with exactly "
        '{"add": [], "update": []}\n\n'
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
        to_add = [
            (str(a.get("fact", "")).strip(), str(a.get("topic", "")).strip() or DEFAULT_TOPIC_LABEL)
            for a in parsed.get("add", [])
            if isinstance(a, dict) and str(a.get("fact", "")).strip()
        ]
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
        embedding = await _embed_text(new_text)
        result = update_user_fact(user_id, str(u["replaces"]), new_text, embedding)
        if result:
            updated_summaries.append(f"{result[0]} → {result[1]}")
        else:
            # "replaces" didn't match anything on file — treat as a fresh
            # fact rather than silently dropping it. No topic hint came with
            # this one, so it falls back to General; if that's wrong the
            # next extraction pass or a manual !topics move can fix it.
            to_add.append((new_text, DEFAULT_TOPIC_LABEL))

    added = []
    consolidation_summaries = []
    consolidation_exhausted_topics: "set[int]" = set()
    if to_add:
        existing_for_dedup = get_user_facts_with_embeddings(user_id)
        for fact, proposed_topic in to_add:
            if not fact:
                continue
            embedding = await _embed_text(fact)
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

            topic_id = await get_or_create_topic(user_id, proposed_topic)

            # Checked per-fact, not just once before the loop — a single
            # message can extract many new facts at once (a big paste), and
            # a single up-front check would let that whole batch sail past
            # the trigger before the sweep got a second look.
            # consolidation_exhausted_topics stops it from retrying a topic
            # that's already been swept and found nothing left to merge this
            # pass — nothing changed, so an immediate retry can't help;
            # add_user_fact's own global-cap trim remains the safety net for
            # whatever's left in that case.
            if topic_id not in consolidation_exhausted_topics:
                topic_count = _count_topic_facts(user_id, topic_id)
                if topic_count >= CONSOLIDATION_TOPIC_TRIGGER_COUNT:
                    result = await consolidate_topic(user_id, topic_id)
                    if result:
                        before, after = result
                        conn = _get_memory_conn()
                        try:
                            row = conn.execute(
                                "SELECT label FROM topics WHERE id = ?", (topic_id,)
                            ).fetchone()
                        finally:
                            conn.close()
                        topic_label = row[0] if row else "?"
                        consolidation_summaries.append(f"{topic_label}: {before} → {after} facts")
                        existing_for_dedup = get_user_facts_with_embeddings(user_id)
                    else:
                        consolidation_exhausted_topics.add(topic_id)

            if add_user_fact(user_id, fact, topic_id, embedding):
                added.append(fact)
                existing_for_dedup.append((fact, embedding))

    consolidation_summary = "; ".join(consolidation_summaries) if consolidation_summaries else None

    if channel is not None and (added or updated_summaries or consolidation_summary):
        lines = [f"-# 🧠 remembered: {f}" for f in added]
        lines += [f"-# 🧠 updated: {s}" for s in updated_summaries]
        if consolidation_summary:
            lines.append(f"-# 🧠 consolidated: {consolidation_summary}")
        await messaging.send_chunked(channel, "\n".join(lines))
