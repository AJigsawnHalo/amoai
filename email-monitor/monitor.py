import os
import re
import sys
import time
import json
import ssl
import socket
import fcntl
import imaplib
import email
import requests
from email.header import decode_header
from urllib.parse import urlsplit
from dotenv import load_dotenv, find_dotenv

# Automatically crawls up parent directories to locate your central .env file
load_dotenv(find_dotenv())

# --- CONFIGURATION ---
MODEL_NAME = "phi4-mini:latest"  # Baked in: Smarter context window & reasoning for email digestion
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")  # Kept in .env

# Optional .env override; otherwise the MODEL_NAME set above is used.
MODEL_NAME = os.getenv("EMAIL_MONITOR_MODEL") or MODEL_NAME

# A cold model load can take far longer than a classification call, so the first
# request gets its own long warm-up timeout; keep_alive keeps the model loaded.
WARMUP_TIMEOUT = int(os.getenv("EMAIL_MONITOR_WARMUP_TIMEOUT", 180))
KEEP_ALIVE = os.getenv("EMAIL_MONITOR_KEEP_ALIVE", "10m")
# Ollama reloads a model when load-time options such as num_ctx differ between
# requests, so the warm-up and every real call must use the SAME value.
LLM_NUM_CTX = int(os.getenv("EMAIL_MONITOR_NUM_CTX", 4096))
# hiryu's GPU does roughly 12 tok/s on prompt reading and 3 tok/s on output, so a
# classification (~350 prompt tokens) takes ~30s and a summary around a minute.
CLASSIFY_TIMEOUT = int(os.getenv("EMAIL_MONITOR_CLASSIFY_TIMEOUT", 120))
SUMMARY_TIMEOUT = int(os.getenv("EMAIL_MONITOR_SUMMARY_TIMEOUT", 180))
# How much of an email the model reads. Reading speed (~12 tok/s here) is the real
# cost, not context size. Keep CLASSIFY_CHARS + the prompt well under LLM_NUM_CTX.
CLASSIFY_CHARS = int(os.getenv("EMAIL_MONITOR_CLASSIFY_CHARS", 500))
SUMMARY_CHARS = int(os.getenv("EMAIL_MONITOR_SUMMARY_CHARS", 3000))
# Mail with a List-Unsubscribe header is bulk mail: archive it without the model.
# Senders matching EMAIL_MONITOR_KEEP_SENDERS (comma-separated, case-insensitive
# substrings, e.g. your bank or employer) always go through the model instead.
UNSUB_SHORTCUT = os.getenv("EMAIL_MONITOR_UNSUB_SHORTCUT", "1").lower() not in ("0", "false", "no", "off")
KEEP_SENDERS = [x.strip().lower() for x in os.getenv("EMAIL_MONITOR_KEEP_SENDERS", "").split(",") if x.strip()]
_model_warmed = False

ALERT_STATE_FILE = "/tmp/amoai_email_monitor_last_llm_alert"
ALERT_COOLDOWN_SECONDS = 6 * 3600  # don't spam the channel on every run


class LLMUnavailable(Exception):
    """The LLM call failed or returned nothing usable. Mail must NOT be marked
    read or archived when this happens - leave it untouched for the next run."""


# --- UTILITIES & LLM FUNCTIONS ---

def warm_up_model():
    """Loads the model into memory and waits for it (up to WARMUP_TIMEOUT).
    An empty message list makes Ollama load the model and return. Without this,
    a slow cold load outlasts the 30s request timeout, the client disconnects,
    Ollama cancels the load, and the model never shows up in `ollama ps`."""
    global _model_warmed
    if _model_warmed:
        return
    parts = urlsplit(OLLAMA_API)
    generate_url = f"{parts.scheme}://{parts.netloc}/api/generate"
    try:
        # An empty prompt on /api/generate loads the model and returns (same path
        # `ollama run` uses).
        r = requests.post(
            generate_url,
            json={"model": MODEL_NAME, "prompt": "", "stream": False, "keep_alive": KEEP_ALIVE,
                  "options": {"num_ctx": LLM_NUM_CTX}},
            timeout=WARMUP_TIMEOUT,
        )
        r.raise_for_status()
    except Exception as e:
        raise LLMUnavailable(
            f"model '{MODEL_NAME}' failed to load within {WARMUP_TIMEOUT}s: {e}"
        ) from e
    _model_warmed = True


def query_llm(messages, options=None, timeout=30):
    """Queries the local Ollama instance using the Chat API."""
    warm_up_model()
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
    }
    if options:
        payload["options"] = options

    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=timeout)
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
    except Exception as e:
        raise LLMUnavailable(f"model '{MODEL_NAME}': {e}") from e
    if not content or not content.strip():
        raise LLMUnavailable(f"model '{MODEL_NAME}' returned an empty reply")
    return content


def clean_llm_response(text):
    """Cleans up LLM outputs by removing markdown and punctuation."""
    return text.strip().lower().replace("*", "").replace("`", "").strip(".,!?\"'")


def should_shortcut_newsletter(msg, sender):
    """True for bulk mail (has a List-Unsubscribe header) from a sender that is
    not in KEEP_SENDERS, so it can be archived without spending GPU time."""
    if not UNSUB_SHORTCUT or not msg.get("List-Unsubscribe"):
        return False
    s = (sender or "").lower()
    return not any(k in s for k in KEEP_SENDERS)


def classify_email(subject, sender, snippet):
    """Uses LLM to classify emails into newsletter, important, or ignore.
    Returns None if the reply is unusable (caller leaves the mail untouched).
    Raises LLMUnavailable if the model can't be reached."""
    prompt = f"""Classify this email into exactly one of three categories: "newsletter", "important", or "ignore".
Sender: {sender}
Subject: {subject}
Preview: {snippet}
Rules:
- newsletter: marketing emails, promotions, digests, announcements, subscription content, no-reply addresses, anything with unsubscribe links, livestream notifications
- important: personal messages requiring attention, work-related emails, financial transactions, security alerts, direct replies from real people, anything time-sensitive
- ignore: automated notifications you don't need to act on (order confirmations, shipping updates, social media notifications, calendar invites you already know about, system alerts, receipts)
Respond with ONLY one word: newsletter, important, or ignore"""
    messages = [{"role": "user", "content": prompt}]
    result = query_llm(messages, options={"temperature": 0.1, "num_ctx": LLM_NUM_CTX}, timeout=CLASSIFY_TIMEOUT)
    cleaned = clean_llm_response(result)

    if "newsletter" in cleaned:
        return "newsletter"
    if "important" in cleaned:
        return "important"
    if "ignore" in cleaned:
        return "ignore"
    return None  # unparseable reply: don't guess, don't mark anything read


def summarize_email(subject, sender, body):
    """Uses LLM to summarize email body."""
    text = " ".join(body.split())[:SUMMARY_CHARS]
    prompt = f"""Summarize this email in 2 sentences max.
From: {sender}
Subject: {subject}
Body: {text}
Be concise and factual."""
    messages = [{"role": "user", "content": prompt}]
    # Longer timeout — summarization gets more context than classification
    # and shouldn't get falsely flagged as an "Error:" on slower local models.
    return query_llm(messages, options={"temperature": 0.3, "num_ctx": LLM_NUM_CTX}, timeout=SUMMARY_TIMEOUT)


def send_discord_notification(subject, sender, summary, account):
    """Sends structured embed to the email webhook channel."""
    webhook_url = os.getenv("WEBHOOK_EMAIL")
    if not webhook_url:
        print("Warning: WEBHOOK_EMAIL is not configured in .env", file=sys.stderr)
        return

    payload = {
        "embeds": [{
            "title": "📧 Important Email",
            "color": 0xff6b6b,
            "fields": [
                {"name": "Account", "value": account, "inline": True},
                {"name": "From", "value": sender, "inline": True},
                {"name": "Subject", "value": subject, "inline": True},
                {"name": "Summary", "value": summary, "inline": False}
            ],
            "footer": {"text": "Hiryu Email Monitor"}
        }]
    }
    try:
        requests.post(webhook_url, json=payload, timeout=10)
    except Exception as e:
        print(f"    ❌ Failed sending Discord hook: {e}", file=sys.stderr)


def send_llm_alert_throttled(error_msg):
    """Alerts Discord that the classifier model is down, at most once per cooldown."""
    try:
        if time.time() - os.path.getmtime(ALERT_STATE_FILE) < ALERT_COOLDOWN_SECONDS:
            return
    except OSError:
        pass  # no marker yet
    send_failure_alert(
        "all accounts",
        f"Email classifier model unavailable — mail left untouched (not marked read).\n{error_msg}",
    )
    try:
        with open(ALERT_STATE_FILE, "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def send_failure_alert(account, error_msg):
    """Sends a system-level alert to Discord if account processing crashes."""
    webhook_url = os.getenv("WEBHOOK_EMAIL")
    if webhook_url:
        try:
            requests.post(webhook_url, json={
                "content": f"⚠️ **Email monitor failed for {account}**\n```{error_msg}```"
            }, timeout=10)
        except Exception as e:
            print(f"Failed to deliver system alert to Discord: {e}", file=sys.stderr)


def decode_mime_header(header_value):
    """Safely decodes all parts of an email header."""
    if not header_value:
        return "Unknown"
    decoded_parts = decode_header(header_value)
    header_text = []
    for part, encoding in decoded_parts:
        if isinstance(part, bytes):
            header_text.append(part.decode(encoding or "utf-8", errors="ignore"))
        else:
            header_text.append(str(part))
    return "".join(header_text)


def get_email_body(msg):
    """Extracts text content, preferring text/plain but falling back to
    a crude tag-stripped version of text/html when no plain part exists."""
    plain, html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            charset = part.get_content_charset() or "utf-8"
            if ctype == "text/plain" and not plain:
                payload = part.get_payload(decode=True)
                if payload:
                    plain = payload.decode(charset, errors="ignore")
            elif ctype == "text/html" and not html:
                payload = part.get_payload(decode=True)
                if payload:
                    html = payload.decode(charset, errors="ignore")
    else:
        charset = msg.get_content_charset() or "utf-8"
        payload = msg.get_payload(decode=True)
        decoded = payload.decode(charset, errors="ignore") if payload else ""
        if msg.get_content_type() == "text/html":
            html = decoded
        else:
            plain = decoded

    if plain:
        return plain
    if html:
        return re.sub("<[^<]+?>", " ", html)  # crude tag strip, good enough for a snippet
    return ""


# --- MAIN RUN LOGIC ---

def process_emails(user, password):
    """Connects to IMAP and processes pending unseen messages."""
    mail = imaplib.IMAP4_SSL("imap.gmail.com", timeout=30)
    try:
        mail.login(user, password)
        mail.select("inbox")
        _, message_ids = mail.uid("SEARCH", None, "UNSEEN")
        ids = message_ids[0].split()
        if not ids:
            print(f"No new emails for {user}")
            return

        needs_expunge = False
        for msg_id in ids:
            _, msg_data = mail.uid("FETCH", msg_id, "(RFC822)")
            if not msg_data or msg_data[0] is None:
                continue
            raw = msg_data[0][1]
            msg = email.message_from_bytes(raw)

            subject = decode_mime_header(msg["Subject"])
            sender = decode_mime_header(msg["From"])
            body = get_email_body(msg)
            snippet = " ".join(body.split())[:CLASSIFY_CHARS]

            print(f"Processing: {subject[:50]}")
            if should_shortcut_newsletter(msg, sender):
                classification = "newsletter"
                print("  → newsletter (List-Unsubscribe header, model skipped)")
            else:
                classification = classify_email(subject, sender, snippet)  # may raise LLMUnavailable
                print(f"  → {classification}")
            if classification is None:
                print("  → unusable model reply; leaving unread for next run", file=sys.stderr)
                continue

            if classification == "newsletter":
                mail.uid("STORE", msg_id, "+X-GM-LABELS", "Newsletter")
                mail.uid("STORE", msg_id, "+FLAGS", "\\Seen")
                typ, _ = mail.uid("COPY", msg_id, '"[Gmail]/All Mail"')
                if typ == "OK":
                    mail.uid("STORE", msg_id, "+FLAGS", "\\Deleted")
                    needs_expunge = True
            elif classification == "important":
                try:
                    summary = summarize_email(subject, sender, body)
                except LLMUnavailable:
                    summary = f"(summary unavailable) {body.strip()[:300]}"
                send_discord_notification(subject, sender, summary, user)
                # Mark seen so re-runs don't re-notify on the same email.
                mail.uid("STORE", msg_id, "+FLAGS", "\\Seen")
                print(f"  → Discord notified")
            else:
                mail.uid("STORE", msg_id, "+FLAGS", "\\Seen")
                print(f"  → marked read")

            time.sleep(1)

        if needs_expunge:
            mail.expunge()
    finally:
        # Always logout, even if something above raised — avoids
        # dangling IMAP connections piling up until server timeout.
        try:
            mail.logout()
        except Exception:
            pass


def process_emails_with_retry(user, password, retries=1):
    """Wraps process_emails with one retry on transient SSL/socket errors."""
    for attempt in range(retries + 1):
        try:
            process_emails(user, password)
            return
        except (ssl.SSLError, socket.error, imaplib.IMAP4.abort) as e:
            if attempt < retries:
                print(f"  → Transient error ({e}), retrying in 5s...", file=sys.stderr)
                time.sleep(5)
            else:
                raise


def run_task(task_type):
    # Check if a text block was piped directly via terminal
    if not sys.stdin.isatty():
        input_data = sys.stdin.read().strip()
    else:
        input_data = None

    if task_type == "sort_emails":
        # 1. PIPED MODE (e.g. cat email_draft.txt | python email-monitor/task_runner.py sort_emails)
        if input_data:
            print("Processing raw email content received from standard input...")
            try:
                classification = classify_email("Piped Content", "Command Line Stream", " ".join(input_data.split())[:CLASSIFY_CHARS])
            except LLMUnavailable as e:
                print(f"Error: model unavailable: {e}", file=sys.stderr)
                return
            print(f"Classification: {classification}")

            if classification == "important":
                try:
                    summary = summarize_email("Piped Content", "Command Line Stream", input_data)
                except LLMUnavailable:
                    summary = f"(summary unavailable) {input_data[:300]}"
                send_discord_notification("Piped Content", "Command Line Stream", summary, "Local CLI Pipe")
                print("Important content detected; Discord webhook pinged.")
            return

        # 2. ACTIVE MONITOR MODE (No input piped; check active email accounts)
        lock_path = "/tmp/amoai_email_monitor.lock"
        lock_fd = open(lock_path, "w")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another email monitor run is already in progress, skipping.", file=sys.stderr)
            return

        try:
            accounts_raw = os.getenv("EMAIL_ACCOUNTS")
            if not accounts_raw:
                print("Error: EMAIL_ACCOUNTS not defined in environment.", file=sys.stderr)
                return
            try:
                accounts = json.loads(accounts_raw)
            except json.JSONDecodeError as e:
                print(f"Error parsing EMAIL_ACCOUNTS JSON structure: {e}", file=sys.stderr)
                return

            for account in accounts:
                user = account.get("user")
                password = account.get("password")
                if not user or not password:
                    continue
                print(f"\n--- Checking {user} ---")
                try:
                    process_emails_with_retry(user, password)
                except LLMUnavailable as e:
                    # Every account would fail the same way — stop, leave mail untouched.
                    print(f"  → Model unavailable, aborting run: {e}", file=sys.stderr)
                    send_llm_alert_throttled(str(e))
                    break
                except Exception as e:
                    print(f"  → Failed: {e}", file=sys.stderr)
                    send_failure_alert(user, str(e))
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            lock_fd.close()
    else:
        print(f"Error: Unknown task type '{task_type}'", file=sys.stderr)


if __name__ == "__main__":
    task_arg = sys.argv[1] if len(sys.argv) > 1 else "sort_emails"
    run_task(task_arg)
