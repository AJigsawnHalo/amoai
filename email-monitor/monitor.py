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
from dotenv import load_dotenv, find_dotenv

# Automatically crawls up parent directories to locate your central .env file
load_dotenv(find_dotenv())

# --- CONFIGURATION ---
MODEL_NAME = "gpt-oss:20b-cloud"  # Baked in: Smarter context window & reasoning for email digestion
OLLAMA_API = os.getenv("OLLAMA_API", "http://localhost:11434/api/chat")  # Kept in .env


# --- UTILITIES & LLM FUNCTIONS ---

def query_llm(messages, options=None, timeout=30):
    """Queries the local Ollama instance using the Chat API."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
    }
    if options:
        payload["options"] = options

    try:
        response = requests.post(OLLAMA_API, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("message", {}).get("content", "")
    except Exception as e:
        return f"Error: {e}"


def clean_llm_response(text):
    """Cleans up LLM outputs by removing markdown and punctuation."""
    return text.strip().lower().replace("*", "").replace("`", "").strip(".,!?\"'")


def classify_email(subject, sender, snippet):
    """Uses LLM to classify emails into newsletter, important, or ignore."""
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
    result = query_llm(messages, options={"temperature": 0.1, "num_ctx": 512}, timeout=30)
    cleaned = clean_llm_response(result)

    if "newsletter" in cleaned:
        return "newsletter"
    if "important" in cleaned:
        return "important"
    return "ignore"


def summarize_email(subject, sender, body):
    """Uses LLM to summarize email body."""
    prompt = f"""Summarize this email in 2 sentences max.
From: {sender}
Subject: {subject}
Body: {body[:1000]}
Be concise and factual."""
    messages = [{"role": "user", "content": prompt}]
    # Longer timeout — summarization gets more context than classification
    # and shouldn't get falsely flagged as an "Error:" on slower local models.
    return query_llm(messages, options={"temperature": 0.3, "num_ctx": 1024}, timeout=90)


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
            snippet = body[:300]

            print(f"Processing: {subject[:50]}")
            classification = classify_email(subject, sender, snippet)
            print(f"  → {classification}")

            if classification == "newsletter":
                mail.uid("STORE", msg_id, "+X-GM-LABELS", "Newsletter")
                mail.uid("STORE", msg_id, "+FLAGS", "\\Seen")
                typ, _ = mail.uid("COPY", msg_id, '"[Gmail]/All Mail"')
                if typ == "OK":
                    mail.uid("STORE", msg_id, "+FLAGS", "\\Deleted")
                    needs_expunge = True
            elif classification == "important":
                summary = summarize_email(subject, sender, body)
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
            classification = classify_email("Piped Content", "Command Line Stream", input_data[:300])
            print(f"Classification: {classification}")

            if classification == "important":
                summary = summarize_email("Piped Content", "Command Line Stream", input_data)
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
