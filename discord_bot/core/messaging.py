"""
messaging.py — Discord message-sending helpers: chunking past the 2000-char
limit, reaction-based confirmation prompts, and automatic thread suggestion.

Needs memory_store for conversation-log/thread-suggestion state — that's a
normal top-level import here. memory_store.py, in turn, needs
messaging.send_chunked, but defers that import to inside the one function
that needs it, to break the cycle. See the note at the top of
memory_store.py.
"""
import asyncio
from datetime import datetime

import discord

from discord_client import bot
import memory_store
from tools.reminder_tool import BOT_TIMEZONE

DISCORD_LIMIT = 2000

# Persisted in SQLite (thread_suggestion_state), not just kept in memory —
# history_length is hydrated from the persistent conversation_log on restart,
# so an in-memory-only cooldown tracker would reset to "never suggested" on
# every restart while history_length stays wherever it left off, causing an
# immediate re-suggestion on the first message after every restart.
THREAD_SUGGESTION_THRESHOLD = 6   # messages in history before suggesting
THREAD_SUGGESTION_COOLDOWN = 10   # messages before asking again after a decline/timeout
THREAD_SEED_MESSAGES = 6          # most recent messages copied into the new thread for continuity


async def send_chunked(channel, text: str):
    text = text or "⚠️ (empty response — check bot logs)"
    if len(text) <= DISCORD_LIMIT:
        await channel.send(text)
        return

    remaining = text
    while remaining:
        if len(remaining) <= DISCORD_LIMIT:
            await channel.send(remaining)
            break
        cut = remaining.rfind("\n", 0, DISCORD_LIMIT)
        if cut == -1:
            cut = remaining.rfind(" ", 0, DISCORD_LIMIT)
        if cut == -1:
            cut = DISCORD_LIMIT
        await channel.send(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n ")


async def confirm_with_reaction(message, prompt_text: str, timeout: int = 60) -> bool:
    # Use send_chunked to avoid the 2000 character limit
    await send_chunked(message.channel, prompt_text)

    # Send a small confirmation prompt to add the reactions to
    confirm_msg = await message.channel.send("React ✅ to confirm or ❌ to cancel (60s).")
    await confirm_msg.add_reaction("✅")
    await confirm_msg.add_reaction("❌")

    def check(reaction, user):
        return (
            user == message.author
            and reaction.message.id == confirm_msg.id
            and str(reaction.emoji) in ("✅", "❌")
        )

    try:
        reaction, _ = await bot.wait_for("reaction_add", timeout=timeout, check=check)
        return str(reaction.emoji) == "✅"
    except asyncio.TimeoutError:
        await send_chunked(message.channel, "⏳ No response in time — action cancelled.")
        return False


async def maybe_suggest_thread(message, history_length: int):
    """Checks if the conversation is getting long and offers to spin up a thread,
    using the built-in reaction confirmation system."""
    channel_id = message.channel.id

    if isinstance(message.channel, discord.Thread):
        return

    last_suggested_at = memory_store._get_last_thread_suggestion(channel_id)
    cooldown_met = (last_suggested_at == 0) or (history_length - last_suggested_at >= THREAD_SUGGESTION_COOLDOWN)

    if history_length >= THREAD_SUGGESTION_THRESHOLD and cooldown_met:
        memory_store._set_last_thread_suggestion(channel_id, history_length)

        approved = await confirm_with_reaction(
            message,
            "🧵 This conversation is getting a bit long. Would you like to move this topic to a new thread?"
        )

        if approved:
            try:
                thread_name = f"Topic Discussion - {datetime.now(BOT_TIMEZONE).strftime('%H:%M')}"
                new_thread = await message.create_thread(name=thread_name, auto_archive_duration=1440)

                # Seed the new thread's persistent log with the recent exchange so
                # the bot still has context once the conversation continues there —
                # without this, get_channel_history(new_thread) starts empty and the
                # whole point of moving a long conversation over is lost.
                recent = list(memory_store.get_channel_history(message.channel))[-THREAD_SEED_MESSAGES:]
                for turn in recent:
                    memory_store._append_conversation_log(new_thread.id, turn["role"], turn["content"])
                memory_store._HYDRATED_CHANNELS.discard(new_thread.id)  # force a fresh hydrate on first use

                recap = "\n".join(
                    f"**{'You' if t['role'] == 'user' else 'Amoai'}:** {t['content'][:300]}"
                    for t in recent
                )
                await send_chunked(
                    new_thread,
                    f"🧵 Picking up here — here's where we left off:\n\n{recap}\n\n"
                    "Go ahead with your follow-up questions."
                )
            except discord.HTTPException as e:
                await send_chunked(message.channel, f"⚠️ Failed to create thread: {e}")
