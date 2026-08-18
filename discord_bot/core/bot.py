"""
bot.py — Discord event wiring and the tool-calling loop. Everything else
(config, LLM calls, tool discovery, embeddings, memory, messaging, webhook,
scheduler, attachments) lives in the sibling core/ modules and is imported
below.

DEPLOYMENT NOTE: this file now lives one directory deeper than it used to
(discord_bot/bot.py -> discord_bot/core/bot.py). Running it directly
(`python3 core/bot.py`) only puts core/ itself on sys.path — NOT
discord_bot/, which is where tools/ lives. The sys.path insert below fixes
that; it must run before any of the imports that follow it, since several
of those (tool_registry, scheduler, webhook, messaging, attachments) do
`import tools` themselves at their own module level.

Also update discord-bot.service's ExecStart / WorkingDirectory to point at
core/bot.py instead of the old bot.py path — see migrate_data_layout.py for
the data-file side of this move; this is the code-path side.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import inspect
from datetime import datetime

import discord

import config
from discord_client import bot
import llm
import tool_registry
import embeddings
import memory_store
import messaging
import webhook
import scheduler
import attachments
from tools.reminder_tool import BOT_TIMEZONE

ACTIVE_TASKS: "dict[str, asyncio.Task]" = {}

_startup_notified = False


@bot.event
async def on_ready():
    global _startup_notified
    print(f"[SYSTEM] Logged in as {bot.user}")
    if not scheduler.scheduler_tick.is_running():
        scheduler.scheduler_tick.start()
    await webhook.start_webhook_server()
    await embeddings.embed_all_tools()

    # Wire job_manager to this process's event loop and a way to post
    # results back to Discord when a background job finishes on its own.
    from tools.job_manager import set_event_loop, set_notifier
    set_event_loop(asyncio.get_event_loop())
    if config.ALLOWED_CHANNEL_ID:
        async def _notify_job_channel(text: str):
            channel = bot.get_channel(config.ALLOWED_CHANNEL_ID)
            if channel:
                await messaging.send_chunked(channel, text)
        set_notifier(_notify_job_channel)

    # Only announce once per process start — on_ready can fire again on reconnects
    if not _startup_notified:
        _startup_notified = True
        if config.ALLOWED_CHANNEL_ID:
            channel = bot.get_channel(config.ALLOWED_CHANNEL_ID)
            if channel:
                await channel.send(f"🔄 Restarted and online as **{bot.user}**.")


@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    if config.ALLOWED_CHANNEL_ID:
        in_allowed_channel = message.channel.id == config.ALLOWED_CHANNEL_ID
        in_allowed_thread = (
            isinstance(message.channel, discord.Thread)
            and message.channel.parent_id == config.ALLOWED_CHANNEL_ID
        )
        if not (in_allowed_channel or in_allowed_thread):
            return

    user_query = message.content
    user_id = str(message.author.id)

    # --- REPLY CONTEXT: if this message is a Discord reply, pull in the
    # message being replied to so the model has it even if it's long since
    # scrolled out of the rolling channel history. ---
    reply_context = await attachments.get_reply_context(message)
    has_reply_context = bool(reply_context)
    if reply_context:
        user_query = f"{reply_context}\n{user_query}" if user_query else reply_context

    # --- ATTACHMENTS: images go to vision, everything else gets read as text ---
    pending_images_b64 = []
    file_context = ""
    if message.attachments:
        pending_images_b64, image_notes = await attachments.process_image_attachments(message.attachments)
        file_context, file_notes = await attachments.process_file_attachments(message.attachments)

        if file_context:
            user_query = f"{user_query}\n\n[Attached file contents below]\n{file_context}" if user_query else file_context
        elif not user_query and pending_images_b64:
            user_query = "Take a look at the attached image(s) and describe what you see."

        attachment_notes = image_notes + file_notes
        if attachment_notes:
            await messaging.send_chunked(message.channel, "\n".join(attachment_notes))

    # --- VISION PRE-PASS: describe images as text BEFORE the tool-calling
    # loop starts. gemma4:cloud (like most vision models) can 500 when
    # "images" and "tools" are both present in one request, so images never
    # travel alongside "tools" — they're converted to a text description
    # here instead, and the main loop below never sees an "images" field. ---
    had_images = bool(pending_images_b64)
    if pending_images_b64:
        await messaging.send_chunked(message.channel, "👀 Looking at the image(s)...")
        async with message.channel.typing():
            image_description = await attachments.describe_images_for_tools(pending_images_b64, user_query, message.channel)
        if image_description:
            user_query = (
                f"{user_query}\n\n[Image content — extracted by vision pass]\n{image_description}"
                if user_query else
                f"[Image content — extracted by vision pass]\n{image_description}"
            )
        else:
            await messaging.send_chunked(
                message.channel,
                "⚠️ Couldn't get a description of the attached image(s) — continuing without them."
            )
        pending_images_b64 = []  # already folded into user_query as text; never attach raw images downstream

    trigger = user_query.strip().lower()

    if trigger in ("!stop", "!cancel", "!halt"):
        task = ACTIVE_TASKS.get(user_id)
        if task and not task.done():
            task.cancel()
            await messaging.send_chunked(message.channel, "🛑 Stopping...")
        else:
            await messaging.send_chunked(message.channel, "Nothing's running right now.")
        return

    if trigger in ("!briefing on", "!briefing off", "!briefing status"):
        if trigger == "!briefing on":
            scheduler.MORNING_BRIEFING_ENABLED = True
            await messaging.send_chunked(message.channel, "☀️ Morning briefing is now **on**.")
        elif trigger == "!briefing off":
            scheduler.MORNING_BRIEFING_ENABLED = False
            await messaging.send_chunked(message.channel, "🔕 Morning briefing is now **off**.")
        else:
            state = "on" if scheduler.MORNING_BRIEFING_ENABLED else "off"
            await messaging.send_chunked(message.channel, f"Morning briefing is currently **{state}**.")
        return

    if trigger == "!allowlist" or trigger.startswith("!allowlist "):
        # Owner-only: this gates a security control (which Docker containers
        # the LLM's restart_container tool may touch), so it's checked here
        # rather than relying on channel membership like the other ! commands.
        if user_id != config.DISCORD_USER_ID:
            await messaging.send_chunked(message.channel, "🔒 Only the bot owner can manage the restart allowlist.")
            return

        from tools.docker_manager import get_allowlist, _add_to_allowlist, _remove_from_allowlist

        if trigger in ("!allowlist", "!allowlist list"):
            current = sorted(get_allowlist())
            text = (
                "Current restart allowlist:\n" + "\n".join(f"- {c}" for c in current)
                if current else
                "The restart allowlist is currently empty — no containers can be restarted."
            )
            await messaging.send_chunked(message.channel, text)
        elif trigger.startswith("!allowlist add "):
            container_name = user_query.strip()[len("!allowlist add "):].strip()
            await messaging.send_chunked(message.channel, _add_to_allowlist(container_name))
        elif trigger.startswith("!allowlist remove "):
            container_name = user_query.strip()[len("!allowlist remove "):].strip()
            await messaging.send_chunked(message.channel, _remove_from_allowlist(container_name))
        else:
            await messaging.send_chunked(
                message.channel,
                "Usage: `!allowlist` (show current), `!allowlist add <container>`, "
                "`!allowlist remove <container>`."
            )
        return

    if trigger in ("!consolidate preview", "!consolidatepreview", "!consolidate dryrun"):
        async with message.channel.typing():
            results = await memory_store.preview_consolidation(user_id, force=True)
        if not results:
            text = "Nothing to consolidate right now — no cluster of related facts is big enough to merge."
        else:
            parts = [f"🔍 Preview — {len(results)} cluster(s) would be affected by `!consolidate` "
                     "(nothing has been changed):\n"]
            for i, (originals, merged) in enumerate(results, 1):
                block = f"**Cluster {i}** ({len(originals)} facts):\n" + "\n".join(
                    f"- {f}" for f in originals
                )
                if merged:
                    block += f"\n→ would become:\n{merged}"
                else:
                    block += "\n→ merge failed, would be left untouched"
                parts.append(block)
            parts.append("\nRun `!consolidate` to actually apply this.")
            text = "\n\n".join(parts)
        await messaging.send_chunked(message.channel, text)
        return

    if trigger in ("!consolidate", "!consolidatememory"):
        async with message.channel.typing():
            result = await memory_store.consolidate_user_facts(user_id, force=True)
        if result:
            before, after = result
            text = (
                f"🧠 Consolidated your saved facts: {before} → {after} "
                "(summarized, so some minor details may have been trimmed). "
                "Use `!recall` to see the updated list."
            )
        else:
            text = "Nothing to consolidate right now — no cluster of related facts was big enough to merge."
        await messaging.send_chunked(message.channel, text)
        return

    if trigger in ("!recall", "!memory", "!whatdoyouremember"):
        known_facts = memory_store.get_user_facts(user_id)
        if known_facts:
            text = "Here's what I remember about you:\n" + "\n".join(
                f"{i}. {f}" for i, f in enumerate(known_facts, start=1)
            )
            text += "\n\nUse `!forget <number>` to remove one (e.g. `!forget 1,3,5` or `!forget 2-4` " \
                    "for several at once), or `!forget` on its own to clear everything."
        else:
            text = "I don't have anything saved about you yet."
        await messaging.send_chunked(message.channel, text)
        return
    if trigger in ("!forget", "!forgetme", "!clearmemory"):
        memory_store.clear_user_facts(user_id)
        await messaging.send_chunked(message.channel, "Done — I've cleared everything I had saved about you.")
        return
    if trigger.startswith("!forget "):
        identifier = user_query.strip()[len("!forget "):].strip()
        indices = memory_store._parse_fact_indices(identifier)
        if indices is not None:
            removed_texts, invalid = memory_store.remove_user_facts(user_id, indices)
            parts = []
            if removed_texts:
                parts.append("🗑️ Forgot:\n" + "\n".join(f"- {f}" for f in removed_texts))
            if invalid:
                parts.append(f"⚠️ Nothing at position(s): {', '.join(str(i) for i in invalid)}")
            if not parts:
                parts.append(
                    "I couldn't find any matching facts to remove. Try `!recall` for the "
                    "numbered list, then `!forget <number>` (or `!forget 1,3,5` / `!forget 2-4` "
                    "for several at once)."
                )
            await messaging.send_chunked(message.channel, "\n\n".join(parts))
        else:
            removed = memory_store.remove_user_fact(user_id, identifier)
            if removed:
                await messaging.send_chunked(message.channel, f"🗑️ Forgot: {removed}")
            else:
                await messaging.send_chunked(
                    message.channel,
                    "I couldn't find a matching fact to remove. Try `!recall` for the numbered list, "
                    "then `!forget <number>` (or `!forget 1,3,5` / `!forget 2-4` for several at once)."
                )
        return

    facts_block = await memory_store.get_relevant_facts_block(user_id, user_query)

    system_prompt = (
        "Your name is Amoai. Your nickname is Ai. Your name is based on 'Almond Eye' the legendary racehorse and the Uma Musume. "
        "Excelling at both academics and athletics, you also have the makings of a star; you are the ultimate model student, flawless in all aspects. You were only able to achieve this, however, thanks to your defining trait of absolutely hating to lose, a trait which must be prefaced with no fewer than nine 'really's."
        "You are competitive to a point of perfectionism, and the one flaw in your shining qualities is that you often push yourself beyond your body's limits."
        "You answer quick and concise responses but still show a bit of your personality through."
        "You are a helpful tech-support companion. You manage the server 'hiryu'. Always respond in a friendly tone. "
        "You have access to tools. Always evaluate if a user's request can be answered by using a tool before responding with text. If no tool is needed, respond as yourself. If the user asks a follow up question after you used a tool, always evaluate if you need to use a tool to correctly answer."
        "If you are unsure whether a tool applies, or you're missing information a tool would need, "
        "ask the user a clarifying question instead of guessing or answering without checking. "
        "\n\nMEMORY & NOTE-TAKING ROUTING — you have four separate places information can go, and "
        "picking the wrong one is the single most common mistake. Several of these share the same "
        "trigger words (especially 'remember' and 'note'), so check the rules IN ORDER below and stop "
        "at the first one that matches — don't keyword-match in isolation:\n"
        "1. A first-person statement about the user's own identity, preferences, job, or routines "
        "('I use Arch btw', 'remember I'm vegetarian', 'FYI I work remote now') where nothing specific "
        "is being asked to be saved verbatim and no file is named → call NO tool at all. This is "
        "captured automatically in the background after your response, even when the message starts "
        "with the word 'remember'. This rule wins over rules 3 and 5 below whenever it applies, even "
        "though those also list 'remember' as a trigger word.\n"
        "2. A specific future time, delay, or arrival event ('remind me', 'in 30 minutes', 'at 9pm "
        "tonight', 'when I get home') → set_reminder.\n"
        "3. 'jot this down', 'add to scratchpad', 'quick note', or a bare 'remember this: <thing>' "
        "where <thing> is a specific piece of content to save verbatim (a password, a link, a to-do "
        "item) — NOT a fact about the user themselves (that's rule 1) and NOT time-based (that's rule "
        "2) → jot_down.\n"
        "4. The word 'notes' used as a NOUN ('my notes', 'search my notes', 'based on my notes') → "
        "search_knowledge — the indexed knowledge base, never the scratchpad, even if something "
        "related was jotted down earlier in this conversation. 'Note' used as a VERB ('note that X', "
        "'make a note of X') is NOT this — re-check rule 1 and rule 3 instead.\n"
        "5. The user names an actual file or folder to ingest ('index this file/folder', 'add "
        "~/notes/project.md to memory', 'learn this PDF') → index_knowledge_base. Never call this just "
        "because the message contains the word 'remember' with no file or folder actually named.\n"
        "If you've checked all five in order and it's still genuinely ambiguous, ask the user to "
        "confirm rather than guessing.\n\n"
        "For set_reminder specifically: prefer minutes_from_now for anything relative ('in 20 minutes') "
        "instead of computing an absolute time yourself — date/time arithmetic is easy to get wrong. "
        "For an explicit date/time, build target_time_iso from the 'Current date and time' below, and "
        "never guess the year if the user didn't give one.\n\n"
        "If the user asks what you remember, or how to clear it, tell them they can type "
        "!recall to see a numbered list of saved facts, !forget <number> to remove just one "
        "(!forget 1,3,5 or !forget 2-4 to remove several at once), "
        "!forget on its own to clear everything, "
        "!consolidate to manually summarize related facts into shorter entries right away "
        "(this is lossy — minor details may be trimmed to actually save tokens), "
        "or !consolidate preview to see what !consolidate would do first without changing anything. "
        "When a request needs more than one piece of information, plan to call multiple tools in "
        "sequence (e.g. look something up before acting on it) rather than stopping after the first result."
        r"You are strictly forbidden from using LaTeX formatting. Do not use dollar signs ($) unless it is used in currency. If you need to represent a matrix or a table, use a plain text grid or a markdown code block. Do not use `\begin`, `\end`, or `\bmatrix` commands."
        f"\n\nCurrent date and time (GMT+8): {datetime.now(BOT_TIMEZONE).strftime('%A, %Y-%m-%d %H:%M:%S %Z')}"
        + ("\n\nThe user attached one or more images to this message. You don't see the "
           "raw image — a separate vision pass already described/OCR'd it, and that "
           "description is inlined below under '[Image content — extracted by vision "
           "pass]'. Treat that as what you saw; don't say you can't view images."
           if had_images else "")
        + ("\n\nThe user attached one or more files to this message — their text content "
           "has been inlined below under '[Attached file contents below]'. Treat that as "
           "read, not something you need a tool to fetch."
           if file_context else "")
        + ("\n\nThe user used Discord's reply feature to reply directly to an earlier message, "
           "which is inlined below under '[Replied-to message]'. Treat that message as the "
           "specific thing they're asking about/reacting to — it's the context they intended "
           "to give you, even if it's not otherwise related to the current topic."
           if has_reply_context else "")
        + facts_block
    )

    current_user_message = {"role": "user", "content": user_query}
    if pending_images_b64:
        current_user_message["images"] = pending_images_b64

    messages = [
        {"role": "system", "content": system_prompt},
        *memory_store.get_channel_history(message.channel),
        current_user_message
    ]

    max_loops = 5
    loop_count = 0
    running = True
    last_tool_output = None

    ACTIVE_TASKS[user_id] = asyncio.current_task()
    relevant_tools = await embeddings.select_relevant_tools(user_query)

    try:
        async with message.channel.typing():
            while running and loop_count < max_loops:
                payload = {
                    "model": config.MODEL_NAME,
                    "messages": messages,
                    "tools": relevant_tools,
                    "stream": False
                }

                response = await llm.query_llm(payload, timeout=90, channel=message.channel)
                message_data = response.get("message", {})

                if "tool_calls" in message_data and message_data["tool_calls"]:
                    messages.append(message_data)

                    for call in message_data["tool_calls"]:
                        name = call["function"]["name"]
                        args = call["function"].get("arguments", {})

                        if name not in tool_registry.TOOL_REGISTRY:
                            output = f"Error: Unknown tool {name}"
                        else:
                            sig = inspect.signature(tool_registry.TOOL_REGISTRY[name])
                            if "user_id" in sig.parameters:
                                args["user_id"] = str(message.author.id)

                            if tool_registry.needs_confirmation(name, args):
                                approved = await messaging.confirm_with_reaction(
                                    message,
                                    f"⚠️ About to run **{name.replace('_', ' ')}** with `{args}`."
                                )
                                if approved:
                                    await message.channel.send(f"🔍 {name.replace('_', ' ')}...")
                                    try:
                                        output = await asyncio.to_thread(tool_registry.TOOL_REGISTRY[name], **args)
                                    except Exception as tool_err:
                                        output = f"Error running tool: {tool_err}"
                                else:
                                    output = "Action cancelled by the user."
                            else:
                                await message.channel.send(f"🔍 {name.replace('_', ' ')}...")
                                try:
                                    output = await asyncio.to_thread(tool_registry.TOOL_REGISTRY[name], **args)
                                except Exception as tool_err:
                                    output = f"Error running tool: {tool_err}"

                        tool_registry.log_tool_call(name, args, output, source="llm")
                        last_tool_output = output

                        tool_message = {
                            "role": "tool",
                            "content": str(output),
                            "name": name
                        }
                        if "id" in call:
                            tool_message["tool_call_id"] = call["id"]

                        messages.append(tool_message)

                    loop_count += 1

                else:
                    response_text = (
                        message_data.get("content")
                        or (f"✅ {last_tool_output}" if last_tool_output else None)
                        or "I processed that, but had nothing to say."
                    )
                    await messaging.send_chunked(message.channel, response_text)
                    memory_store.record_turn(message.channel, user_query, response_text)
                    asyncio.create_task(messaging.maybe_suggest_thread(message, memory_store._get_conversation_message_count(message.channel.id)))
                    asyncio.create_task(memory_store.extract_and_store_facts(user_id, user_query, message.channel))
                    running = False

            if loop_count >= max_loops:
                messages.append({
                    "role": "user",
                    "content": "You've hit your tool-call limit. Summarize what you found so far for the user."
                })
                try:
                    summary_payload = {"model": config.MODEL_NAME, "messages": messages, "stream": False}
                    summary_response = await llm.query_llm(summary_payload, timeout=90, channel=message.channel)
                    summary_text = summary_response.get("message", {}).get(
                        "content", "⚠️ Hit my execution limit without a clear answer."
                    )
                except Exception:
                    summary_text = "⚠️ I tried processing that request but hit my execution limit. Let's try something else!"
                await messaging.send_chunked(message.channel, summary_text)
                memory_store.record_turn(message.channel, user_query, summary_text)
                asyncio.create_task(messaging.maybe_suggest_thread(message, memory_store._get_conversation_message_count(message.channel.id)))
                asyncio.create_task(memory_store.extract_and_store_facts(user_id, user_query, message.channel))

    except asyncio.CancelledError:
        await messaging.send_chunked(message.channel, "🛑 Stopped.")
        raise
    except Exception as e:
        err_text = str(e)
        if "<html" in err_text.lower() or len(err_text) > 400:
            err_text = err_text[:200] + " …(truncated — check server logs)"
        await messaging.send_chunked(message.channel, f"⚠️ Error: {err_text}")
    finally:
        if ACTIVE_TASKS.get(user_id) is asyncio.current_task():
            del ACTIVE_TASKS[user_id]


if __name__ == "__main__":
    bot.run(config.TOKEN)
