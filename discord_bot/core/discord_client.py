"""
discord_client.py — the shared discord.py Bot client instance.

Split out from bot.py so that other core/ modules (currently: messaging.py,
for confirm_with_reaction's bot.wait_for) can reach the live client without
importing bot.py itself. bot.py imports FROM messaging.py, tool_registry.py,
etc. — if those modules had to import the client back out of bot.py, that
would be a circular import. Nothing in this file imports from bot.py, and
nothing here should ever need to.
"""
import discord
from discord.ext import commands

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="", intents=intents)
