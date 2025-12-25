import discord
from discord.ext import commands
import asyncio
import os
from src.discord_bot import setup


intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)


@bot.event
async def on_ready():
    print(f'Quantum Fractals logged in as {bot.user}')
    print(f'Ready to analyze fractals')


async def main():
    await setup(bot)
    
    TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    
    if not TOKEN:
        print("Error: DISCORD_BOT_TOKEN environment variable not set")
        print("Set it with: export DISCORD_BOT_TOKEN='your_token_here'")
        return
    
    async with bot:
        await bot.start(TOKEN)


if __name__ == "__main__":
    asyncio.run(main())
