#!/bin/bash
export DISCORD_BOT_TOKEN=$(grep DISCORD_BOT_TOKEN .env | cut -d '=' -f2)
python3 discord_bot_launcher.py
