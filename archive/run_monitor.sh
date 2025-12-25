#!/bin/bash

while true; do
    echo "Checking trades at $(date)"
    python3 monitor_trades.py
    echo "Next check in 1 hour..."
    sleep 3600
done
