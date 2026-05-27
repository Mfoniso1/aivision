#!/bin/bash
# ======================================================================
# INTENT-EYE AI // RASPBERRY PI KINETIC LAUNCHER
# ======================================================================

# Navigate to the script's home directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo -e "\033[1;36m=== INTENT-EYE AI KINETIC LAUNCHER ===\033[0m"

# 1. Initialize pigpio hardware-timed PWM daemon for MG995 servos
if command -v pigpiod &> /dev/null; then
    if ! pgrep -x "pigpiod" > /dev/null; then
        echo -e "\033[1;33m[SYSTEM] pigpiod not running. Launching hardware PWM daemon...\033[0m"
        sudo pigpiod
        sleep 1
    else
        echo -e "\033[1;32m[SYSTEM] pigpiod hardware PWM daemon is active.\033[0m"
    fi
else
    echo -e "\033[1;31m[WARNING] pigpiod daemon not found! MG995 servos will fall back to software PWM (may introduce mechanical jitter).\033[0m"
    echo -e "\033[1;31m          To fix, run: sudo apt install pigpio\033[0m"
fi

# 2. Check and activate local virtual environment
if [ -d "venv" ]; then
    echo -e "\033[1;32m[SYSTEM] Activating virtual environment 'venv'...\033[0m"
    source venv/bin/activate
else
    echo -e "\033[1;31m[WARNING] Virtual environment 'venv' not detected in current directory!\033[0m"
    echo -e "\033[1;33m          Attempting to run with system-wide python interpreters...\033[0m"
fi

# 3. Start uvicorn backend
echo -e "\033[1;36m[SYSTEM] Starting optical tracking server and dashboard stream...\033[0m"
python backend/app.py
