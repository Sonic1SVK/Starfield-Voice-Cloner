#!/usr/bin/env bash
# Linux / macOS launcher: activates the virtual environment and starts the GUI.
set -e
cd "$(dirname "$0")"

if [ -f venv/bin/activate ]; then
    source venv/bin/activate
elif [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "No virtual environment found (venv/ or .venv/)."
    echo "Create one first:  python3 -m venv venv && source venv/bin/activate"
    echo "Then install PyTorch for your hardware and: pip install -r requirements.txt  (see README.md)"
    exit 1
fi

exec python qwen_gui.py "$@"
