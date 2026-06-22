#!/usr/bin/env bash
# Launch the VoxPolish web UI (assumes setup.sh has been run).
set -e

if [ -f venv/bin/activate ]; then
  source venv/bin/activate
else
  source venv/Scripts/activate
fi

python app.py
