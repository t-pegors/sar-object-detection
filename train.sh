#!/usr/bin/env bash
# Convenience wrapper — always runs from the project root with correct env
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load credentials
set -a && source .env && set +a

# Run training with project root on PYTHONPATH
PYTHONPATH="$SCRIPT_DIR" .venv/bin/python3.12 src/train.py "$@"
