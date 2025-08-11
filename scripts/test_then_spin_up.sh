#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from the project root regardless of where the script is invoked
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "==> Running test suite (scripts/run_tests.py) in an ephemeral container…"
docker run --rm -t \
  -v "$PWD":/app -w /app \
  python:3.12-slim bash -lc "
    python -m venv .ci-venv &&
    . .ci-venv/bin/activate &&
    pip install -U pip &&
    # Install backend deps and pytest so the runner doesn't need to --user install
    pip install -r requirements.api.txt pytest &&
    python scripts/run_tests.py
  "

echo "==> Tests passed. Building and starting the app (detached)…"
docker compose up --build -d