#!/bin/bash
set -e
cd "$(dirname "$0")"
if [ -d .venv ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
elif [ -d venv ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi
HOST=${API_HOST:-0.0.0.0}
PORT=${FORAGE_PORT:-8003}
exec uvicorn ci_forage.api:app --host "$HOST" --port "$PORT" --reload
