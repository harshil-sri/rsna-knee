#!/usr/bin/env bash
set -euo pipefail

if [ -n "${H100_HOST:-}" ]; then
    HOST="$H100_HOST"
elif [ -f "$HOME/h100.host" ]; then
    HOST="$(cat "$HOME/h100.host" | tr -d '[:space:]')"
else
    echo "Error: H100_HOST env var not set and $HOME/h100.host not found." >&2
    exit 1
fi

ARGS="$*"
CMD="cd ~/rsna-knee && . .venv/bin/activate && python3 -m src.infer $ARGS"

ssh "$HOST" "tmux new-session -d -s infer $(printf '%q' "$CMD")"
echo "Inference launched in tmux session 'infer' on $HOST"
