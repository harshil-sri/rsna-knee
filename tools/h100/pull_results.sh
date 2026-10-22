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

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

mkdir -p "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/runs"

rsync -avz --include="best.pt" --include="last.pt" --exclude="*" "$HOST:~/rsna-knee/checkpoints/" "$PROJECT_ROOT/checkpoints/" || true
rsync -avz "$HOST:~/rsna-knee/runs/" "$PROJECT_ROOT/runs/" || true
rsync -avz "$HOST:~/rsna-knee/levers*.csv" "$PROJECT_ROOT/" || true

echo "=========================================================="
echo "REMINDER: Remote storage quota is 100GB."
echo "Free remote quota by running:"
echo "  ssh $HOST 'rm -rf ~/rsna-knee/data ~/rsna-knee/.venv'"
echo "=========================================================="
