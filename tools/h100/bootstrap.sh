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
LOCAL_DATA="/mnt/ssd/rsna-knee/data"

if [ -d "$LOCAL_DATA" ]; then
    DATA_SIZE_KB="$(du -sk "$LOCAL_DATA" | cut -f1)"
    if [ "$DATA_SIZE_KB" -gt 94371840 ]; then
        echo "WARNING: $LOCAL_DATA is larger than 90GB ($DATA_SIZE_KB KB). 100GB remote quota may be exceeded!" >&2
    fi
fi

ssh "$HOST" "mkdir -p ~/rsna-knee ~/rsna-knee/data ~/.kaggle"

rsync -a \
    --exclude "checkpoints" \
    --exclude "logs" \
    --exclude "data" \
    --exclude ".git" \
    --exclude "__pycache__" \
    --exclude ".venv" \
    "$PROJECT_ROOT/" "$HOST:~/rsna-knee/"

if [ -d "$LOCAL_DATA" ]; then
    rsync -a --info=progress2 "$LOCAL_DATA/" "$HOST:~/rsna-knee/data/"
fi

if [ -f "$HOME/.kaggle/access_token" ]; then
    scp "$HOME/.kaggle/access_token" "$HOST:~/.kaggle/access_token"
elif [ -f "$HOME/.kaggle/kaggle.json" ]; then
    scp "$HOME/.kaggle/kaggle.json" "$HOST:~/.kaggle/kaggle.json"
fi

ssh "$HOST" bash -s << 'EOF'
set -euo pipefail
cd ~/rsna-knee
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
EOF
