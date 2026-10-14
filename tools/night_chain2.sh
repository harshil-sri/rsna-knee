#!/usr/bin/env bash
set -uo pipefail
cd /home/harshil/kaggle/rsna-knee
DATA=/mnt/ssd/rsna-knee/data
S=logs/STATUS

echo "RUNNING select $(date '+%H:%M:%S')" > "$S"
python3 tools/download_subset.py --data-dir "$DATA" select --budget-gb 55 > logs/select.log 2>&1 \
  || { echo "FAIL select $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING download $(date '+%H:%M:%S')" > "$S"
python3 tools/download_subset.py --data-dir "$DATA" --workers 48 download > logs/download.log 2>&1 \
  || { echo "FAIL download $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING verify $(date '+%H:%M:%S')" > "$S"
python3 tools/download_subset.py --data-dir "$DATA" verify > logs/verify.log 2>&1 \
  || { echo "FAIL verify $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "DONE $(date '+%H:%M:%S')" > "$S"
