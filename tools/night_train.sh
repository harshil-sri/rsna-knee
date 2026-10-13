#!/usr/bin/env bash
set -uo pipefail
cd /home/harshil/kaggle/rsna-knee
mkdir -p logs runs/v0
S=logs/TRAIN_STATUS

echo "RUNNING train fold0 $(date '+%H:%M:%S')" > "$S"
python3 -m src.train --fold 0 --epochs 10 > logs/train_fold0.log 2>&1 \
  || { echo "FAIL train $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING infer $(date '+%H:%M:%S')" > "$S"
python3 -m src.infer --ckpt checkpoints/best.pt --out runs/v0/preds.csv > logs/infer_v0.log 2>&1 \
  || { echo "FAIL infer $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING score $(date '+%H:%M:%S')" > "$S"
python3 score_gold.py --train /mnt/ssd/rsna-knee/data/train.csv --pred runs/v0/preds.csv > logs/score_v0.log 2>&1 \
  || { echo "FAIL score $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "DONE $(date '+%H:%M:%S')" > "$S"
