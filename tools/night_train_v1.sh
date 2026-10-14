#!/usr/bin/env bash
set -uo pipefail
cd /home/harshil/kaggle/rsna-knee
mkdir -p logs runs/v1
S=logs/TRAIN_STATUS

echo "RUNNING train fold0 v1 $(date '+%H:%M:%S')" > "$S"
python3 -m src.train --fold 0 --epochs 3 --no-gold-only > logs/train_fold0_v1.log 2>&1 \
  || { echo "FAIL train v1 $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING infer v1 $(date '+%H:%M:%S')" > "$S"
python3 -m src.infer --ckpt checkpoints/best.pt --out runs/v1/preds.csv > logs/infer_v1.log 2>&1 \
  || { echo "FAIL infer v1 $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING score v1 $(date '+%H:%M:%S')" > "$S"
python3 score_gold.py --train /mnt/ssd/rsna-knee/data/train.csv --pred runs/v1/preds.csv > logs/score_v1.log 2>&1 \
  || { echo "FAIL score v1 $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "DONE v1 $(date '+%H:%M:%S')" > "$S"
