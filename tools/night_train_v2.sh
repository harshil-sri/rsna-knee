#!/usr/bin/env bash
set -uo pipefail
cd /home/harshil/kaggle/rsna-knee
mkdir -p logs runs/v2
S=logs/TRAIN_STATUS

echo "RUNNING train fold0 v2 $(date '+%H:%M:%S')" > "$S"
python3 -m src.train --fold 0 --epochs 2 --no-gold-only \
  --labels /mnt/ssd/rsna-knee/data/train_weak.csv > logs/train_fold0_v2.log 2>&1 \
  || { echo "FAIL train v2 $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING infer v2 $(date '+%H:%M:%S')" > "$S"
python3 -m src.infer --ckpt checkpoints/best.pt --out runs/v2/preds.csv > logs/infer_v2.log 2>&1 \
  || { echo "FAIL infer v2 $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING score v2 $(date '+%H:%M:%S')" > "$S"
python3 score_gold.py --train /mnt/ssd/rsna-knee/data/train.csv --pred runs/v2/preds.csv > logs/score_v2.log 2>&1 \
  || { echo "FAIL score v2 $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "DONE v2 $(date '+%H:%M:%S')" > "$S"