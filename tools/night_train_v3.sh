#!/usr/bin/env bash
set -uo pipefail
cd /home/harshil/kaggle/rsna-knee
mkdir -p logs runs/v3
S=logs/TRAIN_STATUS

echo "RUNNING train fold0 v3 $(date '+%H:%M:%S')" > "$S"
python3 -m src.train --fold 0 --epochs 4 --no-gold-only \
  --labels /mnt/ssd/rsna-knee/data/train_weak.csv > logs/train_fold0_v3.log 2>&1 \
  || { echo "FAIL train v3 $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING infer v3 $(date '+%H:%M:%S')" > "$S"
python3 -m src.infer --ckpt checkpoints/best.pt --out runs/v3/preds.csv > logs/infer_v3.log 2>&1 \
  || { echo "FAIL infer v3 $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "RUNNING score v3 $(date '+%H:%M:%S')" > "$S"
python3 score_gold.py --train /mnt/ssd/rsna-knee/data/train.csv --pred runs/v3/preds.csv > logs/score_v3.log 2>&1 \
  || { echo "FAIL score v3 $(date '+%H:%M:%S')" > "$S"; exit 1; }

echo "DONE v3 $(date '+%H:%M:%S')" > "$S"