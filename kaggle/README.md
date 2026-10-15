# RSNA Knee — Kaggle Notebook Pipeline (runbook for the team)

Local training hit a storage wall (full dataset = 569 GB; only ~115 GB cached locally).
The competition is a **code competition** (hidden test revealed at submit time), so the
final submission MUST run as a Kaggle notebook anyway. Everything trains/infers on Kaggle;
the laptop is for prototyping on the local cache only.

## Assets

| asset | where | status |
|---|---|---|
| competition data (full 569 GB train) | auto-attached in notebooks | ready |
| `dinov2-timm-cache` (HF/timm weights, ~100 MB) | Kaggle dataset | **needs one-time upload (script below)** |
| `rsna-knee-weak-labels` (train_weak.csv + train_weak_v2.csv) | Kaggle dataset | published |
| `rsna-knee-embeddings-v1` (~150 MB) | NB1 output | run NB1 → publish |
| `rsna-knee-ckpt-v1` (best.pt ~25 MB) | NB2 output | run NB2 → publish |

Account: `harshilsrivastav` (Kaggle API token at `~/.kaggle/access_token` works with
`export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)` for CLI/API).

## One-time: build the DINOv2 weights dataset (~100 MB)

```bash
python3 tools/build_timm_cache.py          # writes kaggle-out/dinov2-timm-cache.tar
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
mkdir -p /tmp/opencode/dinov2-cache && cd /tmp/opencode/dinov2-cache
tar xf /home/harshil/kaggle/rsna-knee/kaggle-out/dinov2-timm-cache.tar
cat > dataset-metadata.json <<'EOF'
{"title": "dinov2-timm-cache", "id": "harshilsrivastav/dinov2-timm-cache", "licenses": [{"name": "CC0-1.0"}]}
EOF
kaggle datasets create -p .
```

Needs ~120 MB free on the main drive (deferred until storage is cleared).

## Notebook sequence

1. **Upload `kaggle/nb1_features.ipynb`** → Kaggle → New Notebook → File → Import.
   - Accelerator: **GPU T4 x2**. Internet: **OFF** (required for comp data).
   - Add data: competition `rsna-knee-abnormality-detection` + dataset `dinov2-timm-cache`.
   - Run (~1-1.5 h: 819k DICOMs → 4407 × [42, 384] fp16 embeddings ≈ 150 MB).
   - Save Version → Quick Save. Then "New Dataset" from the output → name `rsna-knee-embeddings-v1`.
2. **Upload `kaggle/nb2_train.ipynb`** → GPU (T4 x2), Internet OFF.
   - Add data: competition data + `rsna-knee-embeddings-v1` + `rsna-knee-weak-labels`.
   - Run (~10 min: head-only training on embeddings). Saves `best.pt` + prints val AUC.
   - Save Version → publish output as dataset `rsna-knee-ckpt-v1`.
3. **Fork `kaggle/nb3_submit.ipynb`** (per teammate) → GPU, Internet OFF.
   - Add data: competition data + `rsna-knee-ckpt-v1` + `dinov2-timm-cache`.
   - Verify it runs on the 3 public test studies → **Submit to competition** (reruns on hidden test).

## Quotas / gotchas

- Kaggle GPU: ~30 h/week per account — 3 accounts ≈ 90 h/week pooled. Sessions cap ~12 h (fine for NB1/NB2; NB3 inference ~10 min).
- Internet is permanently OFF for competition notebooks — that's why weights ship as a dataset.
- If timm's `pretrained=True` fails offline in NB1/NB3: the HF_HOME env points into the dataset; check the cache layout matches `hf/` + `torch/` (build_timm_cache.py preserves it).
- Never write big intermediates to `/kaggle/working` outside final outputs (20 GB version limit).

## Local laptop role

- Prototype/debug pipeline logic on the 115 GB local cache (`/mnt/ssd/rsna-knee/data`).
- Local baseline runs (v2 = 0.5447 gold-58 MACRO, `checkpoints/best_v2.pt`) calibrate whether
  the Kaggle recipe is learning — compare NB2 val AUC against local equivalents.
- No further large downloads. The 454 GB full-set download is CANCELLED (was for local
  training, which is off the table).
