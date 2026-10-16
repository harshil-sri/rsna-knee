# Model Handoff — MIL Ensemble (public LB 0.950) — Harshil → Kartik

Delivery: `20261016T0205Z-mil-ensemble-pub0950`
Owner: Harshil Srivastav (harshilsrivastav on Kaggle)
Created: 2026-10-16 02:05 UTC (07:35 IST)

## SCOPE OF TRANSFER — NOT MODEL TAKEOVER

Per the team division of labor agreed on 2026-10-15: **Kartik's role is NOT model takeover.**
This is a read-only reference package for Kartik's bounded scope: **inference-side levers on
EXISTING models** (TTA, precision, window/view selection, runtime efficiency). Kartik does NOT
take ownership of model training, architecture, lever iteration, or strategy — those remain with
Harshil. Any change beyond inference-side scope requires explicit re-agreement between Harshil
and Kartik before work starts. Nothing in this package should be redesigned, retrained,
optimized, or reinterpreted; if a change seems needed, raise it with Harshil first.

## What this is

Our owned best model: **5-fold gated-attention MIL over DINOv2-S/14 embeddings**, trained on
Daksh's teacher soft labels (v1_targets) with canonical folds, scored **gold-58 macro AUC
0.7798** (strict OOF 0.7454) and **public LB 0.950** (submission 2026-10-16T00:01Z, crash-fixed
inference v5). It is the strongest *owned* artifact and the team's accuracy anchor for our track
(clean, no OAI). The highest public-LB artifact (0.949–0.954 fork) is **external** (see manifest
§external_data) and is NOT included as a trainable asset here.

## Quickstart

```bash
# 0. verify integrity (from this directory)
sha256sum -c SHA256SUMS

# 1. reproduce the gold-58 receipt (needs python3 + pandas + scikit-learn)
python3 source/score_gold.py --train artifacts/gold58_truth.csv --pred artifacts/gold_preds.csv
# expected: 58 matched, MACRO 0.7798 (see receipts/receipt-gold58-ensemble.txt)

# 2. reproduce inference on Kaggle (GPU, internet OFF)
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)   # token file stays local, never committed
mkdir -p /tmp/nb3-push && cp source/nb3_submit.ipynb configs/kernel-metadata-nb3.json /tmp/nb3-push/
kaggle kernels push -p /tmp/nb3-push   # → kernel harshilsrivastav/rsna-knee-nb3-submit-test
```

## Package map

- `HANDOFF_MANIFEST.yaml` — full manifest: files+hashes, Kaggle refs, contract, lineage, issues
- `SHA256SUMS` — hashes for every file in this package (self excluded)
- `source/` — `nb1_features.ipynb` (embeddings), `nb2_train.ipynb` (training, v9 ensemble),
  `nb3_submit.ipynb` (inference/submission), `score_gold.py`, `make_folds.py`
- `configs/` — exact kernel-metadata for nb2/nb3 (dataset/competition attachments)
- `receipts/` — score receipts, regenerated receipts, kernel run logs, public-LB JSON
- `artifacts/` — `best.pt` (5-fold ensemble checkpoint), `gold_preds.csv`, `oof_preds.csv`,
  `gold_score.json`, `gold58_truth.csv`
- `levers.csv` — iteration history (requested format `config,gold_auc,owner,date,notes`)
- `candidate_levers.csv` — isolated-change ledger with hashes
- `receipts/score_receipts.csv` — per-score receipts in the requested format

## Anchors (nominated)

- **accuracy_anchor**: `kaggle-nb3-mil-ensemble-v9-ckpt` — gold-58 macro AUC 0.7798 (58/58
  matched, receipt REC-001), public LB 0.950 (REC-005).
- **efficiency_anchor**: `kaggle-nb2-head-8ep-v1targets-folds` — gold-58 0.7690, single
  gated-attention head (≈1/5 head compute of the ensemble, same preprocessing); receipt REC-003.

## Known issues / fallback behavior (summary — full list in manifest)

- No series dir for a study → all-0.5 fallback row (submission shape preserved, no crash).
- Multi-series studies: round-robin window budget proportional to series size, cap 42 windows.
- MONOCHROME1 photometric inversion applied; slices ordered by ImagePositionPatient[2]
  (SliceLocation fallback); short series padded by repeating the last slice.
- Public-LB-visible rerun ran 3 studies in ~64 s; hidden rerun projected 4–6 h for ~1.3k
  studies (9 h cap) — projection, not a receipt.
- Kaggle switched to nested mount layout (`/kaggle/input/competitions/<comp>/`,
  `/kaggle/input/datasets/<owner>/`) on 2026-10-16; nb3 resolves paths by depth-agnostic glob
  and is verified robust. nb1/nb2 still use some flat-path globs — patch before any rerun
  (tracked in manifest known_issues).

## UNAVAILABLE items (with reason + ETA)

- v6 (efficiency anchor) prediction file — earlier kernel version logs/outputs not retained;
  score is receipt-backed only by levers.csv + NB2 v9 log lineage. ETA: reproducible on request
  by re-running `configs/kernel-metadata-nb2.json` with `EPOCHS=8` head-only config (~11 min).
- Dataset version numbers (Kaggle CLI exposes no version listing) — pinned instead by
  SHA-256 of the exact files and dataset creationDateTime; both recorded in the manifest.
- `candidate_levers` prediction hashes for LB candidates cand10/cand11 submission CSV of
  record: cand11 hash IS captured (`receipts/` pointer in manifest); cand10's submission.csv
  was not retained → UNAVAILABLE (kernel output downloadable on request).

## Explicitly excluded

Credentials/token files; competition DICOMs; raw reports (only Report-derived teacher
probabilities in v1_targets are included — no verbatim report text in this package beyond
`artifacts/gold58_truth.csv` label columns, which contain numeric labels only).
