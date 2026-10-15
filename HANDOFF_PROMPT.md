# Handoff — RSNA Knee (paste into a new session)

You are the orchestrator for Harshil's RSNA Knee Abnormality Detection entry (team of 3, incl. 2 friends). Repo: `/home/harshil/kaggle/rsna-knee`. Today is Oct 15, 2026. **Team merge is TODAY; competition final Oct 22.** Work autonomously unless blocked.

## Read first (in this order — don't re-derive)
1. `/home/harshil/orchestrator/sessions/2026-10-15-rsna-knee.md` — session digest (yesterday's full arc + ops lessons)
2. `/home/harshil/kaggle/rsna-knee/HANDOFF.md` — chronological state log (skim the latest 4 bullets)
3. `/home/harshil/kaggle/rsna-knee/levers.csv` — every experiment + result
4. `/home/harshil/kaggle/rsna-knee/kaggle/README.md` — Kaggle pipeline runbook (the current plan)

## Where we are (context compressed)
- Competition: `rsna-knee-abnormality-detection`, **code competition** (public test = 3 studies with no images; hidden test revealed only at submit → final submission MUST run as a Kaggle notebook). 12 findings per study, scored MACRO ROC-AUC.
- **Pivot (done):** all training/inference moved to Kaggle notebooks. Local full-dataset download CANCELLED (454 GB was still missing; storage can't fit; would be useless for submission). Local role = prototyping on 115 GB cache only.
- Baselines (gold-58 MACRO, `score_gold.py`): chance 0.5073 → **v2 = 0.5447** (`checkpoints/best_v2.pt`, 2ep weak labels) → v3 = 0.5012 (4ep overfit — more epochs ≠ better; don't trust val-AUC-on-soft-labels, only score_gold counts).
- Weak labeler v2 = `data/train_weak_v2.csv`, parser-vs-gold MACRO **0.7651** — the label quality lever for everything.
- Kaggle assets: notebooks `nb1_features / nb2_train / nb3_submit` (committed `b5eb0c1`) + `kaggle/README.md` runbook; dataset `harshilsrivastav/rsna-knee-weak-labels` PUBLISHED; `dinov2-timm-cache.tar` (176 MB, offline-load verified) at `kaggle-out/` — **needs upload as dataset**.

## Next moves (in order)
1. Upload `kaggle-out/dinov2-timm-cache.tar` as Kaggle dataset `harshilsrivastav/dinov2-timm-cache` — exact CLI steps in `kaggle/README.md` (needs `export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)`; username `harshilsrivastav`; CLI 2.2.4 installed).
2. Guide Harshil to upload+run **NB1** (T4 x2, internet OFF, attach comp data + timm cache; ~1-1.5 h) → publish output as `rsna-knee-embeddings-v1`. NB1 windowing logic was smoke-tested locally on real public-test DICOMs.
3. Run **NB2** (attach embeddings + weak-labels) → head-only training ~10 min → val AUC + `best.pt` → publish as `rsna-knee-ckpt-v1`. Compare val AUC against the local equivalent; iterate recipe if < ~0.55-ish equivalent.
4. Fork **NB3**, verify it runs on the 3 public test studies (the 557-file listing is at `/tmp/opencode/test_files.txt`; 12 sample slices already at `/mnt/ssd/rsna-knee/data/test_series_dl`) → **submit** (first submission = sanity, score unknown until leaderboard).
5. Then iterate for the merge meeting: labeler v3, backbone/mil upgrades, TTA, folds. All experiments must keep the Kaggle notebook ↔ local `src/` code in sync (source of truth: `src/windows.py`, `src/model.py`, `src/dataset.py`).

## Hard-won gotchas (do not relearn these)
- Kaggle: internet OFF in comp notebooks (weights ship as datasets); 12 h session cap; ~30 GPU-h/week/account (3 accounts ≈ 90); outputs 20 GB version limit; file-listing API pages only 20 rows (`--page-token`).
- This WSL: tmux server dies spontaneously killing children → launch long runs with `setsid nohup bash ... > log 2>&1 &`; `pkill` needs bracket trick (`-f "[s]rc.train"`); vhdx storage crashes come from the Windows host, not `df` inside WSL.
- Delegation: `@coder` rung-1 sometimes fabricates output or refuses on fake LSP errors → fall back to `@coder-2` with "verification bar = py_compile + real end-to-end run with pasted output". LSP/pandas-stub errors in this repo are noise.
- Local data: `/mnt/ssd/rsna-knee/data` (149 GB free, NEVER fill it; other projects live there too — do not delete anything). Big temp files go to `/tmp/opencode`.
- Git: local commits only, NEVER push. Commit after each result.
- H100 kit exists but is blocked on SSH host details — don't block on it.

## Token discipline
Start from the digest; don't re-read logs unless a command fails. Batch tool calls. Keep runs on `setsid nohup` + poll TRAIN_STATUS. Update `levers.csv` + `HANDOFF.md` + digest after each result, then commit.
