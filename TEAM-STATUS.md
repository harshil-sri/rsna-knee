# Team Legacy — status update for Daksh & Kartik

**Written 2026-10-16 ~14:25 IST by Harshil (orchestrator).** Self-contained — paste this whole file into your own LLM to get up to speed. Deadline is close; read the "your lane" section only if short on time.

## Competition facts
- Kaggle **code competition**: `rsna-knee-abnormality-detection`. **Deadline Oct 22, 23:59 UTC.**
- Metric: macro ROC-AUC across 12 findings. Hidden rerun: the submitted kernel is re-executed by Kaggle on the full hidden test set (~1,300 studies); it must write `submission.csv` (StudyInstanceUID + 12 finding columns).
- **5 code submissions/day per team** (shared pool — coordinate before ANY member submits). LB shows your **best** score, not the latest.
- Final selection: exactly **2 kernels**. Frozen rule: `[best-AUC kernel, strip-c384 kernel]`.
- Team: **Harshil** (orchestrator, fusion, all submissions), **Daksh** (own model legs + pseudo-label loop), **Kartik** (GPU provider + read-only inference levers).
- Kaggle GPU quota: **2 concurrent GPU kernels per account, ~30 GPU-h/week per account.** We spread work across Harshil's + Kartik's accounts.

## Leaderboard reality (Oct 16, verified from Kaggle data)
- **6,071 teams.** Top = **0.968**. Gold (#10) = **0.964**. Silver (#304) AND bronze (#607) lines are **both 0.957**.
- Fork plateau: **1,400+ teams at 0.955–0.960** running copied public legs — their errors are fully correlated, so the private shake-up will reshuffle that blob. 71 teams ≥0.960 have real edges.
- **Legacy today: 0.950** (3-leg public fork), public rank ~#1,780.

## Efficiency prize track (decoded from the official LB — 5,955 rows)
- Metric (minimize): `eff = AUC/(benchmark − maxAUC) + RuntimeSeconds/32400`. **1 hour of hidden runtime ≡ 0.05 AUC.**
- Top-3 today: 0.964 / 0.961 / 0.959 AUC at ≲30 min runtime. The frontier demands fast AND strong.
- Our **strip c384 (0.949, ~20 min)** projects efficiency rank ~#85–160 (**top ~2%** of all submissions).
- **Frozen rule: strip c384 is our efficiency selection. Never select a heavy fusion as the efficiency pick.**

## Running right now (Oct 16)
| kernel | account | status | ETA | purpose |
|---|---|---|---|---|
| `rsna-knee-nb4-train25d` (2.5D fold-0) | Harshil | RUNNING | ~15:30 IST | our unique 2.5D leg family; gold gate ≥0.80 macro |
| `rsna-knee-helios-957` (helios fork, v2) | Harshil | hidden rerun submitted ~14:30 | score ~22:00 IST | fork ceiling ≈0.957 |
| `rsna-knee-versia7` (versia-7 mega-fusion) | Harshil | queued for GPU slot | launch ~14:30–15:30; score tonight | alt fusion ≈0.956 |
| `fullfork v9` (cnxt decoder fixed) | Harshil | score pending | ~21:00 IST | expect ~0.950–0.954 |

## Daksh — your lane
1. **OOF incoming**: after all four 2.5D folds finish (fold-1 tonight; folds 2–3 tonight on Kartik's GPU), Harshil publishes **`oof_preds_25d.csv`** as a Kaggle dataset — **4,407 rows: StudyInstanceUID + 12 finding columns, fold-blind OOF**. ETA **Oct 17 ~09:00 IST**. Your pseudo-label v2 consumes this.
2. **Pseudo-label v2 retrain**: replace v1 weak labels with calibrated OOF ranks where confident (e.g., rank >0.9/<0.1 per finding); retrain your leg(s) once (one GPU pass). Rank-calibrate per finding before thresholding.
3. **GPU**: use **your own Kaggle account** (create one if needed — free, ~30 GPU-h/week, 2 concurrent slots). That is not a limitation for one retrain pass (~2–6 h). Harshil does **not** need your API token. If you truly can't get an account, Kartik's frees after Oct 17 morning.
4. **Dataset shares you'll need (ask Harshil in chat)**: `harshilsrivastav/rsna-knee-labels-v1`, `harshilsrivastav/rsna-knee-weak-labels`, `harshilsrivastav/rsna-knee-embeddings-v1`.
5. **Deliverable: v2 inference kernel or prediction dataset by Oct 19, 12:00 IST** → Harshil re-fuses Oct 19–20. If you have any earlier leg improvement in flight, tell Harshil before Oct 17 evening so GPU slots get sequenced.

## Kartik — your lane
1. **Token**: Harshil holds it (chmod 600, never in chats/files). **Do not paste it anywhere.**
2. **Dataset shares needed** (Harshil shares via UI → Sharing → invite your Kaggle username; your username ka1205 — confirmed shared & accessible): `harshilsrivastav/rsna-knee-labels-v1`, `harshilsrivastav/rsna-knee-weak-labels`, `harshilsrivastav/rsna-knee-ckpt-v1`, `harshilsrivastav/dinov2-timm-cache`, `harshilsrivastav/convnext-tiny-weights`.
3. **Tonight: your 2 GPU slots run nb4 folds 2–3** (Harshil pushes the kernels under your account; ~6.5 h each, launched as soon as the shares land — target 16:00–18:00 IST tonight, else tomorrow morning). That's ~13 h of your ~30 weekly GPU-h. **Nothing else is required from you today.**
4. Oct 18–19: possibly one backup kernel slot. Final freeze Oct 21–22 is Harshil's job. Your scope stays **read-only inference-side levers** — don't modify the frozen handoff package.

## Fusion plan (Harshil, Oct 18–20)
- Legs: best public fork legs + our 2.5D 4-fold + Daksh v2 → **per-finding weight grid {0, 0.25, 0.5, 0.75}** tuned on the 58 gold studies, shrunk toward uniform (LOO-validated so we don't overfit 58 studies). Target public **0.958–0.961**.
- Oct 21–22: freeze, full pre-deadline rerun of both final kernels, select `[best-AUC, strip-c384]`.

## Standing rules
- All submissions go through Harshil — the 5/day limit is **per team**, and LB = best, not latest.
- Never paste API tokens in chat, files, or commits.
- Kaggle quirks: datasets auto-extract `.zip` at ingest (uploaded subdirectories are dropped — upload zips, not folders); max 2 concurrent GPU kernels/account; always verify with `kaggle kernels status owner/slug` before assuming success.
- Questions/status → team chat. Harshil updates this file when something material changes.
