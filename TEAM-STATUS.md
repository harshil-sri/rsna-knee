# Team Legacy — status update for Daksh & Kartik

**Written 2026-10-16 ~17:05 IST by Harshil (orchestrator).** Supersedes the 14:25 version. Self-contained — paste this whole file into your own LLM to get up to speed. Deadline is close; read the "your lane" section only if short on time.

## Competition facts
- Kaggle **code competition**: `rsna-knee-abnormality-detection`. **Deadline Oct 22, 23:59 UTC.**
- Metric: macro ROC-AUC across 12 findings. Hidden rerun: the submitted kernel is re-executed by Kaggle on the full hidden test set (~1,300 studies); it must write `submission.csv` (StudyInstanceUID + 12 finding columns).
- **5 code submissions/day per team** (shared pool — coordinate before ANY member submits). **The day is a UTC day**: allowance resets at UTC midnight = **05:30 IST**, not local midnight. LB shows your **best** score, not the latest.
- **Efficiency-prize eligibility (settled from the official rules page Oct 16, verbatim):** a submission is eligible only if it is **among the 2 submissions selected by the team** (or Kaggle's auto-selection). Manual selection at freeze is **mandatory** — this killed the old "best public auto-qualifies" assumption. One submission **may win both** a Leaderboard Prize and an Efficiency Prize.
- Final selection: exactly **2 kernels**, chosen manually at Oct 21–22 freeze: **[best-AUC kernel, efficiency horse]**.
- Team: **Harshil** (orchestrator, fusion, all submissions), **Daksh** (own model legs + pseudo-label loop), **Kartik** (GPU provider + read-only inference levers).
- Kaggle GPU quota: **2 concurrent GPU kernels per account, ~30 GPU-h/week per account.** We spread work across Harshil's + Kartik's accounts.

## Leaderboard reality (Oct 16, verified from Kaggle data)
- **6,071 teams.** Top = **0.968**. Gold (#10) = **0.964**. Silver (#304) AND bronze (#607) lines are **both 0.957**.
- Fork plateau: **1,400+ teams at 0.955–0.960** running copied public legs — their errors are fully correlated, so the private shake-up will reshuffle that blob. 71 teams ≥0.960 have real edges.
- **Legacy today: 0.956 (NEW BEST)** — helios fork v2 (Raptor ⊕ 4-fold ConvNeXt-small, 50/50 rank blend) scored 17:37 IST. That is **0.001 below the bronze line (0.957)**. Full-fork 0.950 lineage closed; more forks add zero.

## Efficiency prize track (decoded from the official LB — 5,955 rows)
- Metric (minimize): `eff = −(AUC−0.5)/0.467 + RuntimeSeconds/32400` (normalizer = global **private** max; public 0.967 is our planning proxy). **+0.001 AUC ≡ 69 s runtime; 1 hour ≡ 0.05 AUC.**
- Top-3 today: 0.964 / 0.961 / 0.959 AUC at ≲30 min runtime. The frontier demands fast AND strong.
- Teams ≥0.957 grew 167 → ~600 in 11 h (fork flood) — the easy window is closing.
- Our **strip c384 (0.949, ~20 min)** projects efficiency rank ~#85–160 (**top ~2%** of all submissions).
- **REVISED Oct 16 (supersedes "never select heavy fusion for eff"):** the efficiency pick is a kernel **born fast**. The new fusion kernel is designed ≤25 min (2.5D fold-0 **solo** + strip leg), so if it scores ≥0.952 public it becomes BOTH the best-AUC pick and the efficiency pick. The old rule's intent survives: nothing >1 h in the efficiency-eligible slot.

## Running right now (Oct 16, 17:00 IST)
| item | account | status | ETA |
|---|---|---|---|
| `rsna-knee-nb4-train25d` (2.5D fold-0) | Harshil | RUNNING (long — overrun cause reads from its log at completion; per-epoch checkpoint saves survive any timeout) | verdict within hours; watch armed |
| helios fork v2 (57262514) | Harshil | **SCORED 0.956 — new team best** | done |
| `rsna-knee-versia7` (mega-fusion fork) | Harshil | interactive COMPLETE (3-study output verified healthy, both legs loaded, blend receipt clean) | auto-submits **05:35 IST Oct 17** at quota reset → score ~09:00–13:00 |
| strip-v2 (`pub954-strip` v2) | Harshil | **verified: predictions byte-identical to 0.949 baseline (MAXDIFF 0.0)** with fast decode + tf32 — this is the fast-decode base the fusion inherits | runtime measured at next real submission |
| `fullfork v9` | Harshil | scored **0.950** — cnxt decoder fix added ZERO → full-fork lineage closed | done |

- Sub budget today (UTC Oct 16): **5/5 used**. Next submissions possible **05:30 IST Oct 17** — versia-7 auto-submits 05:35 IST; it bundles helios legs + CoAtNet + stackers, expected ≥ 0.956.

## Daksh — your lane
1. **Dataset shares** (labels-v1, weak-labels, embeddings-v1, ckpt-v1, dinov2-timm-cache → `Dakxsh23`): being shared via Kaggle web UI today. Say so in chat if anything is missing after tonight.
2. **folds.csv integrity — CONFIRMED**: sha256 `8ea359a25c841c…` matches your ask, verified BOTH in local staging AND inside the published `rsna-knee-labels-v1` dataset that the fold kernels mount at runtime.
3. **nb2 with the mount-path patch — DELIVERED**: zip at Harshil's Downloads (`daksh-nb2-mountpatched.zip`) = patched notebook (all input lookups `recursive=True` for the nested `/kaggle/input/{competitions,datasets}/` layout), `kernel-metadata.json` pre-set to your account, README with push command. ⚠ The copy in the handoff folder's `source/` dir is the PRE-patch version — don't use that one. And remember: nb2 = our 0.756 weak-label MIL lineage (mount-hardening + head-architecture reference), **not** your v2 recipe.
4. **OOF incoming**: after all four 2.5D folds finish, Harshil publishes **`oof_preds_25d.csv`** as a Kaggle dataset — **4,407 rows: StudyInstanceUID + 12 finding columns, fold-blind OOF**. ETA **Oct 17 ~09:00 IST** (folds 2–3 run tonight on Kartik's GPU; fold-1 on Harshil's).
5. **Pseudo-label v2 retrain** (methodology REVISED Oct 16 — soft distillation, no hard cutoffs):
   - Targets = **continuous float OOF predictions** (0.0–1.0), train with `BCEWithLogitsLoss(logits, soft_targets)` — do NOT binarize at 0.9/0.1 (hard cutoffs destroy exactly the borderline ranking that macro-AUC scores).
   - **Gold override**: the 58 gold studies get their exact binary labels with `sample_weight=2.0`; silent/ambiguous weak-label cells get weight `0.0`; low-confidence weak labels (<0.20 confidence or extractor disagreement >0.75) masked `w=0`.
   - Retrain your leg(s) once (one GPU pass on your own account — you don't need Harshil's token).
6. **Deliverable: v2 inference kernel or prediction dataset by Oct 19, 12:00 IST** → Harshil re-fuses Oct 19–20. If a leg improvement is in flight, tell Harshil before Oct 17 evening so GPU slots get sequenced.

## Kartik — your lane
1. **Token**: Harshil holds it (chmod 600, never in chats/files). **Do not paste it anywhere.**
2. **Dataset shares**: all 5 confirmed shared & accessible under `ka1205`.
3. **Tonight: your 2 GPU slots run nb4 folds 2–3** (pushed under your account as soon as fold-0's gate passes — target tonight ~18:00–19:30 IST, else tomorrow morning). ~13 h of your ~30 weekly GPU-h. **Nothing else is required from you today.**
4. Oct 18–19: possibly one backup kernel slot. Final freeze Oct 21–22 is Harshil's job. Your scope stays **read-only inference-side levers** — don't modify the frozen handoff package.

## Fusion plan (Harshil, Oct 18–20) — REVISED Oct 16 after research audit
- **Per-finding weight tuning on the 58 gold studies is BANNED** (evidence from this competition: per-target weight selection lost to untuned fusion 0.8217 vs 0.8495 under 20×5-fold CV; rare findings have only 1–3 positives in gold-58 — tuned weights fit that one patient's noise and die on the ~910-study private split).
- **Primary sub = percentile rank averaging, uniform weights**: per finding, rank-normalize each leg's predictions to [0,1] (`(rankdata(p)-1)/(N-1)`), then average legs. Zero fitted parameters → maximally private-safe.
- **The fusion kernel = fold-0 SOLO + strip leg, ≤25 min rerun** (4-fold inference would cost 20–40 min and blow the budget — the 4-fold ensemble serves the OOF file and your v2 retrain instead). Leg failure = fail-closed (weight 0, survivors renormalized, receipt JSON). If total measured ≤25 min later, fold-1 and/or your v2 leg joins as 3rd leg.
- Target public **0.958–0.961**.
- Oct 21–22: freeze, full pre-deadline rerun of both final kernels, manual selection **[best-AUC, efficiency horse]** — mandatory for eff eligibility.

## Standing rules
- All submissions go through Harshil — the 5/day limit is **per team**, counted on the **UTC day** (resets 05:30 IST). LB = best, not latest.
- Never paste API tokens in chat, files, or commits.
- Kaggle quirks: datasets auto-extract `.zip` at ingest (subdirectories get dropped — upload zips, not folders); max 2 concurrent GPU kernels/account; verify with `kaggle kernels status owner/slug` before assuming success; notebook-surgery claims must be grep-verified (a claimed .ipynb edit can silently no-op).
- Questions/status → team chat. Harshil updates this file when something material changes.
