# Team Legacy: execution plan, Oct 17–22 (from Daksh)

Written Oct 16 2026, 19:30 IST, for **Harshil** and **Kartik** and their agents. This file is self-contained.

**How we work from here:** we are playing for the top of both boards, the main LB and the efficiency LB. A measured +0.001 AUC, or 69 s saved on the hidden run, is worth a full day's work. Nothing counts as "good enough" while a measured lever is still on the table. Every claim below has a source; every task has a gate.

---

## 0. Verified facts (read first)

| # | Fact | Source |
|---|---|---|
| F1 | Our MIL ensemble (nb3 v5) scored **0.756** public. The **0.950** belongs to the 3-leg public fork; strip-c384 scored **0.949**. | `receipts/submissions-public-lb.json` in the Oct 16 handoff (raw Kaggle records) |
| F2 | **Gold-58 tracks the public LB within ~0.02.** MIL: 0.780 gold, 0.756 LB. Raptor baseline: 0.917 gold, 0.924 LB. So gold-58 is a valid gate, and a leg at 0.80 gold *lowers* a 0.957 blend. | F1 + the Raptor script docstring |
| F3 | Efficiency = AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400, computed on **private** data. Eligible only if the submission is one of our **2 selected** submissions; one submission can win both tracks. No hardware restriction (CPU or GPU, ≤ 9 h). | Competition page, *Efficiency Prize Evaluation* |
| F4 | **Exchange rate** at maxAUC ≈ 0.968: **+0.001 AUC = 69 s** of hidden runtime; 1 min = 0.00087 AUC; 25 min = 0.022 AUC. | F3 |
| F5 | External data and models are allowed if public and equally accessible at no cost (Rules 2.6). Host ruling (C. Carr, RSNA, thread 743416): **registration plus licence acknowledgement is OK; institutional agreements or affiliation-only access are not.** KneeCoT is banned. **OAI is still unruled** (open in 743416). The host asks for full disclosure of training data. | Rules 2.6; discussion 743416 |
| F6 | Sending report text to commercial LLM APIs for label extraction is **explicitly permitted**. | Host post, discussion 733965 |
| F7 | Lineage of the public legs: Raptor-384/224 (nartaa) = **OAI**. R7 (heliosli) = **fastMRI+** auxiliary task + **SKM-TEA** anchors + OrthoFoundation SSL init (read from its embedded `compact_train.py`). Every leg of the public 0.957 stack has external-data lineage, and OAI is the unresolved one. | `0-9573.ipynb` cells 2, 4, 6 |
| F8 | Strong competitors report **0.94–0.955 single models** without OAI: 2.5D CoAtNet-224 single fold 0.950, 5-fold ResNet-224 0.954, DINOv2 ViT-S 0.952. All are **end-to-end fine-tuned 2.5D** models. The lever they name is **labels**: multi-source weak labels → teacher models → image pseudo-labels → final model. "Different teachers help different targets." "Don't rely on reports alone." | discussions 735304, 748295 |
| F9 | Gold labels are **image reads** (2 MSK radiologists + adjudicator, borderline graded negative). Report labels are a noisy proxy for what the test scores, which is why image pseudo-labels help. | competition data page; public repos citing the host |
| F10 | Public notebook frontier is **0.9576**. The top 40 public notebooks reuse the same legs (R7 + Raptor-384/224 + 2.5D MIL), so ~1,400 teams share correlated errors. | `kaggle kernels list --sort-by scoreDescending`, Oct 16 |

---

## 1. Design: two kernels, one shared engine

- **Kernel M (max-AUC pick).** Public stack plus our owned legs, uniform percentile-rank average, runtime < 9 h.
- **Kernel F (efficiency pick).** Built fast from the first line:
  - **One DICOM decode per study, shared by every leg.**
  - Both GPUs, fp16.
  - Target ≤ 5 min projected for ~1,300 studies.
- **Owned legs reuse the nb4 preprocessing.** Harshil's 2.5D legs and Daksh's r1 student both use exactly the nb4 pipeline, so in Kernel F they add only GPU forward time, never decode time. This is how F can carry a strong blend inside a small runtime.
- **Final selection (Oct 21–22, manual):** [M, F]. If F is within 0.002 public of M, F also serves as M and we pick the next-best M as the second slot.

---

## 2. Harshil: tasks and gates

| ID | Task | Gate / acceptance |
|---|---|---|
| H1 | **Today:** share **nb4 source** (training + inference + preprocessing/cache code) with Daksh, plus the exact list of datasets nb4 mounts. | Daksh trains the r1 student with **identical** preprocessing (see §1). |
| H2 | OOF contract: `oof_preds_25d.csv`, 4,407 rows, `StudyInstanceUID` + 12 columns in sample_submission order; each study predicted by the fold model that did **not** train on it; folds.csv sha256 `8ea359a25c841c…`. ETA Oct 17 ~09:00 IST. | Daksh verifies row count, no NaN, column order, sha. |
| H3 | **Leg admission gate** (replaces "≥ 0.80 gold"). A leg enters M or F only if **all three** hold: (a) gold-58 macro alone **≥ 0.93**; (b) paired bootstrap Δ(blend with leg − without) on gold ≥ 0 with P ≥ 0.70; (c) Spearman with the current blend < 0.95 on gold + the 130-study pseudo-test. | Daksh runs the referee check and returns a receipt. |
| H4 | Fusion stays **uniform percentile-rank average** (agreed, F2/F8). No per-finding weight tuning, no LB-driven weight search. | — |
| H5 | **Gold rows are excluded from every gated training run.** They are our only clean measurement. Gold hard labels go into training **only** in the final full-data fit, after the gates have passed. | Daksh's targets ship two files: `exclude` (gated runs) and `include` (final fit). |
| H6 | **Pseudo-label method: use Daksh's r1 targets, not pure OOF targets.** A teacher at ~0.8 gold is weaker than the 0.898 label consensus, so pure-OOF targets would cap the student at the teacher. r1 = label consensus + soft teacher mix on the findings where the image teacher beats the labels on gold. The weight is set once by a rule (0.3, or 0.5 if the teacher beats the labels on gold overall), never tuned on the LB. | Paired fold-0 experiment (D2) decides. |
| H7 | **External-data lineage table** for every leg in every submitted kernel (dataset, access terms, which leg). Ask the host in 743416 whether **public weights derived from OAI** are permitted. Keep a fallback of M and F with OAI-derived legs removed, ready by Oct 20. | Decision on OAI-derived legs by **Oct 20**. |
| H8 | Run the submission ladder in §5. | — |

---

## 3. Kartik: tasks and gates (inference side only, no training)

**K1. Build the pseudo-test harness (Kaggle T4×2 notebook).**
- **Studies:** 188 total = 130 non-gold train studies (fixed list: `folds.csv` with `is_gold == 0`, `sample(130, random_state=0)`) + all 58 gold.
- **Layout:** mirror the hidden-test layout. Build a data root with `test.csv` and `test_series.csv` (rows from `train_series.csv`), and symlink `test_series/<study>/<series>` → the mounted `train_series/...`, so **any submission kernel runs unchanged**.
- **Timing:** log per-stage seconds with `torch.cuda.synchronize()` at each boundary. Stages: startup/imports, series pick, header scan, pixel decode, preprocessing, GPU forward per leg, blend + write.
- **Projection:** `fixed_overhead + per_study × 1300`.

**K2. Parity receipt for every fast variant.** Run the fast variant and the reference kernel on the same 188 studies:
- per-finding Spearman **≥ 0.995**;
- gold-58 macro Δ **≥ −0.002** (paired bootstrap, `score_gold.py`).

**K3. Speed levers, in order of expected value. Each must keep parity.**
1. **Startup:** one Python process; no runtime `pip install` (use attached wheels only if unavoidable, and time them); import only what is used.
2. **Series pick from `test_series.csv`** (plane, Fluid_Sensitive, Fat_Suppression are given). Never open headers of series no leg uses.
3. **Header-only scan** (`pydicom.dcmread(stop_before_pixels=True, specific_tags=[...])`) to order slices, then **decode only the slices a leg consumes**. Studies hold ~165 slices; legs use far fewer.
4. **Shared decode:** decode once, derive every leg's input (384 field, 224 centre crop, nb4 windows) from the same arrays.
5. **Pipeline:** 4 CPU decode workers (producers) → one consumer process per GPU, pinned memory, prefetch, decode overlapped with GPU.
6. **GPU:** fp16 autocast, `channels_last`, `cudnn.benchmark`, `inference_mode`, windows batched across studies.
7. **TTA, window count, extra legs:** each is kept **only if it passes the exchange rule below**.

**Exchange rule (F4):** a component stays only if its gold-58 gain is **≥ 0.001 per 69 s** of projected hidden runtime (≈ 0.053 s/study).

**K4. Deliverables:**
- `frontier.csv` rows: `config, gold_auc, proj_seconds, parity_spearman_min`;
- the per-stage timing table;
- parity receipts;
- **Kernel F candidate by Oct 19, 18:00 IST.**

**K5.** Run nb4 folds on your GPU slots when Harshil schedules them.

---

## 4. Daksh: deliverables and interfaces

| ID | Deliverable | When |
|---|---|---|
| D1 | **r1 targets** as a Kaggle dataset: `r1_targets.csv` (4,407 × 13, floats in [0, 1], NaN = masked), `r1_weights.csv`, `r1_eval.csv`, in **exclude** and **include** gold variants, plus a receipt (gold-58 agreement with CI, teacher weight, mixed findings, source list with lineage). | Oct 17, ~2 h after the OOF file lands |
| D2 | **r1 student, paired experiment:** nb4 recipe, **same fold-0 and seed as Harshil's fold-0**, only the targets differ (v1 vs r1). Gold-58 paired bootstrap decides. If r1 wins: **full-data student** + weights dataset + inference snippet that plugs into nb4 inference. | Verdict Oct 18; weights **Oct 19, 12:00 IST** |
| D3 | **Referee service:** gold-58 paired bootstrap for every candidate leg/blend, correlation matrix, H3 admission receipts, efficiency projections (`efficiency.py`). | 24 h turnaround |
| D4 | **Labels and validation package** (as requested in the `daksh` handoff spec): targets, folds, scorer, leakage and duplicate-report audit, 7 adversarial scorer tests, receipts with SHA-256. | Oct 17 |

**Label lineage for disclosure (F5):**
- Report labels: 8 sources (public LLM label sets, our Opus-4.5 labels, our DeepSeek re-read of the ~450 most-disagreed reports), fused by equal-weight NaN-mean.
- Image teacher: owned 2.5D OOF.
- No external images.
- v2 consensus gold-58 agreement: **0.8981** [0.8606, 0.9304]. The rehearsal with a public image teacher at w = 0.3 on the fixed list (Effusion, Synovitis, Contusion, Medial OA) gave **0.9053**.

---

## 5. Submission ladder (UTC days, 5 per day, written in advance)

| UTC day | Slots |
|---|---|
| Oct 17 | versia7 (auto, 05:35 IST) · **F0** = strip-v2 fast decode (runtime receipt) · **M1** = helios-957 + 2.5D fold-0 *if it passes H3* · 2 spare |
| Oct 18 | **F1** = shared-decode Raptor-384 + 224 (+ R7 only if it passes the exchange rule) · **M2** = M1 + flip TTA + any further leg passing H3 · spare |
| Oct 19 | **F2** = F1 + r1 student (if D2 passes) · **M3** = M2 + r1 student · spare |
| Oct 20 | Fresh reruns of the best F and M (stability) · clean-fallback rehearsal (H7) · spare |
| Oct 21 | Freeze. Full pre-deadline reruns of both chosen kernels. **Manual selection [M, F].** |
| Oct 22 | No changes. |

Rules:
- Public differences under **0.004** are noise (~390 public studies). Decide on gold + LB jointly, never on LB alone.
- A failed hidden rerun costs a slot, so every kernel passes the K1 harness first.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Host rules against OAI-derived weights at prize review | H7 lineage table + clean fallbacks; owned legs (2.5D, r1 student) carry the clean path |
| Hidden-test crash or timeout | Per-study fallback (never crash); K1 dry run on 188 studies; F stays far below the 9 h cap |
| Preprocessing mismatch between training and inference | Owned legs share nb4 code verbatim; K2 parity receipt on every change |
| Gold-58 overfitting | Gates are paired bootstraps with fixed rules, no per-finding tuning, gold excluded from gated training (H5) |
| GPU quota | Daksh trains on his own account (resets Sat 05:30 IST); Kartik runs nb4 folds; Harshil runs fusion |
