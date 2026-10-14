# RSNA Knee Abnormality Detection — Team Handoff & 7-Day Plan

> **Purpose of this document.** This is the single source of truth for a 3-person team
> (Harshil, Daksh, Kartik) entering the Kaggle **RSNA Knee Abnormality Detection** competition.
> It contains everything we learned in planning so that **each person can paste their own
> section into their own LLM/agent and start working immediately** without re-deriving context.
>
> **Role note:** In this document, **Kartik owns speed & shipping** and **Daksh owns the
> answer key / labels & validation** (these were swapped from an earlier draft — use the
> assignment in *this* file).

---

## 0. TL;DR (read this if you read nothing else)

- We are classifying **12 knee conditions** from **knee MRI studies** (+ radiologist report text in training).
- Score = **average AUC across the 12 conditions**. Higher is better. Top of leaderboard ≈ **0.965**, a good clean target ≈ **0.94–0.95**.
- **The catch:** only **58 of 4,407** training studies have true doctor-given answers. The rest only have free-text reports. So we must **use reports to make training labels** (weak supervision), then validate on the 58.
- **There is a second prize for efficiency:** being accurate *and* fast. Speed is a real scoring axis, not a nicety. A near-top model already exists at ~0.945 with a fast config — we should lean on it.
- **Three jobs, one owner each:** Harshil = the model; Daksh = labels + referee (scorecard); Kartik = speed + packaging + submission decisions.
- **Deadlines:** join/merge team by **Oct 15, 2026**; final submission **Oct 22, 2026**.
- **Starting assets already built by us:** `score_gold.py`, `gate.py`, `make_folds.py`, `levers.csv`, `frontier.csv` (all in this folder).

---

## 1. The competition in plain language

- **Input (test):** a knee MRI study = a folder of DICOM image series taken in 3 planes
  (Sagittal, Coronal, Axial). **No report text is available at test time.**
- **Input (train):** the same MRI studies **plus** a free-text radiology `Report` (mixed languages).
- **Output:** for each of 12 conditions, a probability between 0 and 1 that the condition is present.
- **Metric:** macro-averaged **ROC-AUC** over the 12 conditions. AUC cares only about *ranking*, not calibration (see Pitfalls).
- **The hard part:** labelled data is almost nonexistent. You learn from reports (noisy) and must generalize to hidden test scans.

### The 12 conditions (targets)
`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`,
`Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

> Verify the exact target column names by opening `sample_submission.csv` / `train.csv`.
> Never hardcode them — read them from the file at runtime.

---

## 2. Competition facts (verified)

| Item | Detail |
|---|---|
| Host | RSNA (Radiological Society of North America), 2026 AI Challenge |
| Platform | Kaggle, **Research Code Competition** (submissions are *notebooks*) |
| URL | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection |
| Task | Multi-label classification of 12 knee abnormalities from MRI studies |
| Metric | Macro-averaged ROC-AUC over 12 targets |
| Prize pool | **$77,000** |
| Main prizes | $9k / $7k / $6.5k / $6k / $5.5k / $5k ×5 |
| Efficiency prizes | $7k / $6k / $5k |
| Submission limit | 5 per day; **2 final submissions** selected |
| Team size | up to 5 (we are 3) |
| Environment | Notebook only, **internet disabled**, **≤ 9 hours** CPU/GPU total |
| External data | "Freely & publicly available" external data + pretrained models allowed (see §7 OAI risk) |
| Data license | MIRA (data use); winner license CC-BY-NC 4.0 (with conflicting commercial-use wording — unclear) |

### Timeline (2026)
| Date | Event |
|---|---|
| Jul 30 | Competition start |
| **Oct 15** | **Entry deadline + Team merger deadline** (hard) |
| **Oct 22** | **Final submission deadline** |
| Nov 5 | Winners' requirement / deliverables |
| Nov 29 – Dec 3 | RSNA 2026, Chicago (recognition) |

### Medals
- Bronze: top **10%**
- Silver: top **5%**
- Gold: top **10 teams + 0.2%**

Current leaderboard shape: frontier ≈ **0.94**, top ≈ **0.965**, bronze line ≈ **0.950**.

---

## 3. Data (verified schema)

| File | Shape | Columns |
|---|---|---|
| `train.csv` | 4,407 × 14 | `StudyInstanceUID`, `Report`, + 12 label columns |
| `train_series.csv` | 24,371 × 5 | `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane` |
| `test.csv` | hidden set | `StudyInstanceUID` only (no Report, no labels) |
| `test_series.csv` | hidden set | same as `train_series.csv` (no Report) |
| `sample_submission.csv` | — | `StudyInstanceUID` + 12 label columns, all `0.5` |
| Images | — | `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm` |

**Structure:** Study → Series → Slices. A study is one patient exam. A study has 3–14 series.
Each series is a stack of DICOM slices in one plane. We score at the **study** level.

**Verified quirks / gotchas:**
1. **Only 58 of 4,407 train studies have gold labels.** Everything else is report-only. (~1.32%)
2. **`Fluid_Sensitive` == `Fat_Suppression`** on every row → it is effectively ONE binary feature, not two.
3. **Every study has Sagittal + Coronal + Axial.** Planes are always present.
4. **3–14 series per study.**
5. **Local `test.csv` shows only ~3 example studies.** The real test set (~1,300) is swapped in at scoring
   time. You cannot overfit to the local test set.
6. **No `PatientID` column**, so the same patient with multiple studies can appear in different folds.
   You can only group by `StudyInstanceUID`.
7. **Reports are multilingual** (~12 languages).

---

## 4. The weak-supervision concept (why this competition is unusual)

Because 4,349/4,407 studies have no labels, the winning pattern is a **two-stage pipeline**:

1. **Make labels from reports.** Turn free-text `Report` into 12 probabilities per study using
   rules and/or an LLM. Quality is measured by AUC **on the 58 gold studies**.
   - Rule-based extractors reach ~**0.81** macro-AUC.
   - LLM-based extractors reach ~**0.86–0.90** macro-AUC (Synovitis is the hardest condition).
2. **Train the image model on those noisy labels.** Multi-label, study-level.
3. **Validate on the 58 gold studies** using the same split every time.

**Consequences:**
- Errors in the label extractor propagate. Noisy labels matter, especially for rare conditions.
- 58 studies give a **noise floor of ~±0.17** (95% CI) on the rarest conditions — so tiny AUC differences
  on gold are **not** meaningful. Don't chase 0.002 wiggles on the 58.
- Some **public checkpoints were trained on the 58 gold studies**, which contaminates any validation
  that uses them. Be suspicious of "0.94" numbers from public weights until you know their training data.

---

## 5. Efficiency metric (the second prize — understand this)

The efficiency leaderboard does **not** just rank by accuracy. It ranks by:

```
Efficiency = [normalized AUC gap] + RuntimeSeconds / 32400      (lower is better)
```

- `32400` seconds = the **9-hour** evaluation cap.
- The AUC gap is measured from the top (~0.965), normalized against a 0.5 baseline.
- **Only *selected* submissions are scored for efficiency.** If you never tick "select for efficiency"
  on a submission, you get **nothing** for the efficiency prize — this is the #1 documented way teams
  lose it for free.
- You must beat the `sample_submission` benchmark to be eligible.
- Runtime is measured for the **whole notebook evaluation**, not just the model forward pass.

**Exchange rate (rule of thumb):** ~**1 extra minute of runtime ≈ 0.001–0.002 raw AUC**.
So an extra minute must buy **at least ~0.002 AUC** to be worth it. Heavy blends/TTA usually are **not** worth it.
Winning shape = **small + fast** (e.g. distill a big ensemble into one fast student).

There is a live **efficiency leaderboard notebook** (rank-only, updated daily):
`ryanholbrook/rsna-knee-abnormalities-efficiency-lb`
https://www.kaggle.com/code/ryanholbrook/rsna-knee-abnormalities-efficiency-lb

---

## 6. What we learned from the discussion tab (intel digest)

> Note: Kaggle's discussion tab is JS-rendered and rate-limits direct scraping (HTTP 429).
> This is reconstructed from indexed thread content + cross-references.

**Threads found:**
| Thread | Topic |
|---|---|
| `discussion/737696` | "RSNA Raptor Weights" — public CoAtNet weights + the window-count dial |
| `discussion/738096` | "DINOv2 Base vs Small at 224" — the **metastrategy** thread (most useful) |
| `discussion/738172` | shared **gold 0.9083 vs LB 0.924** (gold↔LB calibration) |
| `discussion/735304`, `discussion/736635` | participants sharing gold vs LB scores |
| `discussion/733343`, `discussion/744056` | not fully readable (rate-limited) |

### 6.1 The metastrategy (thread 738096) — most important
Top participants run **cheap experiments**: single fold, smallest DINOv2, 224px — and mark those
submissions **for efficiency**. They only train the **heavy** config (5-fold, largest model, 448px)
at the very end for the leaderboard score.
- **DINOv2-small ties DINOv2-base at convergence for ~4× less compute** → small is the
  "obviously better choice for the efficiency prize."
- Practical rule: **ablations are cheap; the final heavy run is last.**

### 6.2 Raptor weights window dial (thread 737696)
Same public CoAtNet weights, scored with different numbers of stacked windows:
- **42 windows → 0.924**
- **62 windows → 0.927**
- **78 windows → 0.932**
- Single unblended model tops out ~0.936–0.938. SWA (weight averaging) ≈ no gain.
- Geometry matters: slice **density** (slices per stack) beats total **span**.

### 6.3 Gold ↔ LB calibration
Everybody sees the same pattern: local gold ≈ **0.90–0.91** ↔ public LB ≈ **0.92–0.925**.
"Hitting a wall" is normal and expected.

### 6.4 Pilkwang's methodology rules (apply these)
1. **AUC is rank-only** → **rank-average** ensembles; **never calibrate** (calibration is invisible to AUC).
2. **Grade report mentions — never binarize.** An annotator's threshold ≠ a radiologist's threshold.
   Keep soft 0/0.5/1 style signals where possible.
3. **No text branch at test time.** `Report` exists only in train, so the model must be pure-image at inference.
4. **Recover acquisition axes from DICOM headers** — the delivered `Fluid_Sensitive`/`Fat_Suppression`
   flags are identical (useless as two features).
5. **Laterality is "Type 2C" and is absent on ~half of studies.** Leaving it blank silently declares those
   knees **left-sided**. This is a real bug and a real accuracy loss — recover the side from the DICOM header.
6. **Duplicate report text must be grouped by report-hash** or you leak between folds.
7. **58 studies cannot arbitrate label tables** (rare-condition CI ≈ ±0.17). Use them as a sanity check, not gospel.

### 6.5 Public assets worth using
| Asset | What it is |
|---|---|
| Raptor CoAtNet weights (author `dreaddevelopment`) | Strong public single model; window-count dial |
| `nartaa/rsna-knee-publication-swa-weights-20261007` | **Accuracy model** = 384-trained seed42 SWA, **0.949** (native320/K94/anatomical mirroring). **Efficiency model** = 224-crop seed42 SWA, **0.945** (K94/autocast/one view). Uses **2,399 OAI knees** for masked external supervision. |
| `dariushafshar/rsna-knee-2026-grouped-cv-folds` | **Canonical public 5-fold split** — use it so all of us score on the same partition |
| Public label tables | `yunusgmsoy` (2-source), `laymond` Qwen2.5-32B (**0.8686** macro-AUC), `pilkwang`, `flight0234`, `stevenleehans`. **Warning:** at least five tables copy the 58 gold labels verbatim — do NOT use those to validate against gold. |
| Public rank-ensembles | ~**0.940–0.943** (DINOv2 + RadImageNet + Raptor + CoAtNet readers, rank-averaged) |
| `konradcinkusz/rsna-knee-abnormalities-efficiency-lb` | Efficiency-track reference repo (also documents the CSV columns) |
| `TranBaDat2607/RSNA-Knee-Abnormality-Detection` | LLM report labeler + CoAtNet 2.5-D MIL pipeline reference |
| `leminhhung0101/knee-model` (HF) | Series → study aggregation reference |

### 6.6 Technical meta
- Typical architecture: **CoAtNet 2.5-D attention-MIL** (stack a few adjacent slices into a 2.5D image,
  pool over a variable number of "windows" with attention).
- DINOv2 backbones (small/base) at 224px are the efficiency workhorse.
- Fed through study-level aggregation (multiple series → one 12-vector).

### 6.7 Previous RSNA challenge winners (context)
- 2022 Cervical Spine (qishen + ha), 2023 Abdominal Trauma (Team Oxygen), 2024 Lumbar Spine (Avengers).
- Recurring strong names: haqishen/qishen, theoviel, jiweiliu, maxjeblick, yujiariyasu, IanPan.

---

## 7. The OAI external-data risk (read before choosing a strategy)

- The strongest public weights (`nartaa`) train on **2,399 OAI knees** for "masked external supervision"
  and reach **0.949 (accuracy) / 0.945 (efficiency)**.
- **OAI is free but controlled-access** (requires an application + NDA). It is ambiguous whether it counts
  as "freely & publicly available external data" under the rules.
- **Hosts have not ruled on this.** This is an eligibility risk, not just a technical one.
- **Decision (Day 0/1):** default to a **clean track** (no OAI) for our efficiency entry — it is safer and
  the efficiency prize rewards *speed*, not exotic data. Keep an **OAI track in reserve** only as a lever
  if the clean track stalls. Do not silently use OAI-trained public weights without flagging it.
- Also flagged: the **MIRA-data vs CC-BY-NC-winner-license "commercial use" contradiction** — worth a rules question.

---

## 8. Team roles (SWAPPED per request)

> **One artifact, one owner.** Each person edits only their own files. If you need something from
> someone else, ask — never do their job for them.

| Person | Role | Mission | Owns | Never touches |
|---|---|---|---|---|
| **Harshil** | **The Model** | Make accuracy go up | Backbone choice, 2.5D windowing, MIL head, training recipe, laterality/orientation fix, TTA/blend policy, distillation | Labels, ledger, gate, runtime packaging |
| **Daksh** | **The Answer Key & Referee** | Give the model good labels; keep scores honest | Report→label extraction, label-table selection, CV split, macro-AUC + gold-58 scorecard, duplicate-report grouping, experiment ledger, **submission gate** | Model code, runtime code |
| **Kartik** | **Speed & Shipping** | Make it fast and ship it safely | DICOM decode/caching, dual-GPU/autocast/prefetch, profiler, AUC-vs-time frontier, offline packaging, efficiency-LB monitoring, **submission selection & budget** | Labels, training |

### Deliverables per person
- **Harshil:** trained checkpoint(s) + inference code.
- **Daksh:** `train_targets.csv` (the chosen labels), `folds.csv` (the shared split), working `score_gold.py` + `gate.py`, maintained ledger.
- **Kartik:** a packaged offline notebook that produces `submission.csv` within the time limit, plus the AUC-vs-time frontier and the two selected final submissions.

---

## 9. Shared interface files (the handoff contract)

These live in this folder and were already built and smoke-tested:

| File | Owner | Purpose |
|---|---|---|
| `score_gold.py` | Daksh | Compute macro AUC on the 58 gold studies. Everyone uses this same script. |
| `gate.py` | Daksh | **Submission checker.** Row count, columns, UID set, NaN/inf, range [0,1], degenerate columns. Exit 0 = safe to submit. |
| `make_folds.py` | Daksh | Build study-grouped folds, or merge the canonical public split (`--canonical`). |
| `levers.csv` | Harshil writes, Kartik reads | Harshil lists each config with its standalone gold AUC. |
| `frontier.csv` | Kartik writes | Kartik adds measured seconds + marginal AUC-per-minute + selected flag. |

**Rules for these files:**
- Columns of `levers.csv`: `config,gold_auc,owner,date,notes`
- Columns of `frontier.csv`: `config,gold_auc,seconds,marginal_auc_per_min,selected,notes`
- Only one person writes each file. The other appends via the owning person.

### Usage examples
```bash
# Score a prediction file on the 58 gold studies
python3 score_gold.py --train train.csv --pred preds.csv

# Validate a submission before uploading
python3 gate.py --sub submission.csv --sample sample_submission.csv

# Build 5 folds grouped by study (use the canonical public split if available)
python3 make_folds.py --series train_series.csv --out folds.csv --n 5
python3 make_folds.py --series train_series.csv --canonical canonical_folds.csv --out folds.csv
```

---

## 10. The 7-day plan (plain language), per person

> Today = **Oct 13, 2026**. Final submission = **Oct 22**. Team merge deadline = **Oct 15**.

### Day 1 — Get *anything* working end to end
- **Harshil:** Run one existing public model notebook start to finish. Just have *a* working prediction.
- **Daksh:** Create the first label file (a public label table + report rules). Stand up the 58-study scorecard.
- **Kartik:** Time how long the public model takes; produce a valid `submission.csv`.
- **Done when:** all three can score the same prediction and get the same number.

### Day 2 — Pick labels, make it cheap
- **Harshil:** Run small, cheap experiments (1 fold, small model, 224px).
- **Daksh:** Choose the best label source by testing on the 58 (public table ~0.899 vs LLM ~0.87).
- **Kartik:** Submit the first entry and **tick "efficiency"** so it counts for the speed prize.

### Day 3 — Improve model, speed it up
- **Harshil:** Improve the model; **fix the left/right laterality bug**.
- **Kartik:** Make inference faster (half-precision, both GPUs, pre-decode scans).
- **Daksh:** Re-check every score; watch the public efficiency leaderboard.

### Day 4 — Accuracy vs. speed trade
- **Harshil:** Add accuracy boosters (extra views, mirrored scans).
- **Kartik:** Measure the *seconds* each booster costs; pick the best "accuracy per second" setting.
- **Daksh:** Confirm the numbers are real, not noise.

### Day 5 — Robustness + full dry run
- **Harshil:** Make sure it survives hard/messy scans.
- **Kartik:** Run the whole thing offline, end to end, inside the 9-hour limit with margin.
- **Daksh:** Final format/row checks + independent validation.

### Day 6 — Lock it in
- Build **two final submissions**: best accuracy, and best accuracy-per-second (efficiency).
- **Select both.** Freeze — no risky changes after this.

### Day 7 (Oct 20–21) — Buffer
- Only safe improvements. Monitor the leaderboard. No new ideas.

### Oct 22 — Submit early; confirm both are selected; stop.

---

## 11. Ground rules (avoid scope creep and duplicated work)

1. **Stay in your lane.** Each person edits only their own files/artifacts.
2. **One scorecard, one split.** Everyone always scores on the same 58 studies with the same folds.
3. **Nothing gets submitted unless `gate.py` passes.** Kartik owns the submission button.
4. **Cheap first.** Ablations use 1 fold / small model / 224px. Heavy runs are last.
5. **Respect the exchange rate:** an extra minute of runtime must buy ≥ ~0.002 AUC.
6. **Never calibrate for AUC; rank-average instead.**
7. **Flag OAI-trained weights explicitly.** Do not silently use them.

---

## 12. Known pitfalls checklist

- [ ] Splitting rows randomly instead of **grouping by `StudyInstanceUID`** → leakage, fake scores.
- [ ] Using label tables that **copy the 58 gold labels** to validate on gold → circular, useless.
- [ ] **Binarizing** report mentions instead of grading them.
- [ ] Leaving **laterality blank** (silently becomes "left").
- [ ] **Calibrating** probabilities (invisible to AUC; wastes effort).
- [ ] Adding a **text branch** that only exists in training (fails at test).
- [ ] Forgetting to **tick "efficiency"** on the submission that should count for the speed prize.
- [ ] Trusting **±0.005 gold differences** as real (the 58-study noise floor is ~±0.02 overall, ±0.17 on rare conditions).
- [ ] Overfitting to the **local 3-study test set** (real test is swapped at scoring).
- [ ] Blowing the **9-hour** runtime cap or assuming **internet** access.

---

## 13. Ready-to-paste prompts for each person's LLM/agent

### Harshil (Model)
```
Context: Kaggle RSNA Knee Abnormality Detection. Multi-label, 12 knee conditions, metric = macro ROC-AUC,
study-level. Only 58 of 4407 train studies have gold labels; the rest are report-only (weak supervision).
Test has NO report text. Architecture de facto standard: CoAtNet 2.5-D attention-MIL over stacked adjacent
slices, attention-pooled over windows. Public assets: Raptor CoAtNet weights (window dial: 42->0.924,
62->0.927, 78->0.932), nartaa SWA weights (accuracy 0.949 / efficiency 0.945 at 224px), DINOv2-small ~= base
at 4x less compute. Cheap-first protocol: single fold, small DINOv2, 224px for ablations; heavy 5-fold/448px last.
AUC is rank-only -> rank-average, never calibrate. Fix laterality (Type 2C absent on ~half of studies silently
reads as left; recover from DICOM header). Fluid_Sensitive == Fat_Suppression (one feature).
My job: raise accuracy. Deliverable: trained checkpoint(s) + inference code.
Produce: a concrete training plan + first ablation run.
```

### Daksh (Answer Key & Referee)
```
Context: Kaggle RSNA Knee Abnormality Detection, metric = macro ROC-AUC over 12 conditions, study-level.
train.csv = StudyInstanceUID, Report, 12 labels; ONLY 58 studies are gold-labeled, rest report-only.
I must turn Report free text (multilingual) into 12 soft labels used for training, then validate on the 58.
Rules: grade mentions, do NOT binarize; group duplicate reports by report-hash to avoid leakage;
AUC is rank-only; 58 studies have a noise floor ~±0.17 on rare conditions so tiny differences are not real.
Public label tables exist (yunusgmsoy, laymond Qwen2.5-32B macro-AUC 0.8686, pilkwang, flight0234,
stevenleehans) but at least five copy the 58 gold labels verbatim -> do not validate against those.
My job: produce trustworthy labels + the honest scorecard + the submission gate.
Deliverables: train_targets.csv, folds.csv (canonical or study-grouped), running score_gold.py and gate.py.
Produce: an extraction/validation plan and the first label table v1.
```

### Kartik (Speed & Shipping)
```
Context: Kaggle RSNA Knee Abnormality Detection. Healthy team has accuracy ~0.94-0.95; the second prize
is EFFICIENCY: Efficiency = normalized AUC gap + RuntimeSeconds/32400, lower is better, 32400s = 9h cap.
Only SELECTED submissions are scored for efficiency (forgetting to tick "efficiency" = lose the prize free).
Runtime = whole notebook evaluation. Exchange rate ~ 1 min runtime ~= 0.001-0.002 AUC, so an extra minute
must buy >=~0.002 AUC. Winning shape = small + fast (distill heavy ensemble into one fast student).
Environment: notebook only, internet DISABLED, <=9h CPU/GPU, 5 submissions/day, 2 final selected.
Live efficiency leaderboard: kaggle .../ryanholbrook/rsna-knee-abnormalities-efficiency-lb (rank-only, daily).
My job: make it fast, package the offline notebook, and own submission selection/budget.
Deliverables: packaged offline notebook producing submission.csv in time; AUC-vs-seconds frontier; 2 selected finals.
Produce: a profiling + packaging plan and the first efficient end-to-end run.
```

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **Study** | One patient exam = the unit we score. Contains multiple series. |
| **Series** | One stack of slices in one plane. |
| **Gold-58** | The 58 training studies with real doctor labels — our only honest validation. |
| **Weak labels** | Labels inferred from report text, used to train because gold labels are missing. |
| **2.5-D / MIL** | Feeding a few adjacent slices as one image; "multiple instance learning" pools many such windows into a study prediction. |
| **Window** | A small stack of adjacent slices fed to the model; more windows = more compute and usually more AUC. |
| **AUC** | Ranking metric; 0.5 = random, 1.0 = perfect. Rank-only, calibration-invariant. |
| **Efficiency** | The prize for accuracy + speed combined (see §5). |
| **OAI** | Osteoarthritis Initiative — a controlled-access knee dataset used by some public weights; eligibility unclear (§7). |
| **Gate** | The checker (`gate.py`) a submission must pass before uploading. |
| **Frontier** | The table of configs vs their accuracy and runtime; used to pick the best "accuracy per second" point. |

## 15. v0 baseline + overnight status (Oct 14, 08:10)

- **Data**: all 95,483 subset files (55GB) downloaded and verified (0 missing, 0 size mismatches) — pulled as 25.76GB compressed via HTTP range-reads from the 265GB Kaggle zip (`tools/zip_subset.py`).
- **v0** (dinov2-small MIL, 42 windows/study, trained on Gold-58 only, fold 0, 10 ep = 120 steps): **MACRO AUC 0.5073 on Gold-58 — chance level.** Expected: massively undertrained and only 46 train studies/fold.
- **v1 launched**: same arch, `--no-gold-only` → trains on all weak+gold labeled studies (~2,546), 3 epochs (~2.5h on the 4060). Watch `logs/TRAIN_STATUS`; score lands in `logs/score_v1.log`.
- Runs: `runs/v0/preds.csv`, score in `logs/score_v0.log`; levers.csv has the v0 row.
- **v1b killed** (gold-58, 30 ep): val AUC stuck 0.46–0.53 through epoch 10 — more epochs on 46 studies doesn't help; data was the bottleneck.
- **KEY FINDING**: train.csv has NO weak labels — 4,349/4,407 rows are all-NaN (only gold-58 labeled). Reports exist for every row. `tools/weak_labels.py` now generates them: multilingual report → 12 soft labels, **parser-vs-gold MACRO AUC 0.709** on the 58 (Baker's .81, MedMen .81, ACL .79, Effusion .53 worst). Output: `data/train_weak.csv` (positives: meniscus 39/24%, effusion 69%, ACL 12%).
- **v2 launched 08:32**: trains on all 4,407 studies w/ soft weak+gold labels (`--labels train_weak.csv --no-gold-only`), 2 epochs (~3h on 4060), then infer+score. Watch `logs/TRAIN_STATUS`, `logs/score_v2.log`.

## Update 14:10 — v2 scored, v3 running (autonomous session)
- **v2 (weak-label training) scored MACRO 0.5447** vs 0.5073 baseline (+0.037). Per-label: Fracture .692, Effusion .632, MCL .573, Contusion .570, Medial OA .578, MedMen .557, ACL .531, Baker's .517, PF OA .494, Synovitis .504, LatMen .457, LatOA .433. Best ckpt preserved: `checkpoints/best_v2.pt`; preds `runs/v2/preds.csv`; levers.csv row added.
- Val AUC was still climbing at 2 epochs (0.517→0.552) → **v3 launched 14:06: same config, 4 epochs** (~88 min/ep on 4060, ETA score ~21:50). Watch `logs/TRAIN_STATUS` / `logs/score_v3.log`.
- **Submission format checked**: `sample_submission.csv` = StudyInstanceUID + 12 prob cols (all 0.5 placeholders); public `test.csv` has only 3 studies / 15 series (real test hidden). Merge deliverable = submission.csv from best ckpt.
- Weak labeler v1 committed (MACRO .709 vs gold). Next round targeting worst labels: Effusion .53, PF OA .63, Synovitis .64, Medial OA .65 → `train_weak_v2.csv` (v4 candidate).
- How to regen weak labels: `python3 tools/weak_labels.py` (writes `data/train_weak.csv`; validation table vs gold-58 prints at end).
- **Submission pipeline ready**: `python3 tools/make_submission.py` (--ckpt best_v2.pt default) → submission.csv. Verified end-to-end. CAVEAT: the 3 public test studies have NO images on disk (0/15 series exist; no UID overlap with train — real test is hidden). Until real test images appear, predictions use all-zero dummy windows (identical ~0.5s-ish probs). Pipeline auto-handles real test data when present.
- **Weak labeler v2 (committed)**: parser-vs-gold MACRO **0.7651** (+0.056 vs v1; Effusion .53→.79, PF OA +.10, Synovitis +.09, LatOA +.08). Output `data/train_weak_v2.csv` (not overwritten v1). v4 candidate: retrain on v2 labels.

## Update 21:00 — Night plan (user asleep, autonomous)
- **PARTIAL DATASET CONFIRMED (user was right)**: file_index.jsonl = 819,640 files; only 95,495 on disk (~12%). 1,861/4,407 train studies (42%) have NO images on disk — v2/v3 trained those studies on all-zero windows with real soft labels (batch poisoning; likely a key reason scores stuck ~0.55).
- **Night download running** (tmux dl4 → /home/harshil/rsna-ext/data, 360,327 files / 284.4 GB for the missing studies, ~13 it/s ≈ 7.5h, ETA ~03:45; resumable; signed-URL expiry caveat). Root disk has 687GB free. Morning: merge via move+symlink into data/train_series.
- **v3 (4 epochs)** finishing ~22:10 with infer+score. Expectation: epoch-2 val AUC dipped (0.512) vs v2 epoch-2 (0.552) due to slower cosine anneal over 4 epochs; epochs 3-4 climbing (0.529 @ ep3).
- **Overfit sanity test queued** (user directive #2: small tests before long runs) — tools/overfit_test.py written, RUN AFTER v3 frees the GPU (~22:15). If it fails to overfit 16 real studies → pipeline bug hunt before any more long runs.
- **Next lever after download**: v4 = retrain on ALL studies with real images + weak-labeler-v2 labels (MACRO .7651). Also consider: only-download-enough + two-stage.
- **v3 scored 0.5012 (WORSE than v2 0.5447)** — 4 epochs on the poisoned mix overfit noise. Lesson: val-AUC-on-soft-labels is a misleading proxy; trust score_gold (gold-58 MACRO).
- **Overfit test PASSED in substance**: tools/overfit_test.py, 16 real studies, train AUC 0.9476 @ ep60 (loss 0.217 is the expected soft-label floor, not a bug) → pipeline (data→model→loss) is sound.
- **v2b launched 23:05**: labels = train_weak_v2_img.csv (2,546 image-available studies only, labeler-v2 MACRO .7651), 3 epochs, then infer+score (~03:00). Isolates (a) zero-window fix + (b) better labels vs v2.
