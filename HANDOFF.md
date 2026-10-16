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

## 07:00 pivot: Kaggle notebook pipeline (user decision with friends)
- **Local full-set download CANCELLED** (454 GB remaining was the real need; storage won't fit + code-competition means submit runs on Kaggle anyway). Keep 115 GB local cache for prototyping. dl tmux dead — do not resume.
- `kaggle/` scaffolding complete and committed: nb1_features.ipynb (819k DICOMs → 4407×[42,384] fp16 embeddings, ~1-1.5h on T4), nb2_train.ipynb (head-only MIL on embeddings, ~10min), nb3_submit.ipynb (hidden-test inference; windowing+model ported verbatim). README.md = team runbook.
- Published: dataset `harshilsrivastav/rsna-knee-weak-labels` (train_weak.csv + train_weak_v2.csv) ✓ ready.
- Pending: `dinov2-timm-cache` dataset upload (tar built at kaggle-out/, 176 MB, offline-load VERIFIED locally) — needs the user's storage cleanup.
- NB1 extraction logic smoke-tested on real public-test DICOMs (12 slices → windows → embeddings finite ✓). Public test series ARE downloadable via API (557 files listed at /tmp/opencode/test_files.txt) if ever needed locally.
- NOTE: tmux sessions keep dying on this WSL (server vanished ~02:07 killing v2b mid-run). v2b relaunched 06:55 via `setsid nohup` (tmux-independent, heartbeat logs/v2b.pid). DON'T launch long runs in bare tmux anymore.
- v2b ETA ~10:15 (3 ep × 509 steps on 2546 real-image studies + labeler-v2 labels) → score_gold → levers.csv. This validates: better labels + no zero-poisoning before we bake the recipe into NB2.
- **07:30 v2b CANCELLED** (user call, correct): local GPU runs add no information now — the identical question (labeler-v2 + real images) gets answered on Kaggle on the FULL set. best_v2.pt (0.5447) remains the local reference.

## Update 08:05 Oct 15 — Kaggle pipeline live via API (orchestrator)
- `dinov2-timm-cache` dataset PUBLISHED via CLI (`--dir-mode tar`; Kaggle auto-extracts → root has `hf/` + `torch/`, matches NB1/NB3 expectations; `torch/` was empty, harmless). Verified via `kaggle datasets files`.
- Discovered: `kaggle kernels push` runs notebooks HEADLESSLY (Save-&-Run-All equivalent) with data attached via kernel-metadata.json — no manual UI upload needed. Workflow documented in kaggle/README.md.
- **NB1 pushed + RUNNING**: kernel `harshilsrivastav/rsna-knee-nb1-features` (GPU, internet OFF, comp data + timm cache attached). ETA ~1-1.5h → then publish output as `rsna-knee-embeddings-v1`.
- NB2 edited before push: added final cell scoring best.pt on ALL 58 gold studies (mimics score_gold.py, writes gold_preds.csv + gold_score.json) → number directly comparable to local 0.5447, no misleading soft-label proxy needed.
- Next: when NB1 done → publish embeddings-v1 → push NB2 (kernel: rsna-knee-nb2-train) → check gold MACRO ≥ ~0.55 → publish ckpt-v1 → push NB3 fork → submit.

## Update 10:40 Oct 15 — Fully autonomous chain watcher running (orchestrator)
- **Automation upgrade (user directive: no chat polling, delegate mechanical work)**: old polling scripts killed; a detached chain-watcher now drives the WHOLE pipeline end-to-end: wait NB1 → publish `rsna-knee-embeddings-v1` (`--dir-mode tar`, embeddings/ subdir preserved) → push NB2 → on NB2 complete: read `gold_score.json` → append levers.csv row `kaggle-nb2-head-3ep-emb-v1,<MACRO>,harshil,2026-10-15,...` + HANDOFF bullet + git commit → publish `rsna-knee-ckpt-v1` (flat) → push NB3-test (`dataset_sources: ckpt-v1 + dinov2-timm-cache`) → download submission.csv → `gate.py` → **submit only if GATE PASS** → final commit. On any kernel ERROR: log saved to `/tmp/opencode/<slug>-err`, chain halts at `phase=error`.
- **Watcher build was fully delegated** (@grunt authored → review found 4 bugs → @coder fixed + selftest-verified): bugs were (1) run_cmd swallowed stdout so status captures were always empty → infinite poll loop; (2) `local` outside functions → crash under `set -u`; (3) embeddings staging flattened npz into dataset root, which would have broken NB2's `<mount>/embeddings/*.npz` resolver; (4) gate call used an unbounded `*.csv` glob. Verified by a SELFTEST mode (stubbed `kaggle` CLI + fake repo) that walks every phase to `phase=done`, incl. embeddings-subdir regression guard + kernel-metadata.json JSON validation + levers/HANDOFF/git artifacts.
- **LAUNCHED 09:50**: `setsid nohup env -u DRY_RUN -u SELFTEST bash /tmp/opencode/chain_watcher.sh` (PID 4664; state `/tmp/opencode/chain_state.txt` = phase/detail/last_check/pid; log `/tmp/opencode/chain.log`; polls every 4 min; 6h watchdog).
- **Status 10:40: NB1 still RUNNING** (~2h vs 1-1.5h ETA — serial DICOM reads of 819k files slower than estimated; 12h kernel cap OK, watcher watchdog 6h from launch ≈ 15:50).
- **Monitor protocol**: check ONLY on demand — `cat /tmp/opencode/chain_state.txt` (+ `tail /tmp/opencode/chain.log`). No scheduled polling.
- **Repo HEAD = `564c1bc`** (nb3 dual-ckpt-key loader + local NB3 mirror test PASSED gate.py on 3-study fake submission) — supersedes the `1306eaf` hash mentioned earlier.
- **After phase=done**: verify submission on Kaggle (version N of `rsna-knee-nb3-submit-test`), record public LB in levers.csv, then next levers per HANDOFF_PROMPT (stronger head, window count, labeler-v3) — compare NB2 gold MACRO vs local best_v2 0.5447 first.
## Update 15:07 — NB2 gold-58 MACRO 0.6346948937877982 (soft val 0.7138747934759152) on Kaggle full-set embeddings; rsna-knee-ckpt-v1 published; NB3-test pushed.
## Update 15:11 — NB3-test ran clean on public test; gate PASS; sanity submission sent (version 1).

## Correction 15:22 — submission actually registered (the 15:11 note was premature)
- The 15:11 submit FAILED silently: RSNA comp requires `-f submission.csv` (output FileName) on `kaggle competitions submit -k ... -v ...`; without it → HTTP 400 "This competition requires an output FileName for Notebook Submissions" (CLI swallows the body; verified via direct API call). Watcher script patched to always pass `-f submission.csv`.
- **Registered for real 15:20:36**: submission ref 57235426, kernel `rsna-knee-nb3-submit-test` v1, status pending (Kaggle re-runs it server-side on the hidden test). 4 submissions left today.
- **NB2 official result (banked)**: gold-58 MACRO **0.6347** (soft val 0.7139) vs local best_v2 0.5447 → **+0.09**. A v3 run of the same code printed 0.663 → run-to-run variance ~±0.03 (loader shuffle unseeded). Best model number so far; levers.csv row + commit c2055ae.
- NB2 v1–v3 failures = notebook bugs found by real runs, all fixed in `kaggle/nb2_train.ipynb`: (1) gold-scoring cell spliced BEFORE the training cell → FileNotFoundError on best.pt; (2) per-epoch val AUC crashed on soft labels ("continuous format is not supported") → now binarizes (Y>0.5) + uses label mask; (3) missing `import json` → died after printing the score, before gold_score.json.
- NB1 v3 completed in 5h03m (serial 819k DICOM reads). Parallelized+resume-capable NB1 contingency prepped at /tmp/opencode/nb1-v2/ (NOT pushed, not needed).
- Watcher bugs fixed mid-flight: stale-output-dir guard (skipped fresh download because an old debug dir existed → now requires >MIN_NPZ npz to skip), datasets-create idempotency (pre-check "ready" → skip create).
- Pending: public LB score for ref 57235426. Next levers: seed loader shuffle, more epochs (soft-val still climbing 0.69→0.71→0.72), labeler-v3, window count.

## Update 16:15 — NB2 v5 (seeded loader, 8 epochs): gold MACRO 0.6868
- Soft-val climbed 0.6975→0.7385 over 8 ep (monotonic, tiny ep6 dip). Gold-58 MACRO **0.6868** — new best (v4 0.6347, v3 0.663, local 0.5447). +0.14 over baseline.
- Promoted: `rsna-knee-ckpt-v1` bumped to v5 weights; NB3 v2 pushed (uses v5 ckpt); pending submission → will record.
- NB3 v2 ran clean, gate PASS (rows=3 cols=13), submitted as v2 — pending hidden re-run. 3 submissions left today.

## Update 22:45 IST — NB2 v6 on Daksh's labels: gold MACRO 0.7690 (+0.082)
- Integrated team label backbone: pulled dakxsh23/fisrt kernel output (folds.csv 4407x4, v1_targets.csv 4407x12 dense soft targets) → published as dataset `rsna-knee-labels-v1` (files also in repo data/labels-v1/). NB2 v6 = 8ep head trained on v1_targets, val=fold 0 (910 studies, 12 gold unseen). Gold-58 MACRO **0.7690** (soft val 0.7602) vs 0.6868 on old weak labels.
- Label ladder now: our parser 0.765 < v6 model 0.769 < best raw label tables 0.893 (llm_labels_v4_blend). Next lever: per-cell v1_weights in loss, more epochs, ensemble with raw blend table.
- ckpt-v1 dataset bumped to v6 weights; NB3 v3 pushed → submission pending. Note: v1_targets does NOT gold-anchor the 58 gold rows (team protocol, no leakage).

## Update 23:30 IST — NB2 v9 5-fold ensemble: gold MACRO 0.7798 (strict OOF 0.7454)
- NB2 v9 trains 5 models (canonical folds, 8 ep each), saves ensemble best.pt (state_dicts list) + oof_preds.csv (4407x12 — team OOF artifact for Daksh's stacking, now also in ckpt-v1 dataset).
- Levers tonight: v7 v1_weights-in-loss REJECTED (0.7503); v8 12ep no gain (0.7649); v9 ensemble +0.011 (0.7798). Label ladder: parser 0.765 < v6 0.769 < v9 ensemble 0.780 < raw blend table 0.893.
- NB3 v4 adapted to ensemble (mean of 5 heads); gate PASS; submission sent (2 left today → 1 after).

## Update 23:50 IST — Council pivot (Option C hybrid) + public fork anchor submitted
- Research (@researcher + official discussion 742106): parser-echo REFUTED — hidden test labels are an INDEPENDENT process, not the team LLM parser. Top LB (~0.952-0.954) = real 2.5D image models (CoAtNet-384/ConvNeXt MIL, 20+ member ensembles, rank fusion).
- @council verdict: Hybrid — keep fast MIL loop for label iteration, fork public 2.5D recipe as LB vehicle, probit/rank-fuse the two; max 3-4 fold-models on our GPU budget; gold-58 stays the hermetic CV.
- Actioned: forked sameerk2004/rsna-knee-sil-d5h-pub0954, stripped to single CoAtNet-384 arm (cells 0/2/8 only, weights dataset nartaa/rsna-knee-publication-swa-weights-20261007) → kernel rsna-knee-pub954-strip v1 → gate PASS → SUBMITTED (LB anchor, expect ~0.949).
- Next (overnight): NB3 v4 ensemble submit via watcher (last sub today), then build 2.5D ConvNeXt fold-0 training notebook for tomorrow's Kaggle GPU.

## Update 23:57 IST — overnight ledger (night shift)
- Submissions today: #1 v4-ckpt MIL (15:20Z), #2 v5 (16:26Z), #3 v6 (17:10Z), #4 pub954-strip CoAtNet-384 single arm (23:47Z, expect ~0.949) — all pending public score. #5 = NB3 v4 ensemble, watcher auto-submits when COMPLETE (if it errors, sub is lost — acceptable).
- NEW kernel rsna-knee-nb4-train25d v1 PUSHED 23:55 (2.5D ConvNeXt-Tiny fold-0, public KneeNet contract verbatim from pub954 own_src, offline in1k weights dataset harshilsrivastav/convnext-tiny-weights, cache-guard 4.5h, expect ~6-7h run). Locally smoke-tested on RTX 4060 (real forward + backward + AUC on 3 studies).
- Morning checklist: (a) check nb4 result -> levers.csv + compare vs 0.7798 champion; (b) if nb4 gold >= 0.80: push full 3-leg fork (c384+c224+cnxt fusion, needs raptor-knee-arms dataset) as submission after UTC reset 05:30 IST; (c) 2.5D fold 1-3 training if GPU allows (~5.5h/fold, council cap 3-4 folds); (d) probit/rank fusion MIL(5-fold OOF) x 2.5D; (e) Daksh teacher-round partial labels merge; (f) Kartik onboarding + TTA ownership.
- GPU budget after nb4: ~18h left this week. Zero interactive Kaggle runs — local smoke first (council risk rule).
- Watcher: /tmp/opencode/night_watcher.sh -> /tmp/opencode/night.log (LB poll every 10 min, only logs changes).

## Update 00:15 IST — LB ANCHOR LANDED: 0.949 public (strip fork)
- Watcher auto-submitted NB3 v4 ensemble at 18:34Z (5th/last sub today). Both our MIL submissions + ensemble now in queue.
- SUBMISSION #4 (pub954-strip CoAtNet-384 single arm) scored **0.949 public LB** — validates mechanics + sets anchor. Top LB ~0.952 (Brandon Low per research).
- NOTE: v1 (15:20Z) and v2 (16:26Z) show completed hidden-rerun but NO public score yet; v3 still pending. Watch for score attachment (or zero-score bug).
- nb4 2.5D fold-0 running, ETA ~06:30 IST. 5 fresh subs at 05:30 IST. Morning checklist in Update 23:57 stands.

## Update 03:30 IST — crash root-caused & fixed; reset watcher armed
- WHY OUR MIL SUBS SCORED NULL: Kaggle rerun error "notebook hit an unhandled error" — debugger found unguarded pydicom.dcmread (no force=True/try), np.stack crashes on mismatched slice sizes, zero per-study fallback, multi-series edge cases. 1 corrupt DICOM in ~1300 hidden studies killed each rerun. FIX in NB3 v5: per-study try/except -> 0.5 fallback row, round-robin multi-series window sampling (proportional, cap 42), MONOCHROME1 inversion, cv2 resize alignment, robust paths. Local 40-study stress test incl. fake corrupted/multi-series/tiny studies: ALL PASS, 0.85s/study mean.
- pub954-full v1 ERROR: cnxt_v0_fold*.pt live in goodpjw2008/rsna-knee-2-5d-convnext-reader (not in raptor-knee-arms current version); own-leg assert is W_OWN-guarded so reader-only fusion is safe. v2 pushed with that dataset added.
- nb4 v1 ERROR: missing os.makedirs('/kaggle/working/own_src') before %%writefile. v2 pushed (mkdir cell inserted).
- reset_watcher2.sh armed (03:24): gates full-fork, then at UTC rollover (05:30 IST) auto-submits fullfork v2 (~0.954 expected) + NB3 v5 (crash-fix). night_watcher keeps LB polling.
- Timing note: NB3 whole 3-study run = 64s wall in-kernel -> hidden ~1300 studies ≈ 4-6h, within 9h cap.

## Update 07:15 IST — MIL ensemble 0.950 public; Kaggle switched to NESTED mounts
- **nb3 v5 (MIL 5-fold ensemble, crash-fixed) = 0.950 public LB** — crash fix validated, at the top of the public LB. levers row added.
- Kaggle mount layout CHANGED overnight: /kaggle/input now nests comp data under /kaggle/input/competitions/<comp>/ and datasets under /kaggle/input/datasets/<owner>/. This silently broke flat-path kernels (nb4 v2/v3) and will break NB1/NB2 on any future rerun (their train.csv globs are flat-only; patch before any rerun). NB3 already robust (sample_submission glob, any depth). nb4 v4: comp_dir derived from resolved train_series.csv; v1_targets/folds/convnext weights via depth-agnostic find_file; diagnostic cell prints mounts + fail-fast.
- Full fork (pub954-full v2) in hidden rerun 7h+ (pending). nb4 v4 ETA ~14:00 IST. 3 subs left today.

## Update 07:55 IST — Kartik handoff shipped; full-fork 0.756 root-caused; nb4 v5 running
- **Kartik handoff package DELIVERED**: handoffs/incoming/harshil/20261016T0205Z-mil-ensemble-pub0950/ (manifest+SHA256SUMS 23 files all verify, score receipts, candidate_levers, README with explicit "Kartik's role is NOT model takeover" scope note). Committed 5db1141 + fixup.
- **Full fork v2 scored 0.756 public** — root cause: pylibjpeg decoder wheels missing from cnxt-reader leg's assets dir → pip install failed silently (check=False) → cnxt leg ran with broken DICOM decode → fusion poisoned the 0.949 c384 leg. Fork author had wheels; we didn't. Fix: harshilsrivastav/pylibjpeg-wheels-cp313 dataset (cp313 wheels) + cell-15 patch (wheels dir + check=True hard-fail). v3 pushed ~08:00, ETA ~15:00 IST.
- **nb4 v5 running** (recursive-glob find_file fix; v4 died at 4-deep paths). ETA ~14:30 IST.
- nb2 hardened for nested mounts (all globs recursive); nb1 verified already-robust; nb3 verified robust.
- Day watcher: /tmp/opencode/day_watcher.sh → /tmp/opencode/day.log (kernel statuses + LB, 10-min cadence, changes only).
- Sub budget today: 3 (full fork v3 will use 1 on gate-pass → 2 left for MIL/fusion).

## Update 08:15 IST — nb5 fusion kernel assembled (MIL 0.950 + c384 0.949 rank-fuse)
- All public DICOMs are uncompressed (ExplicitVR LE, 132 scanned) — cnxt leg decode was fine locally; poison is hidden-set-specific (likely compressed TS needs pylibjpeg — fork author shipped wheels in assets, ours lacked them). v3 hard-fails if wheels missing.
- nb5 = nb3 MIL cells (output _mil.csv) + strip c384 cells (output _c384.csv) + rank-pct-average fusion → submission.csv. Pushed v1. Public run validates assembly; hidden risk = total runtime (MIL ~5.5h + c384 ?) vs 9h cap — decide submission only after full-fork v3 score lands (~15:00 IST) and nb5 public runtime observed.
- Sub budget: 3. Plan: 1 for full fork v3 (if gate+score good), 1 for nb5, 1 reserve.

## Update 08:25 IST — nb5 submitted; code-comp submit flow corrected
- nb5 v3 gated internally after fusion fix (degenerate-MM on 3 public rows was legs' anti-correlation coincidence; MIL-prob tiebreaker + assert added), submitted via competition_submit_code(kernel ref, version) — NOTE: `-f file.csv` legacy submit returns 400 for code comps now; use submit_code with kernel_owner/slug + version.
- Pending: nb5 (ETA ~14:30), full fork v3 (ETA ~15:00), nb4 v5 fold-0 (ETA ~14:30). All three land mid-afternoon.

## Update 09:55 IST — EFFICIENCY METRIC DECODED (user pasted from site)
- Eff = AUC/(Benchmark − maxAUC) + RuntimeSeconds/32400; minimize. Linear trade: 1h runtime ≡ 0.05 AUC.
- Strategy: strip c384 (0.949, ~20min) = efficiency horse; MIL/fusion (~5.5-6h) CANNOT win efficiency (5.5h must buy >0.27 AUC).
- DEADLINE ACTION (repeat until done): team's 2 final selections MUST include strip-c384 run for efficiency eligibility; other slot = best-AUC sub.
- GPU accel accepted: user will provide Daksh token for parallel nb4 folds 1-3.

## Update 11:30 IST — reboot recovery + full-fork v8
- LESSON: /tmp is WIPED on reboot here → all scratch now lives in ~/kaggle/wt/ (persistent). day_watcher relaunched (pid 2600, fixed pgrep self-match).
- Kaggle datasets auto-EXTRACT .zip at ingest (dirs uploaded as dirs get dropped — root cause of the whole abi saga). abi-zips dataset = abi3xx/abi3xx/... trees, import validated locally.
- full-fork v8 pushed 11:21 (decoder cell: recursive .so find + PYTHONPATH=parent, check=True test). RUNNING 11:24+.
- nb4 v5 fold-0 still RUNNING (3.4h in, ETA ~14:30). nb5 fusion rerun score pending (ETA ~14:30).

## Update 13:25 IST — full fork v9 COMPLETE + SUBMITTED (ref 57260570)
- Decoder fix CONFIRMED on Kaggle: [abi] dir abi313/abi313, "decoders ok, python 3.13.15", cnxt reader 27s (3 public studies).
- gate.py PASS (3x13, full spread per column). Output includes _own.csv (our MIL leg ran inside fork).
- v9 rerun ETA ~21:00 IST (c384+cnxt+c224+own-MIL fusion on hidden).
- Subs today: 2/5 used (nb5 ref 57252363 @02:52Z, fullfork v9 ref 57260570 @~07:55Z). 3 left.

## Update 14:30 IST — LEDGER CORRECTION (assignment swap at 00:01Z resolved)
- CORRECTED mapping: nb3 v5 MIL ensemble = 0.756 (NOT 0.950); full-fork v2 (c384+c224+cnxt-poisoned) = 0.950 (NOT 0.756).
- Evidence: submit-time descriptions match kernels; gold-58 predicted MIL 0.7798 (fits 0.756, never fit 0.950); nb5 (MIL+c384) = 0.891 only explicable if MIL leg ≈ 0.756.
- nb5 fusion v3 = 0.891 = NEGATIVE result (rank-fusion with a 0.756 leg drags c384). No more MIL fusions until MIL > ~0.85.
- Subs used today: 2/5 (nb5 0.891, fullfork-v9 PENDING ETA ~21:00 IST).

## Update 13:55 IST — fold 1-3 launch packages ready (in ~/kaggle/wt/nb4-f{1,2,3}/)
- FOLD param set (1/2/3), SEED=42 normalized, OOF filename val_preds_fold{N}.csv (fold-aware).
- make_oof_25d.py: concatenates fold val_preds -> oof_preds_25d.csv for Daksh (StudyInstanceUID + 12 cols + fold col; uniqueness-asserted).
- Launch protocol on fold-0 pass: publish fold-0 artifacts dataset; push nb4-f1 (our quota) immediately; f2/f3 on Kartik token arrival.

## Update ~14:05 IST — folds 1-3 kernels: v1_weights.csv wired in
- nb4 lacked Daksh's per-cell loss weights (98.7% ones, ~1.3% down-weighted noisy cells).
- Patched f1/f2/f3: StudyDataset(weights=), 5-tuple batches, weighted loss = (loss_mat*valid_mask*cw).sum()/valid_mask.sum(), row-aligned via StudyInstanceUID.
- Decision: fold-0 (RUNNING, unweighted) NOT restarted — gold go/no-go signal first; its ckpt = free diverse 5th leg.
- PROTOCOL FIX (user): all further code/notebook surgery -> @coder/@coder-2 (gemini-3.8-flash), mechanical checks -> @grunt. L1 = review + orchestrate only.

## Update ~14:15 IST — decoder-risk audit (reasoning receipt)
- KEY EVIDENCE: strip c384 scored 0.949 on Oct 15 => its hidden rerun SUCCEEDED => hidden DICOMs decode with plain pydicom (uncompressed). NOT a decoder minefield.
- cnxt reader leg needs pylibjpeg unconditionally (its own code path) — fork v7/v8 failures were exactly this; v9 has wheels => safe.
- nb5 fusion kernel has NO wheels dataset: if ever resubmitted, attach pylibjpeg-abi-zips first. Strip + fork v9 = safe tonight.
- Fusion weight plan (day 4): rank-average, per-finding coarse grid {0,0.25,0.5} per leg tuned on gold-58 (58 rows => overfit-risk; smoothing toward uniform).
- Efficiency trim option (medium priority): strip windows/2 => ~10min hidden (eff gain ~0.05); decide after first daily Efficiency Leaderboard sighting.

## Update 14:30 IST — LB deep-dive (6,071 teams) + efficiency intel
- Distribution: top 0.968 (CentralyDS); gold line #10 = 0.964 (9 teams >=0.965); silver #304 & bronze #607 lines = 0.957 (!); fork plateau 0.955-0.960 = 1,400+ teams; we are #1742 @ 0.950 (28.7%).
- KEY: silver AND bronze lines both 0.957. Heliosli public one-click = 0.957 => forked it (rsna-knee-helios-957, RUNNING 14:20). Recipe: public Raptor leg + their private 4-fold ConvNeXt-small (bundles are PUBLIC datasets), rank-blend 0.5/0.5, per-finding rank fusion, fail-closed stages. If clean -> submit tonight -> 0.957 = bronze line public.
- Efficiency LB pulled (ryanholbrook dataset, 5,955 rows, daily 21:27Z). Frontier: rank1-20 AUC 0.947-0.964; 0.963-AUC team ranks #4279 (slow fusion) vs 0.948 team at #16 (fast). RUNTIME separates frontier; AUC alone worthless there.
- Our strip c384 (0.949, ~20min) projects eff-rank ~#85-160 (top 2%). Top-3 eff needs ~0.959+ AUC at <=30min runtime.
- Fast-leg candidates under analysis (explore2): jiweiliu fast-2xt4 (20-member DINO ensemble + CoAt/Raptor arms, dual-T4), evgendvorkin Versia-7 (430 votes).
- Fork push flags learned: is_private=true required for comp-sourced forks; pop docker_image/machine_shape/id_no from pulled metadata.

## Update 14:10 IST — fast-leg analysis verdicts (explore2) + efficiency math
- jiwei fast-2xt4: 0.943 inherited, 8h, 2xT4 mandatory, 20-member DINO fail-closed, ~16 mounts -> NOT a leg; BORROW ideas (shared frozen DINO prefix, dual-GPU scheduling).
- versia-7: 2-branch (1 CoAtNet + Heliosli R7 4-fold ConvNeXt + stackers), 50/50 rank blend, claims ~0.956 unverified, <=8.3h -> FORKED (rsna-knee-versia7, GPU+7 mounts hardened; push queued behind GPU 2-session limit).
- GPU limit: max 2 concurrent GPU kernels/account. Queue: nb4-f0 (til ~15:30) + helios-957 (til ~20-22h); then f1 + versia.
- EFF MATH (benchmark=0.5 assumption): 1h runtime = 0.111 eff; 0.01 AUC = 0.0214 eff. Slow forks (8h) LOSE to strip on efficiency. Strip 0.949@20min = eff-rank ~#85-160. Top-3 eff profile = 0.959+ AUC at <30min -> model-quality bar we can't fake; long shot, top-20 eff realistic.
- TODO: get exact team display name from user -> locate our row in eff LB (tonight's 21:27Z refresh).

## 14:45 IST Oct 16 — fusion plan REVERSED (research audit, wt/research/calibration-deep.md)
- Per-finding weight grids on gold-58 BANNED: this-comp evidence (Devpost audit 0.8217 vs 0.8495 untuned; DiFiore UnifiedTheory; 18th-place HF blog). Rare findings = 1-3 positives in 58 studies -> tuned weights = pure noise, dies on private ~910 studies.
- New fusion recipe: percentile rank-average (rankdata -> [0,1]) of legs, UNIFORM 1/3 weights = primary sub. Optional secondary = ONE global scalar (0.5/0.3/0.2 leg-level), never per-finding. Target 0.958-0.961.
- Daksh v2 guidance changed: soft distillation (BCEWithLogitsLoss on continuous OOF targets, NO 0.9/0.1 cutoffs), gold-58 override w=2.0, silent/ambiguous cells w=0. TEAM-STATUS.md + Downloads copy updated.
- Corroborates our own negatives: nb5 MIL fusion 0.891 dead; every leg-blend that worked (helios 0.957, versia 0.956) is uniform rank fusion.

## 14:52 IST Oct 16 — eff-prize rules stream + frontier correction
- eff-frontier-whatif.md REWRITTEN (grunt mis-fit: benchmark is 0.5 linear form, NOT 1.0; real counts: 167 teams >=0.957 on Oct-15 eff snapshot, ~600 on Oct-16 main LB -> frontier flooding fast).
- Metric CONFIRMED consistent: eff = -(AUC-0.5)/0.467 + RuntimeSeconds/32400, minimize; +0.001 AUC = 69s runtime; 1h = 0.05 AUC.
- RuntimeSeconds = full rerun wall-clock incl. download/preprocess (Sep 28 clarification). Rerun-measured numbers are authoritative; "1xA100" hardware claim DISCARDED (not a Kaggle GPU; hallucination risk).
- Eligibility conflict: rules-agent says best-public auto-qualifies; earlier verified reading = must be among 2 team-selected subs. HEDGE: make fusion run at strip speed (20-30 min) so it is BOTH best-public AND selected -> satisfies both readings. Resolve with our own Legacy row in tonight 03:00 IST eff refresh.
- Stack confirmed: 3 eff prizes stack with medals.

## 14:56 IST Oct 16 — runtime-cuts stream: verified plan (wt/research/runtime-cuts-verified.md)
- We ALREADY run fp16 autocast in strip/fork lineage (researcher partially outdated).
- Concrete lever: swap build_study_table -> studyone_lut (our own _fastread.py has these variants; byte-identical contract) = 40-50% I/O cut; + K_EVAL 94->48 if gold-58 delta <=0.001; + cudnn.benchmark/tf32 audit.
- REJECTED: torch.compile/ONNX (risk, unverifiable offline), two-stage /kaggle/tmp cache (pointless within one run).
- Target: strip 0.949 at ~10-12 min; fusion kernel BORN fast on same path Oct 18-19 (fusion at strip speed = both best-AUC and efficiency horse).
- Gold-gate caveat: strip gold-58 macro is in the 0.75-0.80 band by design (validation split, half the training data, no val tuning) -> do NOT gate the strip on 0.80; that threshold is for the nb4 leg.

## 14:47 IST Oct 16 — v9 = 0.950 (COMPLETE)
- cnxt leg fix added ZERO public AUC (v2 broken=0.950, v9 fixed=0.950). Full-fork lineage closed for submissions; its value is only as a fusion ingredient + private decorrelation, unmeasurable until rerun.
- Sub budget today: 3/5 (57252363, 57260570, 57262514). Remaining plan: versia-7 tonight (#4) + 1 spare.
- Fold-0 still RUNNING at 14:47; gate ~15:30. helios v2 PENDING (~22:00).

## 14:56 IST Oct 16 — SUB QUOTA CORRECTION (UTC clock!)
- Kaggle daily allowance resets at UTC midnight = 05:30 IST. Oct 16 UTC quota had FIVE subs, not 3: 57247325 + 57247347 at 00:01 UTC (05:31 IST) were invisible to our IST-based count. 5/5 burned since 08:58 UTC.
- versia-7 submit BLOCKED tonight; autosubmit armed (nohup pid 10416) to fire 05:35 IST Oct 17 -> rerun scores ~09:00-13:00 IST tomorrow.
- RULE for team: budget subs against the UTC date; verify with the submissions list, not memory; the 400 error body states allowance remaining.
- Tonight now: helios ~22:00 only. Fold gate ~15:30 unchanged.

## 15:05 IST Oct 16 — coder false-completion caught by verification
- strip-v2 surgery v1: coder reported push-ready but grep proved FASTREAD_VARIANT still study_lut (swap absent), VERIFY.md empty, metadata bad keys present. Lesson: notebooks are JSON — a claimed .ipynb edit can silently no-op; ALWAYS grep the artifact after surgery.
- Discovery: strip ships study_lut (not table as assumed) -> target is studyone_lut, modest gain. coder-2 re-spawned with proof-per-task instructions.

## 15:10 IST Oct 16 — EFF ELIGIBILITY SETTLED (primary source: official Efficiency Prize Evaluation page, fetched via rendering proxy)
- VERBATIM: eligible = "among the submissions selected by a team for the Leaderboard Prize, or else among those automatically selected". The rules-researcher claim (best-public auto-qualifies) was WRONG. Manual selection of [best-AUC, strip/fusion] at freeze is MANDATORY.
- maxAUC = max AUC of ALL submissions on the PRIVATE leaderboard (global normalizer). Public 0.967 is our proxy for planning only.
- "A submission may be eligible for both the Leaderboard Prize and the Efficiency Prize" -> fusion-at-strip-speed fully legal.
- Eff prizes = CASH ($7k/$6k/$5k); GPU/CPU notebooks capped 9h (=32400s denominator). Winners obligations: video + open-source code/weights + public model distribution (Nov 5 deadline) - applies if we win.
- Entries: 25,797 entrants / 7,245 participants / 6,071 teams. Team merger deadline PASSED (Oct 15) - team locked.

## 15:02 IST Oct 16 — fusion kernel SPEC written (wt/fusion-scaffold/SPEC.md)
- KEY DESIGN LOCK: fusion runs 2.5D fold-0 SOLO (4-fold inference = 20-40 min, blows 25-min budget). 4-fold ensemble serves OOF/pseudo-labels; f1 added only if measured rerun shows headroom.
- Fusion = percentile rank-average, uniform, fail-closed per leg, receipt json. Legs: strip-c384 (post-cuts ~10-12 min) + 2.5D-f0 (~5-10 min); Daksh leg 3rd only if fits.

## 15:04 IST Oct 16 — win-recipes stream landed (5/5 research complete)
- ACCEPTED: rank-average fusion (confirms), studyone_lut+fp16+cudnn (confirms, in motion), one-global-scalar secondary (confirms calibration-deep).
- DISCARDED: pylibjpeg wheels +0.05-0.10 AUC claim (our v2-vs-v9 A/B = 0.950 vs 0.950, zero delta; wheels = robustness only, already proven on hidden rerun).
- PARKED: DINOv2-small backbone swap (+0.005-0.010, needs 26 GPU-h retrain) - future campaign material.
- Next: synthesis of all 5 streams via synthesizer-2 (big-pickle) -> wt/research/SYNTHESIS.md.

## 15:13 IST Oct 16 — strip-v2 PUSHED (v2 of pub954-strip)
- coder-2 work VERIFIED for real this time: FASTREAD_VARIANT=studyone_lut present in assignment AND in decoded base64 _fastread blob (65,576 b64 chars -> 49,181 chars = byte-identical to full-out7 reference). cudnn.benchmark + 2x allow_tf32 flags in. metadata clean.
- Watcher armed: completion -> MAXDIFF vs _sota_0949.csv must be exactly 0.0 (decode contract) -> then strip-v2 = verified fast decode base for fusion kernel.
- Runtime measurement of the cut comes from the next real submission (tomorrow quota) or the fusion rerun.
