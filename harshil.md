# Harshil — Model Handoff Request

Prepare a read-only model handoff for Kartik from the current RSNA Knee project. Do not redesign, retrain, optimize, or reinterpret anything.

Deliver a self-contained package under:

```text
handoffs/incoming/harshil/<YYYYMMDDTHHMMZ-short-name>/
├── HANDOFF_MANIFEST.yaml
├── README.md
├── SHA256SUMS
├── source/
├── configs/
├── receipts/
└── artifacts/
```

The manifest must identify the delivery, owner, creation time, purpose, required phases, every file and SHA-256, immutable Kaggle references for large artifacts, external-data status, known issues, and the exact reproduction command.

Include:

- inference and notebook source;
- configurations;
- exact checkpoint Kaggle slug, version, filename, and SHA-256;
- model input/output contract;
- preprocessing and series-selection rules;
- orientation and laterality handling;
- crop, windows, views/TTA, precision, and aggregation;
- output semantics;
- label and fold lineage;
- all external-data lineage, including OAI;
- Gold58 reuse information;
- successful logs, score receipts, and runtime evidence;
- known fallback behavior and limitations.

Provide `levers.csv` with:

```text
config,gold_auc,owner,date,notes
```

Provide `candidate_levers.csv` with:

```text
candidate_id,parent_id,isolated_change,score_type,score_value,score_studies,
prediction_sha256,truth_sha256,fold_sha256,checkpoint_sha256,external_data,
runtime_seconds,runtime_basis,notes
```

Nominate an `accuracy_anchor` and an `efficiency_anchor`. If anything is missing, write `UNAVAILABLE` with the reason and ETA. Verify every hash and report undeclared absolute paths or network downloads.

Every score receipt must include:

```text
candidate_id,score_type,macro_auc,per_class_auc,n_expected,n_matched,
prediction_sha256,truth_sha256,fold_sha256,scorer_sha256,created_utc
```

The delivery will be rejected for missing hashes, ambiguous score sources, undeclared dependencies, omitted external-data lineage, silent Gold58 contamination, absent reproduction instructions, or files that overwrite another owner's canonical work.

Never include credentials, competition DICOMs, or raw reports.
