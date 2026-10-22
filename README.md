# RSNA Knee Abnormality Detection

Team entry for the Kaggle code competition [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (October 2026): predict 12 findings (ACL, MCL, both menisci, three OA compartments, effusion, synovitis, Baker's cyst, contusion, fracture) from knee MRI studies. The metric is macro ROC-AUC over the 12 findings.

**Result:** 0.956 macro-AUC on the public leaderboard (6,000+ teams; top score 0.968).

## The problem

Only 58 of the 4,407 training studies have radiologist labels. Every other study has only a free-text report, and reports are not available at test time. The work was therefore mostly about building usable training labels and validating honestly against the 58 gold studies.

## Approach

1. **Weak labels from reports** (`tools/weak_labels.py`, `data/labels-v1/`): report text is converted to per-finding labels with a confidence score; low-confidence and conflicting cells are masked out of the loss.
2. **Folds and gold validation** (`make_folds.py`, `score_gold.py`, `gate.py`): study-level folds, plus a scorer and acceptance gate that evaluate every candidate on the 58 gold studies before it may use a submission slot.
3. **Models** (`src/`): a 2.5D ConvNeXt over sagittal, coronal and axial series with laterality handling and multi-window inputs, trained 4-fold (`kaggle/nb4_train25d.ipynb`).
4. **Final kernel** (`kaggle/nb7_versia12.ipynb`): a rank blend of our 2.5D model with strong public notebook legs, multi-resolution crops, and a runtime budget for the efficiency prize.
5. **Experiment tracking**: `levers.csv` (one isolated change per row with its gold AUC) and `frontier.csv` (accuracy vs. runtime).

## Layout

| Path | Contents |
|---|---|
| `src/` | Dataset, model, windowing, laterality, training and inference |
| `kaggle/` | Kaggle notebooks from feature extraction to final submission |
| `tools/` | Data download, weak labels, overnight training chains, submission builders |
| `data/labels-v1/` | Weak-label tables used for training |
| `runs/` | Training logs for the early runs |

## Team

Harshil Srivastav (models, fusion, submissions), Daksh (labels and validation), Kartik (inference speed and GPU runs).
