#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_gold_auc(train, pred, uid="StudyInstanceUID"):
    labels = [c for c in train.columns if c != uid and c in pred.columns]
    gold = train.dropna(subset=labels, how="all")
    gold_uids = set(gold[uid])
    matched = gold[gold[uid].isin(set(pred[uid]))]

    per_class = {}
    for c in labels:
        sub = matched[[uid, c]].merge(pred[[uid, c]], on=uid, suffixes=("_t", "_p"))
        y = sub[f"{c}_t"].astype(int).to_numpy()
        p = sub[f"{c}_p"].astype(float).to_numpy()
        if len(np.unique(y)) < 2:
            per_class[c] = float("nan")
            continue
        per_class[c] = float(roc_auc_score(y, p))

    vals = [v for v in per_class.values() if not np.isnan(v)]
    macro = float(np.mean(vals)) if vals else float("nan")
    return macro, per_class, len(gold_uids), len(matched)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--uid", default="StudyInstanceUID")
    args = ap.parse_args()

    train = pd.read_csv(args.train)
    pred = pd.read_csv(args.pred)

    macro, per_class, n_gold, n_matched = macro_gold_auc(train, pred, args.uid)

    print(f"gold studies in train: {n_gold}   matched in preds: {n_matched}")
    for c, v in per_class.items():
        shown = "n/a" if np.isnan(v) else f"{v:.4f}"
        print(f"  {c:<20} {shown}")
    print(f"  {'MACRO':<20} {macro:.4f}")


if __name__ == "__main__":
    main()
