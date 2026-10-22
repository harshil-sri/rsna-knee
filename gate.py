#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import pandas as pd


def check(sub, sample, uid="StudyInstanceUID", rtol=1e-3):
    problems = []
    exp_cols = list(sample.columns)
    got_cols = list(sub.columns)

    if got_cols != exp_cols:
        missing = [c for c in exp_cols if c not in got_cols]
        extra = [c for c in got_cols if c not in exp_cols]
        problems.append(f"column mismatch: missing={missing} extra={extra} order_ok={got_cols == [c for c in got_cols if c in exp_cols]}")
        return problems

    if len(sub) != len(sample):
        problems.append(f"row count {len(sub)} != expected {len(sample)}")

    if sub[uid].duplicated().any():
        problems.append(f"duplicate {uid} values: {int(sub[uid].duplicated().sum())}")

    exp_uids = set(sample[uid])
    got_uids = set(sub[uid])
    if exp_uids != got_uids:
        problems.append(
            f"uid set mismatch: missing={len(exp_uids - got_uids)} unexpected={len(got_uids - exp_uids)}"
        )

    label_cols = [c for c in exp_cols if c != uid]
    vals = sub[label_cols]
    if vals.isna().any().any():
        problems.append(f"NaN cells: {int(vals.isna().sum().sum())}")
    arr = vals.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        problems.append("non-finite values present")
    for c in label_cols:
        col = sub[c].to_numpy(dtype=float)
        if col.min() < 0.0 or col.max() > 1.0:
            problems.append(f"{c}: out of [0,1] min={col.min():.4g} max={col.max():.4g}")
        if col.std() < rtol:
            problems.append(f"{c}: near-constant (std={col.std():.3g}) -> degenerate AUC")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--uid", default="StudyInstanceUID")
    args = ap.parse_args()

    sub = pd.read_csv(args.sub)
    sample = pd.read_csv(args.sample)

    problems = check(sub, sample, args.uid)
    if problems:
        print("GATE: FAIL")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("GATE: PASS")
    print(f"  rows={len(sub)} cols={len(sub.columns)}")


if __name__ == "__main__":
    main()
