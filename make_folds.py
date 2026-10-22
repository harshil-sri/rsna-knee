#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd


def make_folds(uids, n=5, seed=42):
    uids = np.sort(np.asarray(sorted(set(uids))))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uids))
    fold = np.empty(len(uids), dtype=int)
    fold[perm] = np.arange(len(uids)) % n
    return pd.DataFrame({"StudyInstanceUID": uids, "fold": fold})


def merge_canonical(fold_map, series, uid="StudyInstanceUID") -> pd.DataFrame:
    base = series[[uid]].drop_duplicates().reset_index(drop=True)
    folds = base[uid].map(fold_map)
    missing = int(folds.isna().sum())
    if missing:
        raise SystemExit(f"{missing} series' studies absent from canonical fold map")
    return pd.DataFrame({uid: base[uid].to_numpy(), "fold": folds.to_numpy(dtype=int)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True, help="train_series.csv")
    ap.add_argument("--out", default="folds.csv")
    ap.add_argument("--canonical", default=None, help="optional public fold csv with uid+fold")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--uid", default="StudyInstanceUID")
    args = ap.parse_args()

    series = pd.read_csv(args.series, usecols=[args.uid])
    series = series.rename(columns={args.uid: "StudyInstanceUID"})

    out: pd.DataFrame
    if args.canonical:
        can = pd.read_csv(args.canonical)
        fold_col = "fold" if "fold" in can.columns else can.columns[-1]
        fmap = dict(zip(can["StudyInstanceUID"], can[fold_col]))
        out = merge_canonical(fmap, series)
        source = f"canonical:{args.canonical}"
    else:
        out = make_folds(series["StudyInstanceUID"].unique(), args.n, args.seed)
        source = f"synthetic grouped {args.n}-fold seed={args.seed}"

    out.to_csv(args.out, index=False)
    print(f"source: {source}")
    print(f"studies: {out['StudyInstanceUID'].nunique()}  ->  {args.out}")
    print(out["fold"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
