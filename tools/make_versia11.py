#!/usr/bin/env python3
"""
make_versia11.py — Configurable generator for versia-11 notebook (Daily Submission #5).
Generates rsna-knee-versia11.ipynb from the robust 3-way subprocess base (versia-8/10),
inserting custom weights for R7 (ours), Raptor, and SWA CoAtNet-384.
"""

import argparse
import json
import os
import py_compile
import tempfile

def build_versia11(w_ours: float, w_raptor: float, w_sota: float, description: str):
    base_nb_path = '/home/harshil/kaggle/wt/versia8/rsna-knee-versia8.ipynb'
    out_dir = '/home/harshil/kaggle/wt/versia11'
    os.makedirs(out_dir, exist_ok=True)
    out_nb_path = os.path.join(out_dir, 'rsna-knee-versia11.ipynb')

    with open(base_nb_path) as f:
        nb = json.load(f)

    stage3_code = f"""# Stage 3: Versia-11 Multi-Model Rank Blend ({description})
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path('/kaggle/input/rsna-knee-abnormality-detection')
if not (DATA / 'test.csv').is_file():
    DATA = next(Path('/kaggle/input').glob('**/test.csv')).parent

WORK = Path('/kaggle/working')

# Blend weights configured for versia-11:
W_OURS = {w_ours:.4f}
W_RAPTOR = {w_raptor:.4f}
W_SOTA = {w_sota:.4f}

test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
n = len(test_ids)
ss = DATA / 'sample_submission.csv'

# Load Arm 1: Raptor
raptor_cand = WORK / 'raptor_sub.csv'
raptor = pd.read_csv(raptor_cand, dtype={{'StudyInstanceUID': str}}) if raptor_cand.is_file() else None

# Load Arm 2: Heliosli R7
ours_cand = WORK / 'ours_sub.csv'
ours = pd.read_csv(ours_cand, dtype={{'StudyInstanceUID': str}}) if ours_cand.is_file() else None

# Load Arm 3: SWA CoAtNet-384
sota_cand = WORK / '_sota_0949.csv'
if not sota_cand.is_file():
    sota_cand = WORK / 'submission.csv'
sota = pd.read_csv(sota_cand, dtype={{'StudyInstanceUID': str}}) if sota_cand.is_file() else None

cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else (list(ours.columns) if ours is not None else list(raptor.columns))
labels = cols[1:]

def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values

def get_ranks(df, name):
    if df is None:
        print(f"Warning: {{name}} is None, treating as failed")
        return None, np.ones(n, dtype=bool)
    df = df.set_index('StudyInstanceUID').reindex(test_ids)
    failed = df[labels].isna().any(axis=1).values | (df[labels].values == 0.5).all(axis=1)
    ranks = np.full((n, len(labels)), np.nan)
    for j, c in enumerate(labels):
        vals = df[c].values
        valid = ~failed & np.isfinite(vals)
        if valid.any():
            ranks[valid, j] = pct(vals[valid])
    return ranks, failed

r_raptor, fail_raptor = get_ranks(raptor, 'Raptor')
r_ours, fail_ours = get_ranks(ours, 'Heliosli R7')
r_sota, fail_sota = get_ranks(sota, 'SWA CoAtNet-384')

out = np.zeros((n, len(labels)))

for j, c in enumerate(labels):
    for i in range(n):
        cands = []
        w_list = []
        if r_ours is not None and not fail_ours[i] and np.isfinite(r_ours[i, j]):
            cands.append(r_ours[i, j])
            w_list.append(W_OURS)
        if r_raptor is not None and not fail_raptor[i] and np.isfinite(r_raptor[i, j]):
            cands.append(r_raptor[i, j])
            w_list.append(W_RAPTOR)
        if r_sota is not None and not fail_sota[i] and np.isfinite(r_sota[i, j]):
            cands.append(r_sota[i, j])
            w_list.append(W_SOTA)
        
        if cands:
            w_arr = np.array(w_list, dtype=np.float64)
            w_arr /= w_arr.sum()
            out[i, j] = np.sum(np.array(cands) * w_arr)
        else:
            out[i, j] = 0.5

sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert sub['StudyInstanceUID'].tolist() == test_ids, "StudyInstanceUID mismatch"
assert np.isfinite(sub[labels].values).all(), "NaNs detected in predictions"
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all(), "Predictions outside [0, 1]"

out_path = WORK / 'submission.csv'
sub.to_csv(out_path, index=False)
print("=== VERSIA-11 RANK BLEND SUCCESS ===", flush=True)
print(f"Saved {{out_path}} ({{len(sub)}} rows x {{len(sub.columns)}} cols)", flush=True)
print(sub.head().to_string(index=False), flush=True)

receipt = {{
    "mode": "versia11_rank_blend",
    "description": "{description}",
    "studies": n,
    "weights": {{"ours_r7": W_OURS, "raptor": W_RAPTOR, "swa384": W_SOTA}},
    "failed_counts": {{
        "ours_r7": int(fail_ours.sum()),
        "raptor": int(fail_raptor.sum()),
        "swa384": int(fail_sota.sum())
    }}
}}
with open(WORK / 'blend_receipt.json', 'w') as f:
    json.dump(receipt, f, indent=2)
print("Saved blend_receipt.json", flush=True)
"""

    nb['cells'][13]['source'] = [l + '\n' for l in stage3_code.splitlines()[:-1]] + [stage3_code.splitlines()[-1]]

    with open(out_nb_path, 'w') as f:
        json.dump(nb, f, indent=1)

    print(f"Generated {out_nb_path} with W_OURS={w_ours}, W_RAPTOR={w_raptor}, W_SOTA={w_sota}")

    # Syntax verification
    full_code = []
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            full_code.append(''.join(c['source']))
    code_str = '\n\n'.join(full_code)
    clean_lines = [f"# {l}" if l.startswith('%') else l for l in code_str.splitlines()]
    clean_code = '\n'.join(clean_lines)

    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(clean_code)
        tmp_path = f.name
    try:
        py_compile.compile(tmp_path, doraise=True)
        print("py_compile: rsna-knee-versia11.ipynb ALL CELLS COMPILE 100% CLEANLY!")
    finally:
        os.unlink(tmp_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--w-ours', type=float, default=0.80)
    parser.add_argument('--w-raptor', type=float, default=0.10)
    parser.add_argument('--w-sota', type=float, default=0.10)
    parser.add_argument('--desc', type=str, default='High-Confidence R7 Variance Dampener')
    args = parser.parse_args()
    build_versia11(args.w_ours, args.w_raptor, args.w_sota, args.desc)
