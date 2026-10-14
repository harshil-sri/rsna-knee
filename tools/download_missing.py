import os
import sys
import time
import json
import random
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.zip_subset import (
    get_signed_url,
    zip64_eocd,
    fetch_central_dir,
    parse_central_dir,
    fetch_member,
)

INDEX_PATH = "/mnt/ssd/rsna-knee/data/file_index.jsonl"
SERIES_CSV = "/mnt/ssd/rsna-knee/data/train_series.csv"
EXISTING_TRAIN_DIR = "/mnt/ssd/rsna-knee/data/train_series"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="/home/harshil/rsna-ext/data")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dest_root = os.path.abspath(args.dest)
    os.makedirs(dest_root, exist_ok=True)
    failures_csv = os.path.join(dest_root, "dl_failures.csv")

    df = pd.read_csv(SERIES_CSV)
    all_studies = set(df["StudyInstanceUID"].astype(str))
    existing_studies = set(os.listdir(EXISTING_TRAIN_DIR)) if os.path.exists(EXISTING_TRAIN_DIR) else set()
    missing_studies = all_studies - existing_studies

    studies_files = {}
    filtered_count = 0
    filtered_bytes = 0

    with open(INDEX_PATH, "r") as f:
        for line in f:
            item = json.loads(line)
            path = item["path"]
            parts = path.split("/")
            if len(parts) >= 4 and parts[0] == "train_series" and parts[1] in missing_studies:
                study_uid = parts[1]
                size_bytes = item["size_bytes"]
                if study_uid not in studies_files:
                    studies_files[study_uid] = []
                studies_files[study_uid].append((path, size_bytes))
                filtered_count += 1
                filtered_bytes += size_bytes

    print(f"Filtered: {len(studies_files)} missing studies, {filtered_count} files, {filtered_bytes / 1e9:.1f} GB", flush=True)

    random.seed(42)
    study_keys = list(studies_files.keys())
    study_keys.sort()
    random.shuffle(study_keys)

    ordered_files = []
    for s in study_keys:
        files = studies_files[s]
        files.sort(key=lambda x: x[0])
        random.shuffle(files)
        ordered_files.extend(files)

    if args.limit > 0:
        ordered_files = ordered_files[:args.limit]

    signed = get_signed_url()
    cd_off, cd_size = zip64_eocd(signed)
    blob = fetch_central_dir(signed, cd_off, cd_size)
    cd = parse_central_dir(blob)

    path_to_idx = {}
    nb = cd["names_blob"]
    base = 0
    for idx, endo in enumerate(cd["name_offs"]):
        path_to_idx[nb[base:endo].decode()] = idx
        base = endo

    t0 = time.time()
    processed = 0
    total = len(ordered_files)

    for path, expected_size in ordered_files:
        dest_file = os.path.join(dest_root, path)
        if os.path.exists(dest_file) and os.path.getsize(dest_file) == expected_size:
            processed += 1
            if processed % 500 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta_hours = (total - processed) / rate / 3600 if rate > 0 else 0.0
                print(f"Progress: {processed}/{total} ({rate:.1f} files/s, ETA: {eta_hours:.2f}h)", flush=True)
            continue

        cd_idx = path_to_idx.get(path)
        if cd_idx is None:
            err_msg = "not found in central directory"
            with open(failures_csv, "a") as f:
                f.write(f'"{path}","{err_msg}"\n')
            processed += 1
            continue

        try:
            fetch_member(signed, cd, cd_idx, dest_file)
        except Exception as e:
            with open(failures_csv, "a") as f:
                f.write(f'"{path}","{e}"\n')

        processed += 1
        if processed % 500 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0.0
            eta_hours = (total - processed) / rate / 3600 if rate > 0 else 0.0
            print(f"Progress: {processed}/{total} ({rate:.1f} files/s, ETA: {eta_hours:.2f}h)", flush=True)

    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0.0
    print(f"Done: {processed}/{total} files in {elapsed:.1f}s ({rate:.1f} files/s)", flush=True)


if __name__ == "__main__":
    main()
