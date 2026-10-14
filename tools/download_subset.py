import os
import argparse
import json
import random
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
import requests
import pandas as pd
from tqdm import tqdm
from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFileRequest

PAGE_LIST_SLEEP = 1.2

def get_auth_token():
    with open(os.path.expanduser('~/.kaggle/access_token'), 'r') as f:
        return f.read().strip()

def get_kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api

def call_with_429_backoff(func, *args, max_attempts=8, initial_backoff=5.0, max_backoff=60.0, **kwargs):
    backoff = initial_backoff
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            status_code = getattr(e.response, "status_code", None) if e.response is not None else None
            if status_code == 429:
                if attempt == max_attempts - 1:
                    raise
                retry_after = None
                if e.response is not None:
                    header_val = e.response.headers.get("Retry-After")
                    if header_val:
                        try:
                            retry_after = float(header_val)
                        except (ValueError, TypeError):
                            pass
                sleep_time = retry_after if retry_after is not None else backoff
                sleep_time += random.uniform(0.0, 1.0)
                time.sleep(sleep_time)
                backoff = min(backoff * 2, max_backoff)
            else:
                raise
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                if attempt == max_attempts - 1:
                    raise
                sleep_time = backoff + random.uniform(0.0, 1.0)
                time.sleep(sleep_time)
                backoff = min(backoff * 2, max_backoff)
            else:
                raise

def get_signed_download_url(api, comp, file_name):
    def _fetch():
        with api.build_kaggle_client() as client:
            req = ApiDownloadDataFileRequest()
            req.competition_name = comp
            req.file_name = file_name
            resp = client.competitions.competition_api_client.download_data_file(req)
            return resp.request.url
    return call_with_429_backoff(_fetch)

def download_file_stream(signed_url, target_path, expected_size=None, session=None):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    part_path = target_path + ".part"
    sess = session or requests
    def _stream():
        with sess.get(signed_url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with open(part_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
    call_with_429_backoff(_stream)
    if expected_size is not None:
        actual_size = os.path.getsize(part_path)
        if actual_size != expected_size:
            if os.path.exists(part_path):
                os.remove(part_path)
            raise IOError(f"Size mismatch: expected {expected_size}, got {actual_size}")
    os.replace(part_path, target_path)

def download_csvs(data_dir, dry_run=False):
    comp = "rsna-knee-abnormality-detection"
    csv_files = [
        "train.csv",
        "train_series.csv",
        "sample_submission.csv",
        "test.csv",
        "test_series.csv"
    ]
    if dry_run:
        for file_name in csv_files:
            print(f"Would download {file_name}")
        return
    api = get_kaggle_api()
    for file_name in csv_files:
        call_with_429_backoff(api.competition_download_file, comp, file_name, path=data_dir, force=True, quiet=False)

def index_files(data_dir, dry_run=False):
    comp = "rsna-knee-abnormality-detection"
    url = f"https://www.kaggle.com/api/v1/competitions/data/list/{comp}?pageSize=500"
    headers = {"Authorization": f"Bearer {get_auth_token()}"}
    session = requests.Session()
    
    jsonl_path = os.path.join(data_dir, "file_index.jsonl")
    progress_path = os.path.join(data_dir, "index.progress.json")
    
    next_page_token = None
    pages_done = 0
    dry_run_files = []

    if not dry_run and os.path.exists(progress_path):
        with open(progress_path, "r") as pf:
            prog = json.load(pf)
            if isinstance(prog, dict):
                next_page_token = prog.get("next_page_token")
                pages_done = prog.get("pages_done", 0)

    total_files_count = 0
    if not dry_run and os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as jf:
            for _ in jf:
                total_files_count += 1

    def fetch_page(token):
        params = {"pageToken": token} if token else {}
        resp = session.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    while True:
        data = call_with_429_backoff(fetch_page, next_page_token)
        page_files = data.get("files", [])
        next_page_token = data.get("nextPageToken")
        pages_done += 1

        if dry_run:
            dry_run_files.extend(page_files)
            token_tail = (next_page_token[-10:] if next_page_token else "none")
            if pages_done % 100 == 0:
                print(f"page {pages_done}, files {len(dry_run_files)}, token tail {token_tail}")
            if len(dry_run_files) >= 1000 or not next_page_token:
                break
            time.sleep(PAGE_LIST_SLEEP)
            continue

        with open(jsonl_path, "a") as jf:
            for item in page_files:
                name = item.get("name") or item.get("nameNullable", "")
                size = item.get("totalBytes", 0)
                jf.write(json.dumps({"path": name, "size_bytes": size}) + "\n")
                total_files_count += 1

        with open(progress_path, "w") as pf:
            json.dump({"next_page_token": next_page_token, "pages_done": pages_done}, pf)

        token_tail = (next_page_token[-10:] if next_page_token else "none")
        if pages_done % 100 == 0:
            print(f"page {pages_done}, files {total_files_count}, token tail {token_tail}")

        if not next_page_token:
            break
        time.sleep(PAGE_LIST_SLEEP)

    if dry_run:
        rows = []
        for item in dry_run_files:
            name = item.get("name") or item.get("nameNullable", "")
            size = item.get("totalBytes", 0)
            parts = name.split("/")
            if len(parts) >= 4 and parts[0] in ("train_series", "test_series"):
                study_uid = parts[1]
                series_uid = parts[2]
                sop_uid = parts[3].rsplit(".", 1)[0]
            else:
                study_uid = None
                series_uid = None
                sop_uid = None
            rows.append({
                "path": name,
                "size_bytes": size,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "sop_uid": sop_uid
            })
        df = pd.DataFrame(rows)
        print(f"Dry run: Fetched {len(df)} files. First 5 rows:")
        print(df.head())
        print(f"Last 5 rows:")
        print(df.tail())
        return

    rows = []
    with open(jsonl_path, "r") as jf:
        for line in jf:
            if not line.strip():
                continue
            entry = json.loads(line)
            name = entry["path"]
            size = entry["size_bytes"]
            parts = name.split("/")
            if len(parts) >= 4 and parts[0] in ("train_series", "test_series"):
                study_uid = parts[1]
                series_uid = parts[2]
                sop_uid = parts[3].rsplit(".", 1)[0]
            else:
                study_uid = None
                series_uid = None
                sop_uid = None
            rows.append({
                "path": name,
                "size_bytes": size,
                "study_uid": study_uid,
                "series_uid": series_uid,
                "sop_uid": sop_uid
            })

    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(data_dir, "file_index.parquet"), index=False)
    if os.path.exists(progress_path):
        os.remove(progress_path)
    print(f"Saved {len(df)} files to {os.path.join(data_dir, 'file_index.parquet')}")

def select_subset(data_dir, budget_gb, dry_run=False):
    train_series_path = os.path.join(data_dir, "train_series.csv")
    file_index_path = os.path.join(data_dir, "file_index.parquet")
    train_series = pd.read_csv(train_series_path)
    train_series = train_series.rename(columns={
        "StudyInstanceUID": "study_uid",
        "SeriesInstanceUID": "series_uid"
    })
    file_index = pd.read_parquet(file_index_path)
    dcm_index = file_index.dropna(subset=['study_uid', 'series_uid']).copy()
    series_bytes = dcm_index.groupby(['study_uid', 'series_uid'])['size_bytes'].sum().reset_index()
    series_merged = series_bytes.merge(train_series, on=['study_uid', 'series_uid'], how='inner')
    
    is_sagittal = series_merged['Anatomical_Plane'].astype(str).str.contains('Sagittal', case=False, na=False)
    is_coronal = series_merged['Anatomical_Plane'].astype(str).str.contains('Coronal', case=False, na=False)
    is_fluid = series_merged['Fluid_Sensitive'].fillna(0).astype(int) == 1

    first_study_mask = ~series_merged.duplicated(subset=['study_uid'])

    def assign_prio(row):
        idx = row.name
        first = first_study_mask.loc[idx]
        sag = is_sagittal.loc[idx]
        cor = is_coronal.loc[idx]
        flu = is_fluid.loc[idx]
        if sag and flu and first:
            return 1
        elif sag and flu:
            return 2
        elif cor and flu:
            return 3
        elif sag:
            return 4
        return 5

    series_merged['priority'] = [assign_prio(row) for _, row in series_merged.iterrows()]
    series_merged = series_merged.sort_values(by=['priority', 'study_uid', 'series_uid'])

    train_csv = pd.read_csv(os.path.join(data_dir, "train.csv"))
    label_cols = [c for c in train_csv.columns if c not in ("StudyInstanceUID", "Report")]
    gold_uids = set(train_csv.dropna(subset=label_cols, how="all")["StudyInstanceUID"])

    best_per_study = series_merged.drop_duplicates(subset=["study_uid"], keep="first").copy()
    best_per_study["gold"] = best_per_study["study_uid"].isin(gold_uids)
    best_per_study = best_per_study.sort_values(by=["gold", "priority", "study_uid"], ascending=[False, True, True])

    series_merged["key"] = list(zip(series_merged["study_uid"], series_merged["series_uid"]))
    budget_bytes = budget_gb * 1024 * 1024 * 1024
    selected = []
    total_bytes = 0
    chosen = set()

    for _, row in best_per_study.iterrows():
        if total_bytes + row["size_bytes"] <= budget_bytes:
            selected.append(row)
            total_bytes += row["size_bytes"]
            chosen.add(row["key"])

    for _, row in series_merged.iterrows():
        if row["key"] in chosen:
            continue
        if row["priority"] >= 3:
            break
        if total_bytes + row["size_bytes"] <= budget_bytes:
            selected.append(row)
            total_bytes += row["size_bytes"]
            chosen.add(row["key"])

    subset_df = pd.DataFrame(selected).drop(columns=["key"])
    n_studies = subset_df["study_uid"].nunique()
    gold_cov = subset_df[subset_df["study_uid"].isin(gold_uids)]["study_uid"].nunique()
    print(f"coverage: {n_studies} studies of {train_csv['StudyInstanceUID'].nunique()}, gold {gold_cov}/{len(gold_uids)}")
    manifest_files = dcm_index.merge(subset_df[['study_uid', 'series_uid']], on=['study_uid', 'series_uid'], how='inner')
    if dry_run:
        print(f"Dry run: Selected {len(subset_df)} series ({len(manifest_files)} files, {total_bytes / (1024**3):.2f} GB) under budget {budget_gb} GB")
    else:
        manifest_files.to_csv(os.path.join(data_dir, "subset_manifest.csv"), index=False)
        print(f"Saved {len(manifest_files)} files across {len(subset_df)} series to {os.path.join(data_dir, 'subset_manifest.csv')}")

def download_subset(data_dir, dry_run=False, workers=12):
    manifest_path = os.path.join(data_dir, "subset_manifest.csv")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    total_size = manifest['size_bytes'].sum()
    free_bytes = shutil.disk_usage(data_dir).free
    required_free = total_size + 5 * 1024 * 1024 * 1024
    if free_bytes < required_free:
        raise RuntimeError(f"Insufficient disk space: free {free_bytes / (1024**3):.2f} GB, required {required_free / (1024**3):.2f} GB")
    if dry_run:
        print(f"Dry run: Would download {len(manifest)} files ({total_size / (1024**3):.2f} GB)")
        return
    api = get_kaggle_api()
    comp = "rsna-knee-abnormality-detection"
    failures_file = os.path.join(data_dir, "download_failures.csv")

    def download_item(item):
        path = item.path
        size_bytes = item.size_bytes
        dest = os.path.join(data_dir, path)
        if os.path.exists(dest) and os.path.getsize(dest) == size_bytes:
            return
        try:
            signed_url = get_signed_download_url(api, comp, path)
            download_file_stream(signed_url, dest, expected_size=size_bytes)
        except Exception as e:
            with open(failures_file, 'a') as f:
                f.write(f"{path},{e}\n")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(tqdm(executor.map(download_item, manifest.itertuples()), total=len(manifest)))

def verify_downloads(data_dir):
    manifest_path = os.path.join(data_dir, "subset_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    missing = []
    size_mismatches = []
    for item in manifest.itertuples():
        dest = os.path.join(data_dir, item.path)
        if not os.path.exists(dest):
            missing.append(item.path)
        elif os.path.getsize(dest) != item.size_bytes:
            size_mismatches.append(item.path)
    print(f"Verified {len(manifest)} files: {len(missing)} missing, {len(size_mismatches)} size mismatches")
    if missing:
        pd.DataFrame({"path": missing}).to_csv(os.path.join(data_dir, "missing_files.csv"), index=False)
    if size_mismatches:
        pd.DataFrame({"path": size_mismatches}).to_csv(os.path.join(data_dir, "size_mismatches.csv"), index=False)

def main():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--data-dir", default="/mnt/ssd/rsna-knee/data", help="Directory to store data")
    parent_parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parent_parser.add_argument("--workers", type=int, default=12, help="Number of workers for parallel downloads")
    parser = argparse.ArgumentParser(description="Download subset of RSNA Knee data", parents=[parent_parser])
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("csvs", parents=[parent_parser], help="Download CSV files")
    subparsers.add_parser("index", parents=[parent_parser], help="Index all files")
    select_parser = subparsers.add_parser("select", parents=[parent_parser], help="Select subset of data")
    select_parser.add_argument("--budget-gb", type=int, default=55, help="Budget in GB")
    subparsers.add_parser("download", parents=[parent_parser], help="Download selected subset")
    subparsers.add_parser("verify", parents=[parent_parser], help="Verify downloads")
    args = parser.parse_args()
    os.makedirs(args.data_dir, exist_ok=True)
    if args.command == "csvs":
        download_csvs(args.data_dir, args.dry_run)
    elif args.command == "index":
        index_files(args.data_dir, args.dry_run)
    elif args.command == "select":
        select_subset(args.data_dir, args.budget_gb, args.dry_run)
    elif args.command == "download":
        download_subset(args.data_dir, args.dry_run, args.workers)
    elif args.command == "verify":
        verify_downloads(args.data_dir)

if __name__ == "__main__":
    main()
