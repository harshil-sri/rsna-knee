import os
import argparse
import requests
import pandas as pd
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFileRequest

def get_auth_token():
    with open(os.path.expanduser('~/.kaggle/access_token'), 'r') as f:
        return f.read().strip()

def get_kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api

def get_signed_download_url(api, comp, file_name):
    with api.build_kaggle_client() as client:
        req = ApiDownloadDataFileRequest()
        req.competition_name = comp
        req.file_name = file_name
        resp = client.competitions.competition_api_client.download_data_file(req)
        return resp.request.url

def download_file_stream(signed_url, target_path, expected_size=None, session=None):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    part_path = target_path + ".part"
    sess = session or requests
    with sess.get(signed_url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(part_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
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
        api.competition_download_file(comp, file_name, path=data_dir, force=True, quiet=False)

def index_files(data_dir, dry_run=False):
    comp = "rsna-knee-abnormality-detection"
    url = f"https://www.kaggle.com/api/v1/competitions/data/list/{comp}?pageSize=500"
    headers = {"Authorization": f"Bearer {get_auth_token()}"}
    session = requests.Session()
    next_page_token = None
    files = []
    while True:
        params = {"pageToken": next_page_token} if next_page_token else {}
        response = session.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        page_files = data.get("files", [])
        files.extend(page_files)
        next_page_token = data.get("nextPageToken")
        if dry_run and len(files) >= 1000:
            break
        if not next_page_token:
            break
    rows = []
    for item in files:
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
    if dry_run:
        print(f"Dry run: Fetched {len(df)} files. First 5 rows:")
        print(df.head())
        print(f"Last 5 rows:")
        print(df.tail())
    else:
        df.to_parquet(os.path.join(data_dir, "file_index.parquet"), index=False)
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
    budget_bytes = budget_gb * 1024 * 1024 * 1024
    selected = []
    total_bytes = 0
    for _, row in series_merged.iterrows():
        if total_bytes + row['size_bytes'] <= budget_bytes:
            selected.append(row)
            total_bytes += row['size_bytes']
    subset_df = pd.DataFrame(selected)
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
        last_error = None
        for attempt in range(3):
            try:
                signed_url = get_signed_download_url(api, comp, path)
                download_file_stream(signed_url, dest, expected_size=size_bytes)
                return
            except Exception as e:
                last_error = e
                time.sleep(2 ** attempt)
        with open(failures_file, 'a') as f:
            f.write(f"{path},{last_error}\n")

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
