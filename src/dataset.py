from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

from src.windows import extract_series_windows

class RSNADataset(Dataset):
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        gold_only: bool = True,
        windows_per_study: int = 8,
        label_cols: Optional[List[str]] = None,
        is_synthetic: bool = False,
        train_csv: Optional[str] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.gold_only = gold_only
        self.windows_per_study = windows_per_study
        self.is_synthetic = is_synthetic

        train_csv_path = Path(train_csv) if train_csv else self.data_root / "train.csv"
        series_csv_path = self.data_root / "train_series.csv"

        df_train = pd.read_csv(train_csv_path)
        df_series = pd.read_csv(series_csv_path)

        ignore_cols = {"StudyInstanceUID", "Report"}
        if label_cols is None:
            self.label_cols = [c for c in df_train.columns if c not in ignore_cols]
        else:
            self.label_cols = label_cols

        if self.gold_only:
            mask = df_train[self.label_cols].notna().any(axis=1)
            df_train = df_train.loc[mask].reset_index(drop=True)

        self.df_train = df_train
        self.df_series = df_series
        self.studies = self.df_train["StudyInstanceUID"].tolist()
        self.series_by_study = {uid: group for uid, group in self.df_series.groupby("StudyInstanceUID")}

    def __len__(self) -> int:
        return len(self.studies)

    def _get_sorted_slice_paths(self, series_dir: Path) -> List[Path]:
        dcm_files = list(series_dir.glob("*.dcm"))
        if not dcm_files:
            return []

        def sort_key(p: Path):
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=True)
                if hasattr(ds, "InstanceNumber") and ds.InstanceNumber is not None:
                    return float(ds.InstanceNumber)
                if hasattr(ds, "ImagePositionPatient") and ds.ImagePositionPatient is not None:
                    return float(ds.ImagePositionPatient[2])
            except Exception:
                pass
            return float("inf")

        return sorted(dcm_files, key=sort_key)

    def _load_series_windows(self, study_uid: str, series_uid: str, n_windows: int) -> np.ndarray:
        series_dir = self.data_root / "train_series" / study_uid / series_uid
        cache_file = series_dir / "windows_cache.npz"
        if cache_file.exists():
            data = np.load(cache_file)
            arr = data["windows"]
            if len(arr) >= n_windows:
                return arr[:n_windows]
            repeats = int(np.ceil(n_windows / len(arr)))
            return np.tile(arr, (repeats, 1, 1, 1))[:n_windows]

        slice_paths = self._get_sorted_slice_paths(series_dir)
        if not slice_paths:
            return np.zeros((n_windows, 5, 224, 224), dtype=np.float32)

        raw_slices = []
        for p in slice_paths:
            ds = pydicom.dcmread(str(p))
            raw_slices.append(ds.pixel_array.astype(np.float32))

        stack = np.stack(raw_slices, axis=0)
        return extract_series_windows(stack, n_windows=n_windows, slices_per_window=5, target_size=224)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        study_uid = self.studies[idx]
        matched_df = self.df_train.loc[self.df_train["StudyInstanceUID"] == study_uid]
        study_row = matched_df.iloc[0]

        labels = study_row[self.label_cols].to_numpy(dtype=np.float32)
        labels_tensor = torch.from_numpy(labels)

        if study_uid in self.series_by_study:
            series_group = self.series_by_study[study_uid]
            series_uids = series_group["SeriesInstanceUID"].tolist()
        else:
            series_uids = []

        if not series_uids:
            dummy_windows = np.zeros((self.windows_per_study, 5, 224, 224), dtype=np.float32)
            return torch.from_numpy(dummy_windows), labels_tensor, study_uid

        num_series = len(series_uids)
        counts = [0] * num_series
        for i in range(self.windows_per_study):
            counts[i % num_series] += 1

        all_windows = []
        for series_uid, k in zip(series_uids, counts):
            if k == 0:
                continue
            w_arr = self._load_series_windows(study_uid, series_uid, n_windows=k)
            all_windows.append(w_arr)

        if all_windows:
            windows_stack = np.concatenate(all_windows, axis=0)
        else:
            windows_stack = np.zeros((self.windows_per_study, 5, 224, 224), dtype=np.float32)

        return torch.from_numpy(windows_stack), labels_tensor, study_uid
