import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, UID

from src.windows import extract_series_windows

def make_gaussian_slice(h: int, w: int, center_y: float, center_x: float, sigma: float, amplitude: float) -> np.ndarray:
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    dist_sq = (y - center_y) ** 2 + (x - center_x) ** 2
    blob = amplitude * np.exp(-dist_sq / (2.0 * sigma ** 2))
    noise = np.random.normal(100.0, 10.0, size=(h, w))
    img = np.clip(noise + blob, 0, 65535).astype(np.uint16)
    return img

def create_dicom(
    path: Path,
    pixel_array: np.ndarray,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    instance_num: int,
    plane: str,
    laterality: str,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = UID(sop_uid)
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.SOPInstanceUID = sop_uid
    ds.SOPClassUID = MRImageStorage
    ds.InstanceNumber = instance_num

    if laterality in ("L", "R"):
        ds.ImageLaterality = laterality
    if plane == "Axial":
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    elif plane == "Coronal":
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    else:
        ds.ImageOrientationPatient = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]

    ds.ImagePositionPatient = [0.0, 0.0, float(instance_num * 3)]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = int(pixel_array.shape[0]), int(pixel_array.shape[1])
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = pixel_array.tobytes()
    ds.save_as(str(path), write_like_original=False)

def generate_fake_dataset(out_dir: Path = Path("/tmp/opencode/rsna_fake"), seed: int = 7) -> None:
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    series_dir = out_dir / "train_series"
    series_dir.mkdir(parents=True, exist_ok=True)

    conditions = [f"Condition_{i}" for i in range(12)]
    studies = [f"Study_{i}" for i in range(5)]
    planes = [("Sagittal", "Sagittal_T2", "No"), ("Coronal", "Coronal_PD", "Yes"), ("Axial", "Axial_T2", "Yes")]

    train_rows = []
    train_series_rows = []

    for s_idx, study_uid in enumerate(studies):
        is_gold = s_idx < 3
        row: dict = {"StudyInstanceUID": study_uid, "Report": f"Synthetic report for {study_uid}"}
        pos = (s_idx % 2 == 1)
        for c_idx, c_name in enumerate(conditions):
            if is_gold:
                row[c_name] = 1.0 if (pos and c_idx % 2 == 0) or (not pos and c_idx % 3 == 0) else 0.0
            else:
                row[c_name] = np.nan
        train_rows.append(row)

        laterality = "L" if s_idx % 2 == 0 else "R"
        if s_idx == 4:
            laterality = "Unknown"

        for p_idx, (plane, fluid, fat) in enumerate(planes):
            series_uid = f"{study_uid}_Series_{p_idx}"
            train_series_rows.append({
                "StudyInstanceUID": study_uid,
                "SeriesInstanceUID": series_uid,
                "Anatomical_Plane": plane,
                "Fluid_Sensitive": fluid,
                "Fat_Suppression": fat,
            })

            s_path = series_dir / study_uid / series_uid
            s_path.mkdir(parents=True, exist_ok=True)

            amp = 3000.0 if (pos and is_gold) else 1000.0
            raw_slices = []
            for sl_idx in range(20):
                sop_uid = f"1.2.826.0.1.3680043.8.498.{s_idx}.{p_idx}.{sl_idx}"
                dcm_file = s_path / f"{sop_uid}.dcm"
                cy = 80 + 10 * math.sin(sl_idx / 3.0)
                cx = 80 + 10 * math.cos(sl_idx / 3.0)
                img = make_gaussian_slice(160, 160, cy, cx, sigma=15.0, amplitude=amp)
                create_dicom(dcm_file, img, study_uid, series_uid, sop_uid, sl_idx + 1, plane, laterality)
                raw_slices.append(img)

            raw_stack = np.stack(raw_slices, axis=0)
            windows_arr = extract_series_windows(raw_stack, n_windows=4, slices_per_window=5, target_size=224)
            np.savez_compressed(s_path / "windows_cache.npz", windows=windows_arr)

    pd.DataFrame(train_rows).to_csv(out_dir / "train.csv", index=False)
    pd.DataFrame(train_series_rows).to_csv(out_dir / "train_series.csv", index=False)
    print(f"Fake dataset generated at {out_dir}")

if __name__ == "__main__":
    generate_fake_dataset()
