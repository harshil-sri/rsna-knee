from typing import List
import cv2
import numpy as np

def normalize_percentile(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    p_low = np.percentile(arr, low)
    p_high = np.percentile(arr, high)
    if p_high - p_low < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    clipped = np.clip(arr, p_low, p_high)
    return ((clipped - p_low) / (p_high - p_low)).astype(np.float32)

def resize_volume(vol: np.ndarray, target_size: int = 224) -> np.ndarray:
    d, h, w = vol.shape
    if h == target_size and w == target_size:
        return vol.astype(np.float32)
    resized = np.zeros((d, target_size, target_size), dtype=np.float32)
    for i in range(d):
        resized[i] = cv2.resize(vol[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return resized

def extract_series_windows(
    slices: np.ndarray,
    n_windows: int,
    slices_per_window: int = 5,
    target_size: int = 224,
) -> np.ndarray:
    num_slices = slices.shape[0]
    norm_slices = normalize_percentile(slices)
    resized_slices = resize_volume(norm_slices, target_size=target_size)

    if num_slices < slices_per_window:
        pad_count = slices_per_window - num_slices
        last_slice = resized_slices[-1:]
        padding = np.repeat(last_slice, pad_count, axis=0)
        resized_slices = np.concatenate([resized_slices, padding], axis=0)
        num_slices = slices_per_window

    max_start = num_slices - slices_per_window
    if max_start == 0:
        start_indices = [0] * n_windows
    else:
        start_indices = np.linspace(0, max_start, n_windows, dtype=int).tolist()

    windows = np.zeros((n_windows, slices_per_window, target_size, target_size), dtype=np.float32)
    for idx, start in enumerate(start_indices):
        windows[idx] = resized_slices[start : start + slices_per_window]
    return windows
