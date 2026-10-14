import argparse
import random
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.dataset import RSNADataset
from src.model import RSNAViTMIL


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_macro_auc(targets: np.ndarray, preds: np.ndarray) -> float:
    aucs: List[float] = []
    num_classes = targets.shape[1]
    for c in range(num_classes):
        t = targets[:, c]
        p = preds[:, c]
        if np.isnan(t).all():
            continue
        valid_mask = ~np.isnan(t)
        t_valid = t[valid_mask]
        p_valid = p[valid_mask]
        binary_t = (t_valid > 0.5).astype(np.int32)
        if np.sum(binary_t == 1) < 2 or np.sum(binary_t == 0) < 1:
            continue
        try:
            auc = roc_auc_score(binary_t, p_valid)
            aucs.append(float(auc))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else 0.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--windows", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)

    data_root = Path("/mnt/ssd/rsna-knee/data")
    train_csv = data_root / "train_weak.csv"

    dataset = RSNADataset(
        data_root=data_root,
        gold_only=False,
        windows_per_study=args.windows,
        train_csv=str(train_csv),
    )

    train_series_dir = data_root / "train_series"
    df = dataset.df_train.copy()
    label_cols = dataset.label_cols

    def has_images(uid: str) -> bool:
        study_path = train_series_dir / str(uid)
        return study_path.is_dir()

    df["has_images"] = df["StudyInstanceUID"].apply(has_images)
    df_valid = df[df["has_images"]].copy()

    df_valid["pos_count"] = (df_valid[label_cols] > 0.5).sum(axis=1)
    df_sorted = df_valid.sort_values(by="pos_count", ascending=False).reset_index(drop=True)
    selected_uids = df_sorted["StudyInstanceUID"].iloc[: args.studies].tolist()

    uid_to_idx = {uid: i for i, uid in enumerate(dataset.studies)}
    selected_indices = [uid_to_idx[uid] for uid in selected_uids if uid in uid_to_idx]

    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=4, shuffle=True, drop_last=False)
    eval_loader = DataLoader(subset, batch_size=4, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSNAViTMIL(
        model_name="vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        in_chans=5,
        num_classes=len(label_cols),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    final_loss = 0.0
    final_auc = 0.5

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total_batches = 0

        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(images)
                    valid = ~torch.isnan(targets)
                    loss = criterion(logits[valid], targets[valid])
            else:
                logits = model(images)
                valid = ~torch.isnan(targets)
                loss = criterion(logits[valid], targets[valid])

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        epoch_loss = running_loss / max(1, total_batches)

        if epoch % 5 == 0 or epoch == args.epochs:
            model.eval()
            all_preds: List[np.ndarray] = []
            all_targets: List[np.ndarray] = []
            with torch.no_grad():
                for images, targets, _ in eval_loader:
                    images = images.to(device)
                    if device.type == "cuda":
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            logits = model(images)
                    else:
                        logits = model(images)
                    probs = torch.sigmoid(logits).float().cpu().numpy()
                    all_preds.append(probs)
                    all_targets.append(targets.numpy())

            y_pred = np.concatenate(all_preds, axis=0)
            y_true = np.concatenate(all_targets, axis=0)
            epoch_auc = compute_macro_auc(y_true, y_pred)
            print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Macro-AUC: {epoch_auc:.4f}")

            if epoch == args.epochs:
                final_loss = epoch_loss
                final_auc = epoch_auc

    if final_auc >= 0.90 and final_loss <= 0.15:
        print("OVERFIT OK")
    else:
        print("OVERFIT FAILED — pipeline bug suspected")


if __name__ == "__main__":
    main()
