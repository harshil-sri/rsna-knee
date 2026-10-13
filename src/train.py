import argparse
import os
import random
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.config import CKPT_DIR, DATA_ROOT, LR, SEED, SMOKE_FAKE_DIR, WEIGHT_DECAY
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
        if np.sum(t_valid == 1) < 2 or np.sum(t_valid == 0) < 1:
            continue
        try:
            auc = roc_auc_score(t_valid, p_valid)
            aucs.append(float(auc))
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else 0.5

def train_fn(
    fold: int = 0,
    smoke: bool = False,
    epochs: int = 2,
    batch_size: int = 4,
    labels_path: Optional[str] = None,
    out_dir: Optional[str] = None,
):
    seed_everything(SEED)
    save_dir = Path(out_dir) if out_dir else CKPT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = SMOKE_FAKE_DIR if smoke else Path(DATA_ROOT)
    if labels_path:
        labels_df = pd.read_csv(labels_path)
        ignore_cols = {"StudyInstanceUID", "Report"}
        label_cols = [c for c in labels_df.columns if c not in ignore_cols]
    else:
        label_cols = None

    windows_per_study = 8 if smoke else 42
    full_dataset = RSNADataset(
        data_root=data_dir,
        gold_only=True,
        windows_per_study=windows_per_study,
        label_cols=label_cols,
        is_synthetic=smoke,
    )

    study_uids = full_dataset.studies
    num_studies = len(study_uids)
    if num_studies == 0:
        raise ValueError("No gold studies found in dataset.")

    n_splits = min(5, num_studies)
    if n_splits > 1:
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(study_uids, groups=study_uids))
        train_idx_arr, val_idx_arr = splits[fold % n_splits]
        train_idx = [int(i) for i in train_idx_arr]
        val_idx = [int(i) for i in val_idx_arr]
    else:
        train_idx = list(range(num_studies))
        val_idx = list(range(num_studies))

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    model = RSNAViTMIL(
        model_name="vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        in_chans=5,
        num_classes=len(full_dataset.label_cols),
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = epochs * len(train_loader)
    warmup_steps = len(train_loader)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    global_step = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        optimizer.zero_grad()

        for step, (images, targets, _) in enumerate(pbar):
            images = images.to(device)
            targets = targets.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = criterion(logits, targets) / 2

            loss.backward()

            if (step + 1) % 2 == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            scheduler.step()
            train_loss += loss.item() * 2
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item() * 2:.4f}"})

        avg_loss = train_loss / max(1, len(train_loader))

        model.eval()
        val_preds: List[np.ndarray] = []
        val_targets: List[np.ndarray] = []
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(images)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                val_preds.append(probs)
                val_targets.append(targets.numpy())

        if val_preds:
            y_pred = np.concatenate(val_preds, axis=0)
            y_true = np.concatenate(val_targets, axis=0)
            val_auc = compute_macro_auc(y_true, y_pred)
        else:
            val_auc = 0.5

        print(f"Epoch {epoch+1} finished - Train Loss: {avg_loss:.4f} - Val AUC: {val_auc:.4f}")

        ckpt_payload = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "val_auc": val_auc,
            "label_cols": full_dataset.label_cols,
        }
        torch.save(ckpt_payload, save_dir / "last.pt")

        if val_auc > best_auc or epoch == 0:
            best_auc = val_auc
            torch.save(ckpt_payload, save_dir / "best.pt")

    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gb = peak_bytes / (1024 ** 3)
        print(f"Peak VRAM: {peak_gb:.2f} GB ({peak_bytes} bytes)")
        if smoke:
            assert peak_gb < 4.0, f"Smoke test failed: Peak VRAM {peak_gb:.2f} GB exceeded 4GB!"

    print(f"Training completed. Checkpoints saved in {save_dir}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--labels", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_fn(
        fold=args.fold,
        smoke=args.smoke,
        epochs=args.epochs,
        batch_size=args.batch_size,
        labels_path=args.labels,
        out_dir=args.out,
    )
