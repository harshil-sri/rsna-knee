import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import CKPT_DIR, DATA_ROOT, SMOKE_FAKE_DIR
from src.dataset import RSNADataset
from src.model import RSNAViTMIL

def infer_fn(
    ckpt_path: str,
    smoke: bool = False,
    out_csv: str = "preds.csv",
    tta_flip: bool = True,
    batch_size: int = 4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_file = Path(ckpt_path)
    checkpoint = torch.load(ckpt_file, map_location=device)
    label_cols = checkpoint["label_cols"]

    model = RSNAViTMIL(
        model_name="vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        in_chans=5,
        num_classes=len(label_cols),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    data_dir = SMOKE_FAKE_DIR if smoke else Path(DATA_ROOT)
    windows_per_study = 8 if smoke else 42
    dataset = RSNADataset(
        data_root=data_dir,
        gold_only=False,
        windows_per_study=windows_per_study,
        label_cols=label_cols,
        is_synthetic=smoke,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_uids: List[str] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, _, uids in loader:
            images = images.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_orig = model(images)
                probs = torch.sigmoid(logits_orig)

                if tta_flip:
                    images_flip = torch.flip(images, dims=[-1])
                    logits_flip = model(images_flip)
                    probs_flip = torch.sigmoid(logits_flip)
                    probs = 0.5 * (probs + probs_flip)

            all_probs.append(probs.float().cpu().numpy())
            all_uids.extend(uids)

    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(label_cols)))

    df_out = pd.DataFrame({"StudyInstanceUID": all_uids})
    for idx, col in enumerate(label_cols):
        df_out[col] = probs_arr[:, idx]

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Predictions successfully written to {out_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=str(CKPT_DIR / "best.pt"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=str, default="preds.csv")
    parser.add_argument("--tta-flip", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    infer_fn(
        ckpt_path=args.ckpt,
        smoke=args.smoke,
        out_csv=args.out,
        tta_flip=args.tta_flip,
        batch_size=args.batch_size,
    )
