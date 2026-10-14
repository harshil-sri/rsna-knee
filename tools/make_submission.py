import argparse
from pathlib import Path
from typing import List
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import RSNADataset
from src.model import RSNAViTMIL

def make_submission(
    ckpt_path: str,
    out_csv: str,
    data_root: str,
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

    test_uids = pd.read_csv(Path(data_root) / "test.csv")["StudyInstanceUID"].tolist()
    test_labels_path = Path("/tmp/opencode/rsna_test_labels.csv")
    test_labels_path.parent.mkdir(parents=True, exist_ok=True)
    test_labels = pd.DataFrame(
        {"StudyInstanceUID": test_uids, "Report": ""}
    )
    for col in label_cols:
        test_labels[col] = np.nan
    test_labels.to_csv(test_labels_path, index=False)

    dataset = RSNADataset(
        data_root=data_root,
        gold_only=False,
        windows_per_study=8,
        label_cols=label_cols,
        train_csv=str(test_labels_path),
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_uids: List[str] = []
    all_probs: List[np.ndarray] = []

    with torch.no_grad():
        for images, _, uids in loader:
            images = images.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_orig = model(images)
                probs = torch.sigmoid(logits_orig)

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
    df_out.to_csv(out_path, index=False, float_format="%.4f")
    print(df_out.to_string(index=False))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/home/harshil/kaggle/rsna-knee/checkpoints/best_v2.pt")
    parser.add_argument("--out", type=str, default="/home/harshil/kaggle/rsna-knee/submission.csv")
    parser.add_argument("--data-root", type=str, default="/mnt/ssd/rsna-knee/data")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    make_submission(
        ckpt_path=args.ckpt,
        out_csv=args.out,
        data_root=args.data_root,
    )